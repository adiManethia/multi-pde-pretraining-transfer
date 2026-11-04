"""Fine-tune on the held-out PDE (viscous Burgers) with N trajectories, from scratch vs pretrained.

Fair-comparison protocol (identical for every arm):
  * same architecture, same step budget, same optimiser/schedule, same learning-rate grid
  * for each (arm, N, seed) the best learning rate is chosen on the SAME validation set
  * the training subset for a given (N, seed) is the same for every arm
  * the test sets (in-distribution and lower-viscosity OOD) are never used for any choice

    python run_transfer.py --ckpts pre_all
    python run_transfer.py --ckpts pre_all pre_heat pre_heat_advection pre_all_n250
"""

import argparse
import json
import os

import numpy as np
import torch

import data
from model import FNO1d
from train import evaluate_trajectories, get_device, set_seed, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["pre_all"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--cache-dir", default="data_cache")
    ap.add_argument("--out", default="results/transfer_results.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = get_device()

    ckpts = {}
    for name in args.ckpts:
        ckpts[name] = torch.load(os.path.join(args.ckpt_dir, f"{name}.pt"), map_location="cpu")
    configs = [c["config"] for c in ckpts.values()]
    config = configs[0] if configs else {"width": 64, "modes": 16, "layers": 4}
    assert all(c == config for c in configs), "all checkpoints must share one architecture"

    def build():
        return FNO1d(config["width"], config["modes"], config["layers"])

    pool_n = 1024            # FIXED pool size: the training subset for a given (N, seed) must not
                             # depend on which --sizes were requested in this invocation.
                             # (Only the first N of a seeded permutation are used.)
    assert max(args.sizes) <= pool_n, "sizes above 1024 need a larger fixed pool"
    pool = data.cached("burgers", pool_n, data.SEEDS["burgers_pool"], args.cache_dir)["u"]
    val = data.cached("burgers", args.n_val, data.SEEDS["burgers_val"], args.cache_dir)["u"]
    test = data.cached("burgers", 200, data.SEEDS["burgers_test"], args.cache_dir)["u"]
    ood = data.cached("burgers_ood", 200, data.SEEDS["burgers_ood_test"], args.cache_dir)["u"]
    X_val, Y_val = data.to_transitions(val)

    rows = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            rows = json.load(f)
    done = {(r["arm"], r["n_train"], r["seed"]) for r in rows}

    def save():
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)

    # Zero-shot (no fine-tuning) rows for each pretrained checkpoint
    for name, ck in ckpts.items():
        if (name, 0, 0) in done:
            continue
        m = build()
        m.load_state_dict(ck["state_dict"])
        m.to(device)
        rows.append({"arm": name, "n_train": 0, "seed": 0, "lr": None,
                     "test": evaluate_trajectories(m, test, device),
                     "ood": evaluate_trajectories(m, ood, device)})
        print(f"[zero-shot] {name}: test one-step {rows[-1]['test']['one_step_nrmse']:.4f}")
        save()

    arms = ["scratch"] + list(ckpts)
    for n in args.sizes:
        for seed in args.seeds:
            subset = np.random.default_rng(seed).permutation(pool_n)[:n]     # same for all arms
            X, Y = data.to_transitions(pool[subset])
            for arm in arms:
                if (arm, n, seed) in done:
                    continue
                best = None
                for lr in args.lrs:
                    set_seed(seed)
                    model = build()
                    if arm != "scratch":
                        model.load_state_dict(ckpts[arm]["state_dict"])
                    model, info = train_model(model, X, Y, X_val, Y_val, steps=args.steps, lr=lr,
                                              batch_size=args.batch_size, seed=seed, device=device)
                    if best is None or info["best_val_mse"] < best[0]["best_val_mse"]:
                        best = (info, lr, model)
                info, lr, model = best
                row = {"arm": arm, "n_train": n, "seed": seed, "lr": lr,
                       "best_val_mse": info["best_val_mse"], "train_seconds": info["train_seconds"],
                       "test": evaluate_trajectories(model, test, device),
                       "ood": evaluate_trajectories(model, ood, device)}
                rows.append(row)
                save()
                print(f"[N={n:4d} seed {seed}] {arm:18s} lr {lr:.0e} | test one-step "
                      f"{row['test']['one_step_nrmse']:.4f} | rollout(final) "
                      f"{row['test']['rollout_nrmse_final']:.4f} | OOD rollout(final) "
                      f"{row['ood']['rollout_nrmse_final']:.4f}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
