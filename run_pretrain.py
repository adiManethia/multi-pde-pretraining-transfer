"""Pretrain one FNO on a mixture of PDE families (never including Burgers).

    python run_pretrain.py --name pre_all --pdes heat advection reaction_diffusion
    python run_pretrain.py --name pre_heat --pdes heat
    python run_pretrain.py --name pre_all_n250 --pdes heat advection reaction_diffusion --n-per-pde 250
"""

import argparse
import json
import os

import torch

import data
from model import FNO1d, count_parameters
from train import evaluate_trajectories, get_device, set_seed, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pdes", nargs="+", default=list(data.PRETRAIN_PDES), choices=list(data.PRETRAIN_PDES))
    ap.add_argument("--n-per-pde", type=int, default=2000)
    ap.add_argument("--n-val-per-pde", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="ckpt")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cache-dir", default="data_cache")
    args = ap.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()
    set_seed(args.seed)

    # Data: seeds depend on the PDE (not on the subset chosen), so ablations share trajectories.
    Xs, Ys, Xv, Yv, val_sets = [], [], [], [], {}
    for pde in args.pdes:
        i = data.PRETRAIN_PDES.index(pde)
        tr = data.cached(pde, args.n_per_pde, 10_000 + 100 * i, args.cache_dir)
        va = data.cached(pde, args.n_val_per_pde, data.SEEDS["pretrain_val_offset"] + i, args.cache_dir)
        x, y = data.to_transitions(tr["u"])
        xv, yv = data.to_transitions(va["u"])
        Xs.append(x); Ys.append(y); Xv.append(xv); Yv.append(yv); val_sets[pde] = va["u"]
    X, Y, X_val, Y_val = torch.cat(Xs), torch.cat(Ys), torch.cat(Xv), torch.cat(Yv)

    model = FNO1d(args.width, args.modes, args.layers)
    print(f"[pretrain:{args.name}] PDEs {args.pdes} | {len(X)} train transitions | "
          f"{count_parameters(model):,} parameters | device {device}")
    model, info = train_model(model, X, Y, X_val, Y_val, steps=args.steps, lr=args.lr,
                              batch_size=args.batch_size, seed=args.seed, device=device,
                              eval_every=200, verbose=True)

    # Sanity checks: error on held-out trajectories of the SEEN families, and zero-shot Burgers.
    report = {pde: evaluate_trajectories(model, u, device) for pde, u in val_sets.items()}
    burgers = data.cached("burgers", 200, data.SEEDS["burgers_test"], args.cache_dir)
    report["burgers_zero_shot"] = evaluate_trajectories(model, burgers["u"], device)
    for k, v in report.items():
        print(f"  {k:22s} one-step nRMSE {v['one_step_nrmse']:.4f} | rollout(final) {v['rollout_nrmse_final']:.4f}")

    ckpt = {"state_dict": model.state_dict(),
            "config": {"width": args.width, "modes": args.modes, "layers": args.layers},
            "args": vars(args), "best_val_mse": info["best_val_mse"]}
    torch.save(ckpt, os.path.join(args.ckpt_dir, f"{args.name}.pt"))
    with open(os.path.join(args.results_dir, f"pretrain_{args.name}.json"), "w") as f:
        json.dump({"args": vars(args), "best_val_mse": info["best_val_mse"],
                   "train_seconds": info["train_seconds"], "report": report,
                   "history": info["history"]}, f, indent=2)
    print(f"Saved {args.ckpt_dir}/{args.name}.pt")


if __name__ == "__main__":
    main()
