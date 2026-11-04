"""Plots and tables from results/transfer_results.json (mean +/- std over seeds).

    python plot_results.py --arms scratch pre_all --out-prefix results/main
    python plot_results.py --arms scratch pre_heat pre_heat_advection pre_all --out-prefix results/diversity
    python plot_results.py --arms scratch pre_all_n250 pre_all_n1000 pre_all pre_all_n4000 --out-prefix results/scale
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"one_step_nrmse": "One-step nRMSE",
          "rollout_nrmse_final": "Rollout nRMSE (final step)",
          "rollout_nrmse_mean": "Rollout nRMSE (mean over steps)"}
SPLITS = {"test": "in-distribution test", "ood": "OOD test (lower viscosity)"}


def aggregate(rows, split, metric):
    groups = defaultdict(list)
    for r in rows:
        if r["n_train"] > 0:
            groups[(r["arm"], r["n_train"])].append(r[split][metric])
    return groups


def make_plot(rows, arms, split, metric, path):
    agg = aggregate(rows, split, metric)
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=150)
    for arm in arms:
        pts = sorted((n, np.mean(v), np.std(v)) for (a, n), v in agg.items() if a == arm)
        if not pts:
            continue
        ns, mean, std = map(np.array, zip(*pts))
        ax.errorbar(ns, mean, yerr=std, marker="o", capsize=3, label=arm)
    for r in rows:
        if r["n_train"] == 0 and r["arm"] in arms:
            ax.axhline(r[split][metric], ls=":", lw=1, alpha=0.7, label=f"{r['arm']} (zero-shot)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Burgers fine-tuning trajectories N")
    ax.set_ylabel(LABELS[metric])
    ax.set_title(f"{LABELS[metric]} - {SPLITS[split]}", fontsize=10)
    ax.grid(alpha=0.4, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_table(rows, arms, split, metric):
    agg = aggregate(rows, split, metric)
    sizes = sorted({n for (_, n) in agg})
    print(f"\n{LABELS[metric]} - {SPLITS[split]} (mean +/- std over seeds)")
    print("| arm | " + " | ".join(f"N={n}" for n in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for arm in arms:
        cells = []
        for n in sizes:
            v = agg.get((arm, n))
            cells.append(f"{np.mean(v):.4f} +/- {np.std(v):.4f} (n={len(v)})" if v else "-")
        print(f"| {arm} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/transfer_results.json")
    ap.add_argument("--arms", nargs="+", default=["scratch", "pre_all"])
    ap.add_argument("--out-prefix", default="results/transfer")
    ap.add_argument("--max-n", type=int, default=256,
                    help="ignore results with more fine-tuning trajectories than this")
    args = ap.parse_args()
    with open(args.results) as f:
        rows = json.load(f)
    rows = [r for r in rows if r["n_train"] <= args.max_n]     # keeps zero-shot rows (n_train = 0)
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    for split in SPLITS:
        for metric in ("rollout_nrmse_final", "one_step_nrmse"):
            make_plot(rows, args.arms, split, metric, f"{args.out_prefix}_{split}_{metric}.png")
            print_table(rows, args.arms, split, metric)
    print(f"\nSaved figures with prefix {args.out_prefix}")


if __name__ == "__main__":
    main()
