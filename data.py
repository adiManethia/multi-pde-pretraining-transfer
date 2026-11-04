"""
Dataset generation for the multi-PDE pretraining / transfer study.

Every trajectory has T_FRAMES = 21 snapshots at t = 0, 0.05, ..., 1.0 on a 128-point grid.
Datasets are generated from explicit seeds, and train / validation / test sets use DIFFERENT
seeds, so no trajectory is shared between splits. No data statistics are used for
normalisation (all fields are O(1) by construction), so nothing can leak through preprocessing.
"""

import os

import numpy as np

import pdes

N_GRID = 128
T_FRAMES = 21
DT_OUT = 0.05
T_OUT = np.arange(1, T_FRAMES) * DT_OUT          # snapshots after t = 0

# Parameter ranges. Burgers is the held-out family; "burgers_ood" has LOWER viscosity
# (sharper shocks) than anything used for fine-tuning.
PARAM_RANGES = {
    "heat": {"nu": (0.002, 0.02)},
    "advection": {"c": (0.2, 1.0)},                       # sign is randomised
    "reaction_diffusion": {"nu": (0.002, 0.01), "r": (1.0, 4.0)},
    "burgers": {"nu": (0.03, 0.08)},
    "burgers_ood": {"nu": (0.01, 0.03)},
}
PRETRAIN_PDES = ("heat", "advection", "reaction_diffusion")

# Fixed seeds for the evaluation sets (never used for training)
SEEDS = {"burgers_pool": 1000, "burgers_val": 9003, "burgers_test": 9001, "burgers_ood_test": 9002,
         "pretrain_val_offset": 5000}


def generate(pde, n, seed, param_ranges=None):
    """Return {'u': (n, T_FRAMES, N_GRID) float32, 'params': {name: (n,)}}."""
    ranges = dict(PARAM_RANGES[pde] if param_ranges is None else param_ranges)
    rng = np.random.default_rng(seed)
    params = {name: rng.uniform(lo, hi, size=n) for name, (lo, hi) in ranges.items()}
    ic = pdes.random_smooth_ic(rng, n, N_GRID)

    if pde == "heat":
        u0 = ic * rng.uniform(0.5, 1.0, (n, 1))
        traj = pdes.solve_heat(u0, params["nu"], T_OUT)
    elif pde == "advection":
        u0 = ic * rng.uniform(0.5, 1.0, (n, 1))
        params["c"] = params["c"] * rng.choice([-1.0, 1.0], size=n)
        traj = pdes.solve_advection(u0, params["c"], T_OUT)
    elif pde == "reaction_diffusion":
        u0 = 0.5 * (1.0 + 0.9 * ic)                      # values in [0.05, 0.95]
        traj = pdes.solve_reaction_diffusion(u0, params["nu"], params["r"], T_OUT)
    elif pde in ("burgers", "burgers_ood"):
        u0 = ic * rng.uniform(0.5, 1.0, (n, 1))
        traj = pdes.solve_burgers(u0, params["nu"], T_OUT)
    else:
        raise ValueError(pde)

    u = np.concatenate([u0[:, None, :], traj], axis=1).astype(np.float32)
    assert u.shape == (n, T_FRAMES, N_GRID) and np.isfinite(u).all()
    return {"u": u, "params": params}


def cached(pde, n, seed, cache_dir="data_cache", param_ranges=None):
    """generate() with an on-disk cache keyed by (pde, n, seed, ranges)."""
    os.makedirs(cache_dir, exist_ok=True)
    tag = "" if param_ranges is None else "_" + "_".join(f"{k}{lo}-{hi}" for k, (lo, hi) in sorted(param_ranges.items()))
    path = os.path.join(cache_dir, f"{pde}_n{n}_s{seed}{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {"u": z["u"], "params": {k[2:]: z[k] for k in z.files if k.startswith("p_")}}
    d = generate(pde, n, seed, param_ranges)
    np.savez(path, u=d["u"], **{f"p_{k}": v for k, v in d["params"].items()})
    return d


def to_transitions(u):
    """Turn trajectories (n, T, N) into supervised triples.
    inputs  (n*(T-2), 2, N): [u_{t-1}, u_t]   targets (n*(T-2), N): u_{t+1}"""
    import torch

    u = torch.as_tensor(u)
    prev, cur, nxt = u[:, :-2], u[:, 1:-1], u[:, 2:]
    x = torch.stack([prev, cur], dim=2).reshape(-1, 2, u.shape[-1])
    y = nxt.reshape(-1, u.shape[-1])
    return x.contiguous(), y.contiguous()
