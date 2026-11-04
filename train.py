"""Training and evaluation utilities (steps-based, identical for every experimental arm)."""

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def mse_on(model, X, Y, device, batch_size=2048):
    """Mean squared error of next-state prediction over a set of transitions."""
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        total += F.mse_loss(model(xb), yb, reduction="sum").item()
    return total / Y.numel()


def train_model(model, X, Y, X_val, Y_val, steps, lr, batch_size=64, seed=0, device=None,
                weight_decay=1e-4, eval_every=100, verbose=False):
    """Train for a fixed number of gradient steps (AdamW + cosine schedule).
    The checkpoint with the lowest validation MSE is restored at the end.
    Every experimental arm uses this same function, step budget and selection rule."""
    device = device or get_device()
    model.to(device)
    keep_on_gpu = X.numel() * 4 < 1.5e9                      # otherwise index on CPU, move batches
    X_t, Y_t = (X.to(device), Y.to(device)) if keep_on_gpu else (X, Y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    bs = min(batch_size, n)
    perm, pos = torch.randperm(n, generator=gen), 0
    best, best_state, history = float("inf"), None, []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=gen), 0
        idx = perm[pos:pos + bs]
        pos += bs
        if keep_on_gpu:
            xb, yb = X_t[idx.to(device)], Y_t[idx.to(device)]
        else:
            xb, yb = X_t[idx].to(device), Y_t[idx].to(device)

        model.train()
        loss = F.mse_loss(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % eval_every == 0 or step == steps:
            val = mse_on(model, X_val, Y_val, device)
            history.append((step, float(loss.item()), float(val)))
            if val < best:
                best, best_state = val, copy.deepcopy(model.state_dict())
            if verbose:
                print(f"  step {step:5d}/{steps} | train {loss.item():.3e} | val {val:.3e}")

    if best_state is not None:            # stays None only if validation was never finite
        model.load_state_dict(best_state)
    return model, {"best_val_mse": float(best), "history": history,
                   "train_seconds": time.perf_counter() - t0}


@torch.no_grad()
def evaluate_trajectories(model, U, device, batch_size=128):
    """Normalised RMSE ||pred - true||_2 / ||true||_2, averaged over samples.

    one_step: teacher-forced next-step error over all transitions.
    rollout:  autoregressive prediction of frames 2..T-1 starting from the TRUE frames 0 and 1.
    """
    model.eval()
    U = torch.as_tensor(U)
    n, T, N = U.shape
    one_step, rollout = [], []
    for i in range(0, n, batch_size):
        u = U[i:i + batch_size].to(device)
        b = u.shape[0]
        x = torch.stack([u[:, :-2], u[:, 1:-1]], dim=2).reshape(-1, 2, N)
        pred = model(x).reshape(b, T - 2, N)
        one_step.append(((pred - u[:, 2:]).norm(dim=-1) / (u[:, 2:].norm(dim=-1) + 1e-8)).cpu())

        prev, cur, outs = u[:, 0], u[:, 1], []
        for _ in range(T - 2):
            nxt = model(torch.stack([prev, cur], dim=1))
            outs.append(nxt)
            prev, cur = cur, nxt
        outs = torch.stack(outs, dim=1)
        rollout.append(((outs - u[:, 2:]).norm(dim=-1) / (u[:, 2:].norm(dim=-1) + 1e-8)).cpu())

    one_step, rollout = torch.cat(one_step), torch.cat(rollout)       # (n, T-2) each
    return {"one_step_nrmse": float(one_step.mean()),
            "rollout_nrmse_mean": float(rollout.mean()),
            "rollout_nrmse_final": float(rollout[:, -1].mean()),
            "rollout_curve": [float(v) for v in rollout.mean(dim=0)]}
