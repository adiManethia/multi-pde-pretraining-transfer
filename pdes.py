"""
1D periodic PDE solvers on the unit interval [0, 1), plus random initial conditions.

Four families (all periodic, grid x_j = j / N):
  heat                u_t = nu u_xx                         exact in Fourier space
  advection           u_t + c u_x = 0                       exact in Fourier space
  reaction_diffusion  u_t = nu u_xx + r u (1 - u)           Strang splitting (exact sub-steps)
  burgers             u_t + u u_x = nu u_xx                 pseudo-spectral, integrating-factor RK4

Burgers is the HELD-OUT family used for transfer; the others are used for pretraining.
All solvers are vectorised over trajectories and return snapshots at the requested times.
"""

import numpy as np


def wavenumbers(N):
    """Angular wavenumbers k = 2*pi*m for the rfft modes m = 0..N/2 on the unit interval."""
    return 2.0 * np.pi * np.fft.rfftfreq(N, d=1.0 / N)


def random_smooth_ic(rng, n, N, kmax=4, alpha=1.0):
    """Random smooth periodic functions: sum_k a_k sin(2 pi k x + phi_k), a_k ~ N(0,1)/k^alpha.
    Each function is scaled to max |u| = 1."""
    x = np.arange(N) / N
    k = np.arange(1, kmax + 1)
    a = rng.normal(size=(n, kmax)) / k**alpha
    phi = rng.uniform(0, 2 * np.pi, size=(n, kmax))
    u = np.einsum("nk,nkx->nx", a, np.sin(2 * np.pi * k[None, :, None] * x[None, None, :] + phi[:, :, None]))
    return u / (np.abs(u).max(axis=1, keepdims=True) + 1e-12)


def solve_heat(u0, nu, t_out):
    """Exact solution: each Fourier mode decays as exp(-nu k^2 t)."""
    n, N = u0.shape
    k = wavenumbers(N)
    t = np.asarray(t_out)
    u_hat = np.fft.rfft(u0, axis=-1)
    decay = np.exp(-nu[:, None, None] * (k**2)[None, None, :] * t[None, :, None])
    return np.fft.irfft(u_hat[:, None, :] * decay, n=N, axis=-1)


def solve_advection(u0, c, t_out):
    """Exact solution: each Fourier mode is multiplied by exp(-i c k t) (a periodic shift)."""
    n, N = u0.shape
    k = wavenumbers(N)
    t = np.asarray(t_out)
    u_hat = np.fft.rfft(u0, axis=-1)
    phase = np.exp(-1j * c[:, None, None] * k[None, None, :] * t[None, :, None])
    return np.fft.irfft(u_hat[:, None, :] * phase, n=N, axis=-1)


def _logistic(u, r, dt):
    """Exact solution of u' = r u (1 - u) over a time dt (valid for u in [0, 1])."""
    return u / (u + (1.0 - u) * np.exp(-r * dt))


def solve_reaction_diffusion(u0, nu, r, t_out, dt=0.005):
    """Fisher-KPP u_t = nu u_xx + r u (1-u) with Strang splitting (reaction half, diffusion, reaction half).
    Requires u0 in [0, 1]."""
    n, N = u0.shape
    k = wavenumbers(N)
    t_out = np.asarray(t_out)
    steps = np.rint(np.diff(np.r_[0.0, t_out]) / dt).astype(int)
    assert np.allclose(steps * dt, np.diff(np.r_[0.0, t_out])), "output times must be multiples of dt"
    r_ = r[:, None]
    diff = np.exp(-nu[:, None] * (k**2)[None, :] * dt)
    u = u0.copy()
    out = np.empty((n, len(t_out), N))
    for j, s in enumerate(steps):
        for _ in range(s):
            u = _logistic(u, r_, dt / 2)
            u = np.fft.irfft(np.fft.rfft(u, axis=-1) * diff, n=N, axis=-1)
            u = np.clip(u, 0.0, 1.0)          # removes tiny Gibbs undershoot
            u = _logistic(u, r_, dt / 2)
        out[:, j] = u
    return out


def solve_burgers(u0, nu, t_out, dt=5e-4):
    """Viscous Burgers u_t + u u_x = nu u_xx.
    Pseudo-spectral (2/3 dealiasing) with integrating-factor RK4: the stiff diffusion term is
    integrated exactly, the nonlinear term with RK4."""
    n, N = u0.shape
    k = wavenumbers(N)
    m = np.arange(len(k))
    mask = (m <= N // 3).astype(float)
    ik = 1j * k
    t_out = np.asarray(t_out)
    steps = np.rint(np.diff(np.r_[0.0, t_out]) / dt).astype(int)
    assert np.allclose(steps * dt, np.diff(np.r_[0.0, t_out])), "output times must be multiples of dt"

    E = np.exp(-nu[:, None] * (k**2)[None, :] * dt)
    E2 = np.exp(-nu[:, None] * (k**2)[None, :] * dt / 2)

    def nonlin(v):
        u = np.fft.irfft(v * mask, n=N, axis=-1)
        return -0.5 * ik * np.fft.rfft(u * u, axis=-1) * mask

    v = np.fft.rfft(u0, axis=-1) * mask
    out = np.empty((n, len(t_out), N))
    for j, s in enumerate(steps):
        for _ in range(s):
            a = dt * nonlin(v)
            b = dt * nonlin(E2 * (v + a / 2))
            c = dt * nonlin(E2 * v + b / 2)
            d = dt * nonlin(E * v + E2 * c)
            v = E * v + (E * a + 2 * E2 * (b + c) + d) / 6
        out[:, j] = np.fft.irfft(v, n=N, axis=-1)
    return out
