"""Solver verification (NumPy only). Thresholds are set from measured errors with wide margins."""

import numpy as np

import pdes

N = 128
X = np.arange(N) / N


def test_heat_matches_exact_sine_decay():
    t = np.array([0.0, 0.5, 1.0])
    u = pdes.solve_heat(np.sin(2 * np.pi * X)[None], np.array([0.01]), t)
    exact = np.exp(-0.01 * (2 * np.pi) ** 2 * t)[:, None] * np.sin(2 * np.pi * X)[None]
    assert np.abs(u[0] - exact).max() < 1e-12


def test_advection_is_an_exact_shift():
    t = np.array([0.0, 0.5, 1.0])
    u = pdes.solve_advection(np.sin(2 * np.pi * X)[None], np.array([0.7]), t)
    exact = np.sin(2 * np.pi * (X[None] - 0.7 * t[:, None]))
    assert np.abs(u[0] - exact).max() < 1e-12


def test_reaction_diffusion_constant_state_follows_logistic_ode():
    t = np.arange(1, 21) * 0.05
    u = pdes.solve_reaction_diffusion(np.full((1, N), 0.2), np.array([0.005]), np.array([3.0]), t)
    exact = 0.2 / (0.2 + 0.8 * np.exp(-3.0 * t))
    assert np.abs(u[0, :, 0] - exact).max() < 1e-10
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_burgers_conserves_mean_and_dissipates_energy():
    rng = np.random.default_rng(0)
    u0 = pdes.random_smooth_ic(rng, 20, N) * rng.uniform(0.5, 1.0, (20, 1))
    u = pdes.solve_burgers(u0, np.full(20, 0.03), np.arange(1, 21) * 0.05)
    assert np.isfinite(u).all()
    assert np.abs(u.mean(-1) - u0.mean(-1)[:, None]).max() < 1e-10
    energy = np.concatenate([(u0**2).mean(-1)[:, None], (u**2).mean(-1)], axis=1)
    assert (np.diff(energy, axis=1) <= 1e-6).all()


def test_burgers_is_resolved_on_128_points_for_the_viscosity_range_used():
    """N=128 vs N=512 reference (measured max relative error ~2e-5 at nu=0.01)."""
    rng = np.random.default_rng(1)
    for nu in (0.01, 0.03, 0.08):
        u0 = pdes.random_smooth_ic(rng, 10, 512) * rng.uniform(0.5, 1.0, (10, 1))
        t = np.arange(1, 21) * 0.05
        ref = pdes.solve_burgers(u0, np.full(10, nu), t)[:, :, ::4]
        lo = pdes.solve_burgers(u0[:, ::4], np.full(10, nu), t)
        err = np.linalg.norm(lo - ref, axis=-1) / (np.linalg.norm(ref, axis=-1) + 1e-12)
        assert err.max() < 1e-3


def test_random_ic_shape_and_normalisation():
    ic = pdes.random_smooth_ic(np.random.default_rng(0), 50, N)
    assert ic.shape == (50, N)
    assert np.allclose(np.abs(ic).max(axis=1), 1.0)
