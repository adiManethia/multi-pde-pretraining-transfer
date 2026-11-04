"""Dataset checks (NumPy only): shapes, determinism, and no overlap between splits."""

import numpy as np

import data


def test_generate_shapes_and_first_frame():
    for pde in ("heat", "advection", "reaction_diffusion", "burgers", "burgers_ood"):
        u = data.generate(pde, 8, 0)["u"]
        assert u.shape == (8, data.T_FRAMES, data.N_GRID) and u.dtype == np.float32
        assert np.isfinite(u).all()


def test_generation_is_deterministic_and_seed_dependent():
    a = data.generate("burgers", 6, 5)["u"]
    b = data.generate("burgers", 6, 5)["u"]
    c = data.generate("burgers", 6, 6)["u"]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_evaluation_sets_do_not_share_trajectories_with_the_training_pool():
    pool = data.generate("burgers", 64, data.SEEDS["burgers_pool"])["u"][:, 0]
    for name, pde in (("burgers_val", "burgers"), ("burgers_test", "burgers"),
                      ("burgers_ood_test", "burgers_ood")):
        other = data.generate(pde, 64, data.SEEDS[name])["u"][:, 0]
        dist = np.abs(pool[:, None, :] - other[None, :, :]).max(axis=-1)
        assert dist.min() > 1e-3          # no initial condition appears in both sets


def test_pretraining_families_exclude_burgers():
    assert "burgers" not in data.PRETRAIN_PDES


def test_ood_viscosity_is_below_the_fine_tuning_range():
    assert data.PARAM_RANGES["burgers_ood"]["nu"][1] <= data.PARAM_RANGES["burgers"]["nu"][0]


def test_advection_speed_sign_is_randomised():
    c = data.generate("advection", 200, 0)["params"]["c"]
    assert (c > 0).any() and (c < 0).any() and np.abs(c).min() >= 0.2
