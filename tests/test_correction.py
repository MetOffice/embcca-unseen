# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""Tests for the EMBCCA bias-adjustment transform."""

import numpy as np
import pytest

from embcca import correct, correct_area_mean, correct_gridded
from embcca.correction import _align, _covariance_eigen, _standardise

N_YEARS = 40
N_MEMBERS = 25
N_VARS = 2


@pytest.fixture
def obs():
    """Observed data with distinctly different scales per variable."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(N_YEARS, N_VARS)) * [3.0, 0.5] + [10.0, 2.0]


@pytest.fixture
def mod():
    """Model data whose spread deliberately differs from the observed."""
    rng = np.random.default_rng(11)
    return rng.normal(size=(N_YEARS, N_MEMBERS, N_VARS)) * [5.0, 2.0] + [-4.0, 8.0]


def as_grid(mod_or_obs, ensemble):
    """Insert singleton lon/lat axes so a block can be fed to the gridded form."""
    if ensemble:
        return mod_or_obs[:, :, None, None, :]
    return mod_or_obs[:, None, None, :]


# ---------------------------------------------------------------------------
# _standardise
# ---------------------------------------------------------------------------


def test_standardise_produces_zero_mean_unit_variance():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 3)) * [2.0, 7.0, 0.1] + [1.0, -3.0, 40.0]

    standardised, mean, std = _standardise(values)

    assert np.allclose(standardised.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(standardised.std(axis=0), 1.0, atol=1e-12)
    assert np.allclose(mean, values.mean(axis=0))
    assert np.allclose(std, values.std(axis=0))


def test_standardise_without_eps_does_not_floor_std():
    values = np.column_stack([np.arange(10.0), np.full(10, 5.0)])

    with np.errstate(invalid="ignore"):
        standardised, _, std = _standardise(values)

    assert std[1] == 0.0
    assert not np.isfinite(standardised[:, 1]).any()


def test_standardise_floors_std_at_eps_and_reports_the_floored_value():
    values = np.column_stack([np.arange(10.0), np.full(10, 5.0)])
    eps = 1e-6

    standardised, _, std = _standardise(values, eps)

    assert std[1] == eps
    assert np.isfinite(standardised).all()


def test_standardise_leaves_healthy_std_untouched_when_eps_given():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(30, 2))

    _, _, floored = _standardise(values, 1e-6)
    _, _, unfloored = _standardise(values)

    assert np.array_equal(floored, unfloored)


# ---------------------------------------------------------------------------
# _covariance_eigen
# ---------------------------------------------------------------------------


def test_covariance_eigen_returns_ascending_orthonormal_decomposition():
    rng = np.random.default_rng(2)
    standardised, _, _ = _standardise(rng.normal(size=(60, 3)))

    eigenvalues, eigenvectors = _covariance_eigen(standardised, 1e-6)

    assert np.all(np.diff(eigenvalues) >= 0), "eigenvalues should be ascending"
    assert np.allclose(eigenvectors.T @ eigenvectors, np.eye(3), atol=1e-12)


def test_covariance_eigen_reconstructs_the_covariance():
    rng = np.random.default_rng(3)
    standardised, _, _ = _standardise(rng.normal(size=(60, 3)))

    eigenvalues, eigenvectors = _covariance_eigen(standardised, 1e-6)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    assert np.allclose(reconstructed, np.cov(standardised, rowvar=False), atol=1e-12)


def test_covariance_eigen_floors_eigenvalues_at_eps():
    # Two perfectly correlated variables -> one eigenvalue is exactly zero.
    column = np.linspace(0.0, 1.0, 30)
    standardised, _, _ = _standardise(np.column_stack([column, column]))
    eps = 1e-6

    eigenvalues, _ = _covariance_eigen(standardised, eps)

    assert eigenvalues.min() == eps
    assert np.isfinite(1.0 / np.sqrt(eigenvalues)).all()


# ---------------------------------------------------------------------------
# _align
# ---------------------------------------------------------------------------


def test_align_gives_the_block_the_observed_covariance():
    rng = np.random.default_rng(4)
    obs_st, _, _ = _standardise(rng.normal(size=(50, 2)) * [4.0, 0.2])
    mod_st, _, _ = _standardise(rng.normal(size=(50, 2)) * [0.5, 9.0])
    obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, 1e-6)
    mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, 1e-6)

    aligned = _align(
        mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors
    )

    assert np.allclose(
        np.cov(aligned, rowvar=False), np.cov(obs_st, rowvar=False), atol=1e-12
    )


def test_align_is_a_no_op_when_model_and_observed_agree():
    rng = np.random.default_rng(5)
    standardised, _, _ = _standardise(rng.normal(size=(50, 2)))
    eigenvalues, eigenvectors = _covariance_eigen(standardised, 1e-6)

    aligned = _align(standardised, eigenvalues, eigenvectors, eigenvalues, eigenvectors)

    assert np.allclose(aligned, standardised, atol=1e-12)


# ---------------------------------------------------------------------------
# Defining properties of the method
# ---------------------------------------------------------------------------


def test_corrected_data_takes_the_observed_mean(obs, mod):
    corrected = correct_area_mean(mod, obs)

    for member in range(N_MEMBERS):
        assert np.allclose(corrected[:, member, :].mean(axis=0), obs.mean(axis=0), atol=1e-12)


def test_corrected_data_takes_the_observed_correlation(obs, mod):
    corrected = correct_area_mean(mod, obs)
    observed_r = np.corrcoef(obs, rowvar=False)[0, 1]

    for member in range(N_MEMBERS):
        member_r = np.corrcoef(corrected[:, member, :], rowvar=False)[0, 1]
        assert member_r == pytest.approx(observed_r, abs=1e-12)


def test_corrected_data_keeps_the_model_variance(obs, mod):
    """The defining property: variance comes from the model, not the observations.

    This is what distinguishes EMBCCA from methods that transfer the observed
    variance too, and what makes it usable for UNSEEN.
    """
    corrected = correct_area_mean(mod, obs)

    assert np.allclose(corrected.std(axis=0), mod.std(axis=0), rtol=1e-12)


def test_corrected_variance_is_not_the_observed_variance(obs, mod):
    """Guards the property above against a change that would silently invert it."""
    corrected = correct_area_mean(mod, obs)

    assert not np.allclose(corrected.std(axis=0), obs.std(axis=0), rtol=0.1)


def test_correcting_observations_against_themselves_is_the_identity(obs):
    corrected = correct_area_mean(obs[:, None, :].copy(), obs)

    assert np.allclose(corrected[:, 0, :], obs, atol=1e-12)


def test_properties_hold_for_more_than_two_variables():
    rng = np.random.default_rng(6)
    obs = rng.normal(size=(N_YEARS, 3)) * [1.0, 6.0, 0.3]
    mod = rng.normal(size=(N_YEARS, 10, 3)) * [4.0, 0.5, 2.0]

    corrected = correct_area_mean(mod, obs)

    assert np.allclose(corrected.mean(axis=0), obs.mean(axis=0), atol=1e-12)
    assert np.allclose(corrected.std(axis=0), mod.std(axis=0), rtol=1e-12)
    assert np.allclose(
        np.corrcoef(corrected[:, 0, :], rowvar=False),
        np.corrcoef(obs, rowvar=False),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Agreement between the two forms
# ---------------------------------------------------------------------------


def test_single_cell_grid_matches_the_area_mean_form(obs, mod):
    """Locks the two implementations together.

    A grid of one cell is the same problem as the area-mean case, so the two
    code paths must not be allowed to drift apart.
    """
    from_area_mean = correct_area_mean(mod, obs)
    from_grid = correct_gridded(as_grid(mod, ensemble=True), as_grid(obs, ensemble=False))

    assert np.array_equal(from_grid[:, :, 0, 0, :], from_area_mean)


def test_grid_cells_are_corrected_independently(obs, mod):
    """A cell's result must not depend on what its neighbours contain."""
    rng = np.random.default_rng(12)
    grid_mod = np.repeat(np.repeat(as_grid(mod, ensemble=True), 2, axis=2), 2, axis=3)
    grid_obs = np.repeat(np.repeat(as_grid(obs, ensemble=False), 2, axis=1), 2, axis=2)
    # Perturb one cell only.
    grid_mod[:, :, 1, 1, :] += rng.normal(size=(N_YEARS, N_MEMBERS, N_VARS)) * 20.0

    corrected = correct_gridded(grid_mod, grid_obs)

    assert np.array_equal(corrected[:, :, 0, 0, :], corrected[:, :, 0, 1, :])
    assert not np.allclose(corrected[:, :, 0, 0, :], corrected[:, :, 1, 1, :])


# ---------------------------------------------------------------------------
# Array contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_matches_input_shape_and_dtype(obs, mod, dtype):
    corrected = correct_area_mean(mod.astype(dtype), obs.astype(dtype))

    assert corrected.shape == mod.shape
    assert corrected.dtype == dtype


def test_gridded_output_matches_input_shape_and_dtype(obs, mod):
    grid_mod = as_grid(mod, ensemble=True)

    corrected = correct_gridded(grid_mod, as_grid(obs, ensemble=False))

    assert corrected.shape == grid_mod.shape
    assert corrected.dtype == grid_mod.dtype


def test_inputs_are_not_modified(obs, mod):
    obs_before, mod_before = obs.copy(), mod.copy()

    correct_area_mean(mod, obs)

    assert np.array_equal(obs, obs_before)
    assert np.array_equal(mod, mod_before)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_correct_dispatches_three_dimensional_input_to_the_area_mean_form(obs, mod):
    assert np.array_equal(correct(mod, obs), correct_area_mean(mod, obs))


def test_correct_dispatches_five_dimensional_input_to_the_gridded_form(obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(correct(grid_mod, grid_obs), correct_gridded(grid_mod, grid_obs))


def test_correct_forwards_eps_to_the_gridded_form(obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(
        correct(grid_mod, grid_obs, eps=1e-3), correct_gridded(grid_mod, grid_obs, eps=1e-3)
    )


def test_correct_rejects_eps_for_area_mean_input(obs, mod):
    with pytest.raises(TypeError):
        correct(mod, obs, eps=1e-3)


@pytest.mark.parametrize("ndim", [1, 2, 4, 6])
def test_correct_rejects_unsupported_dimensions(ndim):
    shaped = np.ones((2,) * ndim)

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        correct(shaped, shaped)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_gridded_handles_a_cell_with_no_variance(obs, mod):
    """eps keeps a constant cell finite rather than dividing by zero."""
    grid_mod = as_grid(mod, ensemble=True).copy()
    grid_obs = as_grid(obs, ensemble=False).copy()
    grid_mod[:, :, 0, 0, 1] = 3.0

    corrected = correct_gridded(grid_mod, grid_obs)

    assert np.isfinite(corrected).all()


def test_gridded_handles_perfectly_correlated_variables(obs):
    """A singular covariance is regularised by the eigenvalue floor."""
    column = np.linspace(0.0, 1.0, N_YEARS)
    grid_mod = np.stack([column, column], axis=-1)[:, None, None, None, :]
    grid_obs = as_grid(obs, ensemble=False)

    corrected = correct_gridded(grid_mod, grid_obs)

    assert np.isfinite(corrected).all()
