# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""Tests for the EMBCCA bias-adjustment transform."""

import numpy as np
import pytest

from embcca import (
    bias_adjust,
    bias_adjust_area_mean,
    bias_adjust_area_mean_unseen,
    bias_adjust_gridded,
    bias_adjust_gridded_unseen,
    bias_adjust_unseen,
)
from embcca.adjustment import _align, _covariance_eigen, _standardise

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
    corrected = bias_adjust_area_mean_unseen(mod, obs)

    for member in range(N_MEMBERS):
        assert np.allclose(
            corrected[:, member, :].mean(axis=0), obs.mean(axis=0), atol=1e-12
        )


def test_corrected_data_takes_the_observed_correlation(obs, mod):
    corrected = bias_adjust_area_mean_unseen(mod, obs)
    observed_r = np.corrcoef(obs, rowvar=False)[0, 1]

    for member in range(N_MEMBERS):
        member_r = np.corrcoef(corrected[:, member, :], rowvar=False)[0, 1]
        assert member_r == pytest.approx(observed_r, abs=1e-12)


def test_corrected_data_keeps_the_model_variance(obs, mod):
    """The defining property: variance comes from the model, not the observations.

    This is what distinguishes EMBCCA from methods that transfer the observed
    variance too, and what makes it usable for UNSEEN.
    """
    corrected = bias_adjust_area_mean_unseen(mod, obs)

    assert np.allclose(corrected.std(axis=0), mod.std(axis=0), rtol=1e-12)


def test_corrected_variance_is_not_the_observed_variance(obs, mod):
    """Guards the property above against a change that would silently invert it."""
    corrected = bias_adjust_area_mean_unseen(mod, obs)

    assert not np.allclose(corrected.std(axis=0), obs.std(axis=0), rtol=0.1)


def test_correcting_observations_against_themselves_is_the_identity(obs):
    corrected = bias_adjust_area_mean_unseen(obs[:, None, :].copy(), obs)

    assert np.allclose(corrected[:, 0, :], obs, atol=1e-12)


def test_properties_hold_for_more_than_two_variables():
    rng = np.random.default_rng(6)
    obs = rng.normal(size=(N_YEARS, 3)) * [1.0, 6.0, 0.3]
    mod = rng.normal(size=(N_YEARS, 10, 3)) * [4.0, 0.5, 2.0]

    corrected = bias_adjust_area_mean_unseen(mod, obs)

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
    from_area_mean = bias_adjust_area_mean_unseen(mod, obs)
    from_grid = bias_adjust_gridded_unseen(
        as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)
    )

    assert np.array_equal(from_grid[:, :, 0, 0, :], from_area_mean)


def test_grid_cells_are_corrected_independently(obs, mod):
    """A cell's result must not depend on what its neighbours contain."""
    rng = np.random.default_rng(12)
    grid_mod = np.repeat(np.repeat(as_grid(mod, ensemble=True), 2, axis=2), 2, axis=3)
    grid_obs = np.repeat(np.repeat(as_grid(obs, ensemble=False), 2, axis=1), 2, axis=2)
    # Perturb one cell only.
    grid_mod[:, :, 1, 1, :] += rng.normal(size=(N_YEARS, N_MEMBERS, N_VARS)) * 20.0

    corrected = bias_adjust_gridded_unseen(grid_mod, grid_obs)

    assert np.array_equal(corrected[:, :, 0, 0, :], corrected[:, :, 0, 1, :])
    assert not np.allclose(corrected[:, :, 0, 0, :], corrected[:, :, 1, 1, :])


# ---------------------------------------------------------------------------
# Array contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_output_matches_input_shape_and_dtype(obs, mod, dtype):
    corrected = bias_adjust_area_mean_unseen(mod.astype(dtype), obs.astype(dtype))

    assert corrected.shape == mod.shape
    assert corrected.dtype == dtype


def test_gridded_output_matches_input_shape_and_dtype(obs, mod):
    grid_mod = as_grid(mod, ensemble=True)

    corrected = bias_adjust_gridded_unseen(grid_mod, as_grid(obs, ensemble=False))

    assert corrected.shape == grid_mod.shape
    assert corrected.dtype == grid_mod.dtype


def test_inputs_are_not_modified(obs, mod):
    obs_before, mod_before = obs.copy(), mod.copy()

    bias_adjust_area_mean_unseen(mod, obs)

    assert np.array_equal(obs, obs_before)
    assert np.array_equal(mod, mod_before)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_bias_adjust_dispatches_three_dimensional_input_to_the_area_mean_form(obs, mod):
    assert np.array_equal(bias_adjust_unseen(mod, obs), bias_adjust_area_mean_unseen(mod, obs))


def test_bias_adjust_dispatches_five_dimensional_input_to_the_gridded_form(obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(
        bias_adjust_unseen(grid_mod, grid_obs), bias_adjust_gridded_unseen(grid_mod, grid_obs)
    )


def test_bias_adjust_forwards_eps_to_the_gridded_form(obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(
        bias_adjust_unseen(grid_mod, grid_obs, eps=1e-3),
        bias_adjust_gridded_unseen(grid_mod, grid_obs, eps=1e-3),
    )


def test_bias_adjust_forwards_eps_to_the_area_mean_form(obs, mod):
    assert np.array_equal(
        bias_adjust_unseen(mod, obs, eps=1e-3), bias_adjust_area_mean_unseen(mod, obs, eps=1e-3)
    )


@pytest.mark.parametrize("ndim", [1, 6, 7])
def test_bias_adjust_rejects_unsupported_dimensions(ndim):
    shaped = np.ones((2,) * ndim)

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        bias_adjust_unseen(shaped, shaped)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_gridded_handles_a_cell_with_no_variance(obs, mod):
    """eps keeps a constant cell finite rather than dividing by zero."""
    grid_mod = as_grid(mod, ensemble=True).copy()
    grid_obs = as_grid(obs, ensemble=False).copy()
    grid_mod[:, :, 0, 0, 1] = 3.0

    corrected = bias_adjust_gridded_unseen(grid_mod, grid_obs)

    assert np.isfinite(corrected).all()


def test_gridded_handles_perfectly_correlated_variables(obs):
    """A singular covariance is regularised by the eigenvalue floor."""
    column = np.linspace(0.0, 1.0, N_YEARS)
    grid_mod = np.stack([column, column], axis=-1)[:, None, None, None, :]
    grid_obs = as_grid(obs, ensemble=False)

    corrected = bias_adjust_gridded_unseen(grid_mod, grid_obs)

    assert np.isfinite(corrected).all()


# ---------------------------------------------------------------------------
# The two variants
# ---------------------------------------------------------------------------

VARIANTS = [
    pytest.param(
        bias_adjust_unseen,
        bias_adjust_area_mean_unseen,
        bias_adjust_gridded_unseen,
        id="unseen",
    ),
    pytest.param(bias_adjust, bias_adjust_area_mean, bias_adjust_gridded, id="full"),
]


def test_full_variant_takes_the_observed_standard_deviation(obs, mod):
    """bias_adjust transfers the observed spread, unlike bias_adjust_unseen."""
    corrected = bias_adjust_area_mean(mod, obs)

    assert np.allclose(corrected.std(axis=0), obs.std(axis=0), rtol=1e-12)


def test_full_variant_does_not_keep_the_model_standard_deviation(obs, mod):
    corrected = bias_adjust_area_mean(mod, obs)

    assert not np.allclose(corrected.std(axis=0), mod.std(axis=0), rtol=0.1)


def test_the_two_variants_differ_when_the_spreads_differ(obs, mod):
    """They coincide only where model and observed spread already agree."""
    assert not np.allclose(
        bias_adjust_area_mean(mod, obs), bias_adjust_area_mean_unseen(mod, obs)
    )


def test_variants_agree_when_model_and_observed_spread_match(obs):
    """The two rescalings are the same operation when mod_std == obs_std."""
    single = obs[:, None, :].copy()

    assert np.allclose(
        bias_adjust_area_mean(single, obs), bias_adjust_area_mean_unseen(single, obs), atol=1e-12
    )


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_both_variants_take_the_observed_mean(dispatch, area_mean, gridded, obs, mod):
    corrected = area_mean(mod, obs)

    assert np.allclose(corrected.mean(axis=0), obs.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_both_variants_take_the_observed_correlation(dispatch, area_mean, gridded, obs, mod):
    corrected = area_mean(mod, obs)
    observed_r = np.corrcoef(obs, rowvar=False)[0, 1]

    for member in range(N_MEMBERS):
        member_r = np.corrcoef(corrected[:, member, :], rowvar=False)[0, 1]
        assert member_r == pytest.approx(observed_r, abs=1e-12)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_both_variants_dispatch_on_layout(dispatch, area_mean, gridded, obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(dispatch(mod, obs), area_mean(mod, obs))
    assert np.array_equal(dispatch(grid_mod, grid_obs), gridded(grid_mod, grid_obs))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_both_variants_reject_unsupported_dimensions(dispatch, area_mean, gridded):
    shaped = np.ones((2,) * 6)

    with pytest.raises(ValueError, match="Unsupported dimensions"):
        dispatch(shaped, shaped)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_single_cell_grid_matches_area_mean_for_both_variants(
    dispatch, area_mean, gridded, obs, mod
):
    from_area_mean = area_mean(mod, obs)
    from_grid = gridded(as_grid(mod, ensemble=True), as_grid(obs, ensemble=False))

    assert np.array_equal(from_grid[:, :, 0, 0, :], from_area_mean)


# ---------------------------------------------------------------------------
# mod_future
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_output_matches_the_future_shape_not_the_calibration_shape(
    dispatch, area_mean, gridded, obs, mod
):
    """The result corresponds to the data actually adjusted."""
    future = mod[: N_YEARS // 2]

    corrected = area_mean(mod, obs, future)

    assert corrected.shape == future.shape
    assert corrected.shape != mod.shape


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_gridded_output_matches_the_future_shape(dispatch, area_mean, gridded, obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)
    future = grid_mod[: N_YEARS // 2]

    corrected = gridded(grid_mod, grid_obs, future)

    assert corrected.shape == future.shape


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_passing_the_calibration_data_as_future_matches_omitting_it(
    dispatch, area_mean, gridded, obs, mod
):
    """Standardising the calibration block with its own statistics is what the
    no-future path already does, so the two must agree exactly."""
    assert np.array_equal(area_mean(mod, obs, mod), area_mean(mod, obs))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_gridded_calibration_as_future_matches_omitting_it(
    dispatch, area_mean, gridded, obs, mod
):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(gridded(grid_mod, grid_obs, grid_mod), gridded(grid_mod, grid_obs))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_future_is_standardised_with_calibration_statistics(
    dispatch, area_mean, gridded, obs, mod
):
    """A shifted future period must stay shifted after adjustment.

    If the future block were standardised with its own mean, the offset would
    be absorbed and the two results would coincide.
    """
    shifted = mod + 100.0

    assert not np.allclose(area_mean(mod, obs, shifted), area_mean(mod, obs))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_mod_future_is_forwarded_by_the_dispatcher(dispatch, area_mean, gridded, obs, mod):
    future = mod[: N_YEARS // 2]

    assert np.array_equal(dispatch(mod, obs, future), area_mean(mod, obs, future))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_future_inputs_are_not_modified(dispatch, area_mean, gridded, obs, mod):
    future = (mod + 3.0)[: N_YEARS // 2]
    before = future.copy()

    area_mean(mod, obs, future)

    assert np.array_equal(future, before)


# ---------------------------------------------------------------------------
# mod_future shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
@pytest.mark.parametrize("n_future_members", [N_MEMBERS + 2, N_MEMBERS - 2])
def test_future_with_wrong_member_count_is_rejected(
    dispatch, area_mean, gridded, obs, mod, n_future_members
):
    """Too many members used to return uninitialised memory silently."""
    future = np.zeros((N_YEARS // 2, n_future_members, N_VARS))

    with pytest.raises(ValueError, match="Only the leading time axis may differ"):
        area_mean(mod, obs, future)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_future_with_wrong_variable_count_is_rejected(dispatch, area_mean, gridded, obs, mod):
    future = np.zeros((N_YEARS, N_MEMBERS, N_VARS + 1))

    with pytest.raises(ValueError, match="Only the leading time axis may differ"):
        area_mean(mod, obs, future)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_future_with_wrong_rank_is_rejected(dispatch, area_mean, gridded, obs, mod):
    future = np.zeros((N_YEARS, N_MEMBERS, 1, 1, N_VARS))

    with pytest.raises(ValueError, match="must have the same layout"):
        area_mean(mod, obs, future)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_gridded_future_with_wrong_grid_is_rejected(dispatch, area_mean, gridded, obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)
    future = np.zeros((N_YEARS, N_MEMBERS, 1, 3, N_VARS))

    with pytest.raises(ValueError, match="Only the leading time axis may differ"):
        gridded(grid_mod, grid_obs, future)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_dispatchers_reject_a_mismatched_future_too(dispatch, area_mean, gridded, obs, mod):
    future = np.zeros((N_YEARS, N_MEMBERS + 1, N_VARS))

    with pytest.raises(ValueError, match="Only the leading time axis may differ"):
        dispatch(mod, obs, future)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_a_future_differing_only_in_length_is_accepted(dispatch, area_mean, gridded, obs, mod):
    """The guard must not reject the case it exists to support."""
    for n_years in (1, N_YEARS // 2, N_YEARS, N_YEARS * 2):
        future = np.zeros((n_years, N_MEMBERS, N_VARS))
        assert area_mean(mod, obs, future).shape == future.shape


# ---------------------------------------------------------------------------
# Input without an ensemble axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_two_dimensional_input_is_treated_as_one_member(
    dispatch, area_mean, gridded, obs, mod
):
    """A single realisation is an ensemble of one, and must give the same answer."""
    single = mod[:, 0, :]

    without_axis = dispatch(single, obs)
    with_axis = dispatch(single[:, None, :], obs)

    assert without_axis.shape == single.shape
    assert np.array_equal(without_axis, with_axis[:, 0, :])


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_four_dimensional_input_is_treated_as_one_member(
    dispatch, area_mean, gridded, obs, mod
):
    grid_obs = as_grid(obs, ensemble=False)
    single = as_grid(mod, ensemble=True)[:, 0, ...]

    without_axis = dispatch(single, grid_obs)
    with_axis = dispatch(single[:, None, ...], grid_obs)

    assert without_axis.shape == single.shape
    assert np.array_equal(without_axis, with_axis[:, 0, ...])


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_explicit_forms_also_accept_the_reduced_layout(
    dispatch, area_mean, gridded, obs, mod
):
    """The workers behave the same whether reached directly or by dispatch."""
    single = mod[:, 0, :]
    grid_single = as_grid(mod, ensemble=True)[:, 0, ...]

    assert np.array_equal(area_mean(single, obs), dispatch(single, obs))
    assert np.array_equal(
        gridded(grid_single, as_grid(obs, ensemble=False)),
        dispatch(grid_single, as_grid(obs, ensemble=False)),
    )


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_reduced_layout_accepts_mod_future(dispatch, area_mean, gridded, obs, mod):
    single = mod[:, 0, :]
    future = single[: N_YEARS // 2]

    assert dispatch(single, obs, future).shape == future.shape


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_mod_future_must_match_the_reduced_layout_too(dispatch, area_mean, gridded, obs, mod):
    single = mod[:, 0, :]
    future_with_axis = mod[: N_YEARS // 2, :1, :]

    with pytest.raises(ValueError, match="must have the same layout"):
        dispatch(single, obs, future_with_axis)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_explicit_forms_reject_the_wrong_rank(dispatch, area_mean, gridded, obs, mod):
    """Area-mean and gridded each accept exactly two ranks, and no others."""
    grid_mod = as_grid(mod, ensemble=True)

    with pytest.raises(ValueError, match="expected 3 with an ensemble axis"):
        area_mean(grid_mod, as_grid(obs, ensemble=False))

    with pytest.raises(ValueError, match="expected 5 with an ensemble axis"):
        gridded(mod, obs)
