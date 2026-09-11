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


# ---------------------------------------------------------------------------
# Batched and looped implementations
# ---------------------------------------------------------------------------

from embcca.adjustment import (  # noqa: E402
    _batched_covariance,
    _batched_covariance_eigen,
    _from_blocks,
    _gridded_loop,
    _repeat_per_member,
    _to_blocks,
    _vectorised,
)


def test_batched_covariance_matches_numpy_cov_bit_for_bit():
    """The whole agreement between the two paths rests on this.

    A one-ulp difference here can flip an eigenvector sign, which does not
    cancel downstream and surfaces as a difference of hundreds of percent.
    """
    rng = np.random.default_rng(0)
    blocks = rng.normal(size=(40, 30, 3)) * [2.0, 9.0, 0.4]

    batched = _batched_covariance(blocks)
    per_block = np.stack([np.cov(b, rowvar=False) for b in blocks])

    assert np.array_equal(batched, per_block)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_covariance_matches_numpy_cov_for_both_dtypes(dtype):
    rng = np.random.default_rng(1)
    blocks = (rng.normal(size=(25, 30, 2)) * [5.0, 0.3]).astype(dtype)

    assert np.array_equal(
        _batched_covariance(blocks),
        np.stack([np.cov(b, rowvar=False) for b in blocks]),
    )


def test_batched_covariance_eigen_matches_the_single_block_form():
    rng = np.random.default_rng(2)
    blocks = rng.normal(size=(12, 30, 2))

    values, vectors = _batched_covariance_eigen(blocks, 1e-6)
    for i, block in enumerate(blocks):
        one_value, one_vector = _covariance_eigen(block, 1e-6)
        assert np.array_equal(values[i], one_value)
        assert np.array_equal(vectors[i], one_vector)


def test_to_blocks_and_from_blocks_round_trip():
    rng = np.random.default_rng(3)
    field = rng.normal(size=(30, 4, 3, 5, 2))

    blocks = _to_blocks(field)

    assert blocks.shape == (4 * 3 * 5, 30, 2)
    assert np.array_equal(_from_blocks(blocks, (4, 3, 5)), field)


def test_to_blocks_handles_observations_without_an_ensemble_axis():
    rng = np.random.default_rng(4)
    field = rng.normal(size=(30, 3, 5, 2))

    blocks = _to_blocks(field)

    assert blocks.shape == (3 * 5, 30, 2)
    assert np.array_equal(_from_blocks(blocks, (3, 5)), field)


def test_repeat_per_member_matches_the_block_ordering():
    """Observed quantities must line up with the model's member-major blocks."""
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(30, 2, 3, 2))
    n_ensembles = 4

    per_cell = _to_blocks(obs)
    repeated = _repeat_per_member(per_cell, n_ensembles)

    assert repeated.shape == (n_ensembles * 2 * 3, 30, 2)
    for member in range(n_ensembles):
        start = member * 6
        assert np.array_equal(repeated[start : start + 6], per_cell)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale_by_obs_std", [False, True])
def test_batched_and_looped_gridded_agree_bit_for_bit(dtype, scale_by_obs_std):
    rng = np.random.default_rng(6)
    mod = (rng.normal(size=(30, 5, 4, 3, 2)) * [40.0, 1.5] + [300.0, 22.0]).astype(dtype)
    obs = (rng.normal(size=(30, 4, 3, 2)) * [28.0, 1.0] + [295.0, 21.0]).astype(dtype)

    looped = _gridded_loop(mod, obs, None, 1e-6, scale_by_obs_std)
    batched = _vectorised(mod, obs, None, 1e-6, scale_by_obs_std, 5)

    assert np.array_equal(looped, batched)
    assert looped.dtype == batched.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_and_looped_agree_with_mod_future(dtype):
    rng = np.random.default_rng(7)
    mod = rng.normal(size=(30, 4, 3, 3, 2)).astype(dtype)
    obs = rng.normal(size=(30, 3, 3, 2)).astype(dtype)
    future = rng.normal(size=(18, 4, 3, 3, 2)).astype(dtype)

    assert np.array_equal(
        _gridded_loop(mod, obs, future, 1e-6, False),
        _vectorised(mod, obs, future, 1e-6, False, 5),
    )


def test_batched_and_looped_agree_on_degenerate_input():
    rng = np.random.default_rng(8)
    mod = rng.normal(size=(30, 4, 3, 3, 2))
    obs = rng.normal(size=(30, 3, 3, 2))
    mod[:, :, 1, 1, :] = np.nan           # a masked cell
    mod[:, 0, 0, 0, 1] = 5.0              # a constant variable

    looped = _gridded_loop(mod, obs, None, 1e-6, False)
    batched = _vectorised(mod, obs, None, 1e-6, False, 5)

    assert np.array_equal(np.isnan(looped), np.isnan(batched))
    assert np.array_equal(looped[~np.isnan(looped)], batched[~np.isnan(batched)])


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_vectorised_flag_selects_the_implementation(dispatch, area_mean, gridded, obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(
        gridded(grid_mod, grid_obs, vectorised=True),
        gridded(grid_mod, grid_obs, vectorised=False),
    )
    assert np.array_equal(
        dispatch(grid_mod, grid_obs, vectorised=False),
        gridded(grid_mod, grid_obs, vectorised=False),
    )


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_vectorised_defaults_to_true(dispatch, area_mean, gridded, obs, mod):
    grid_mod, grid_obs = as_grid(mod, ensemble=True), as_grid(obs, ensemble=False)

    assert np.array_equal(
        gridded(grid_mod, grid_obs),
        _vectorised(grid_mod, grid_obs, None, 1e-6, gridded is bias_adjust_gridded, 5),
    )


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_vectorised_flag_applies_to_area_mean_too(dispatch, area_mean, gridded, obs, mod):
    """Both layouts offer the choice, and both agree bit for bit."""
    assert np.array_equal(
        dispatch(mod, obs, vectorised=True), dispatch(mod, obs, vectorised=False)
    )
    assert np.array_equal(
        dispatch(mod, obs, vectorised=False), area_mean(mod, obs, vectorised=False)
    )


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
@pytest.mark.parametrize("vectorised, expected", [(True, "batched"), (False, "looped")])
def test_vectorised_flag_actually_dispatches(
    monkeypatch, dispatch, area_mean, gridded, obs, mod, vectorised, expected
):
    """The two implementations agree bit for bit, so equality cannot show which ran.

    Record which one is reached instead, or a flag that silently ignored its
    argument would pass every other test in this file.
    """
    import embcca.adjustment as adjustment

    called = []
    real_loop, real_batched = adjustment._gridded_loop, adjustment._vectorised

    def spy_loop(*args, **kwargs):
        called.append("looped")
        return real_loop(*args, **kwargs)

    def spy_batched(*args, **kwargs):
        called.append("batched")
        return real_batched(*args, **kwargs)

    monkeypatch.setattr(adjustment, "_gridded_loop", spy_loop)
    monkeypatch.setattr(adjustment, "_vectorised", spy_batched)

    gridded(as_grid(mod, ensemble=True), as_grid(obs, ensemble=False), vectorised=vectorised)

    assert called == [expected]


from embcca.adjustment import _area_mean_loop  # noqa: E402


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale_by_obs_std", [False, True])
def test_batched_and_looped_area_mean_agree_bit_for_bit(dtype, scale_by_obs_std):
    rng = np.random.default_rng(9)
    mod = (rng.normal(size=(30, 20, 2)) * [40.0, 1.5] + [300.0, 22.0]).astype(dtype)
    obs = (rng.normal(size=(30, 2)) * [28.0, 1.0] + [295.0, 21.0]).astype(dtype)

    looped = _area_mean_loop(mod, obs, None, 1e-6, scale_by_obs_std)
    batched = _vectorised(mod, obs, None, 1e-6, scale_by_obs_std, 3)

    assert np.array_equal(looped, batched)
    assert looped.dtype == batched.dtype == dtype


def test_batched_area_mean_agrees_with_mod_future():
    rng = np.random.default_rng(10)
    mod = rng.normal(size=(30, 8, 2))
    obs = rng.normal(size=(30, 2))
    future = rng.normal(size=(17, 8, 2))

    assert np.array_equal(
        _area_mean_loop(mod, obs, future, 1e-6, False),
        _vectorised(mod, obs, future, 1e-6, False, 3),
    )


def test_vectorised_serves_both_layouts_from_one_function():
    """A 1x1 grid is the area-mean problem, so the two routes must coincide."""
    rng = np.random.default_rng(11)
    mod = rng.normal(size=(30, 6, 2))
    obs = rng.normal(size=(30, 2))

    area_mean = _vectorised(mod, obs, None, 1e-6, False, 3)
    as_grid_ = _vectorised(mod[:, :, None, None, :], obs[:, None, None, :], None, 1e-6, False, 5)

    assert np.array_equal(area_mean, as_grid_[:, :, 0, 0, :])


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
@pytest.mark.parametrize("vectorised, expected", [(True, "batched"), (False, "looped")])
def test_area_mean_vectorised_flag_actually_dispatches(
    monkeypatch, dispatch, area_mean, gridded, obs, mod, vectorised, expected
):
    import embcca.adjustment as adjustment

    called = []
    real_loop, real_batched = adjustment._area_mean_loop, adjustment._vectorised

    def spy_loop(*args, **kwargs):
        called.append("looped")
        return real_loop(*args, **kwargs)

    def spy_batched(*args, **kwargs):
        called.append("batched")
        return real_batched(*args, **kwargs)

    monkeypatch.setattr(adjustment, "_area_mean_loop", spy_loop)
    monkeypatch.setattr(adjustment, "_vectorised", spy_batched)

    area_mean(mod, obs, vectorised=vectorised)

    assert called == [expected]


# ---------------------------------------------------------------------------
# Masked arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
@pytest.mark.parametrize("vectorised", [True, False])
def test_masked_input_returns_a_plain_array(dispatch, area_mean, gridded, obs, mod, vectorised):
    """`vectorised` selects an implementation; it must not change the return type."""
    masked = np.ma.array(mod, mask=np.zeros(mod.shape, dtype=bool))
    masked[:, 0, :] = np.ma.masked

    result = dispatch(masked, obs, vectorised=vectorised)

    assert not np.ma.isMaskedArray(result)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_masked_entries_become_nan_rather_than_being_adjusted(
    dispatch, area_mean, gridded, obs, mod
):
    """Masked data is missing data, not data to correct."""
    masked = np.ma.array(mod, mask=np.zeros(mod.shape, dtype=bool))
    masked[:, 0, :] = np.ma.masked

    result = dispatch(masked, obs)

    assert np.isnan(result[:, 0, :]).all()
    assert np.isfinite(result[:, 1:, :]).all()


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_masked_input_matches_filling_with_nan_first(dispatch, area_mean, gridded, obs, mod):
    """This is what the China analysis does for itself, via its own `_to_ndarray`."""
    masked = np.ma.array(mod, mask=np.zeros(mod.shape, dtype=bool))
    masked[:, 2, :] = np.ma.masked

    from_mask = dispatch(masked, obs)
    from_nan = dispatch(np.ma.filled(masked, np.nan), obs)

    assert np.array_equal(np.isnan(from_mask), np.isnan(from_nan))
    assert np.array_equal(from_mask[~np.isnan(from_mask)], from_nan[~np.isnan(from_nan)])


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
def test_a_masked_array_with_nothing_masked_is_unaffected(
    dispatch, area_mean, gridded, obs, mod
):
    """The Hunan analysis passes exactly this: a MaskedArray whose mask is all False.

    Area-averaging over latitude and longitude drops any cell-level masking, so
    the container survives but nothing in it is masked.
    """
    unmasked_container = np.ma.array(obs, mask=np.zeros(obs.shape, dtype=bool))

    assert np.array_equal(dispatch(mod, unmasked_container), dispatch(mod, obs))


@pytest.mark.parametrize("dispatch, area_mean, gridded", VARIANTS)
@pytest.mark.parametrize("vectorised", [True, False])
def test_masked_observations_are_handled_too(
    dispatch, area_mean, gridded, obs, mod, vectorised
):
    masked_obs = np.ma.array(obs, mask=np.zeros(obs.shape, dtype=bool))
    masked_obs[:, 1] = np.ma.masked

    result = dispatch(mod, masked_obs, vectorised=vectorised)

    assert not np.ma.isMaskedArray(result)
    assert np.isnan(result).all()


def test_to_ndarray_passes_none_through():
    """`mod_future` is handed to it unconditionally."""
    from embcca.adjustment import _to_ndarray

    assert _to_ndarray(None) is None
    assert isinstance(_to_ndarray([[1.0, 2.0]]), np.ndarray)
