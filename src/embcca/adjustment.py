# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""EMBCCA multivariate bias adjustment.

Eigen-decomposition-based Multivariate Bias Correction with Covariance
Alignment. Model and observed data are standardised, the model is whitened in
its own principal-component basis, and then recoloured in the observed one::

    Zm = mod_st @ V @ Gamma^-1/2 @ Lambda^1/2 @ W.T

where ``V, Gamma`` and ``W, Lambda`` are the eigenvectors and eigenvalues of
the standardised model and observed covariance matrices.

The result is returned to physical units in one of two ways, and that choice is
the only difference between the two entry points:

``bias_adjust``
    ``Zm * obs_std + obs_mean``. Transfers the observed mean, standard
    deviation and correlation structure onto the model.

``bias_adjust_unseen``
    ``Zm * mod_std + obs_mean``. Transfers the observed mean and correlation
    structure, but keeps the model's own standard deviation. UNSEEN analyses
    rely on the ensemble's larger sampled spread, so replacing it with the
    observed spread would remove the property they depend on.

Both entry points accept an optional ``mod_future``. When it is given, the
transform is calibrated on ``mod_calibration`` against ``obs_calibration`` and
then applied to ``mod_future``, which is standardised using the calibration
period's model mean and standard deviation. When it is omitted, the calibration
data is adjusted in place.

Each ensemble member is adjusted independently, and for gridded data each grid
cell is adjusted independently as well. The ensemble axis may be omitted for a
single realisation, in which case the result omits it too.

Missing data is carried as NaN, and is isolated to the block it appears in.
Masked arrays are accepted and their masked entries filled with NaN on the way
in, so the result is always a plain ndarray.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

__all__ = [
    "bias_adjust",
    "bias_adjust_unseen",
    "bias_adjust_area_mean",
    "bias_adjust_area_mean_unseen",
    "bias_adjust_gridded",
    "bias_adjust_gridded_unseen",
]


def _standardise(values, eps=None, axis=0):
    """Standardise over the time axis, for one block or a stack of them.

    Parameters
    ----------
    values : array
        ``(n_times, n_vars)`` for a single block, or ``(n_blocks, n_times,
        n_vars)`` for a stack. ``axis`` says which one.
    eps : float, optional
        If given, standard deviations below ``eps`` are raised to ``eps`` so
        that a variable with little or no variance does not divide by zero.
        If ``None``, no floor is applied.
    axis : int, optional
        The time axis: 0 for a single block, 1 for a stack.

    Returns
    -------
    standardised : array, same shape as ``values``
    mean, std : array
        ``values.shape`` with the time axis removed. ``std`` is the value
        actually divided by, i.e. after any ``eps`` floor.
    """
    mean = np.mean(values, axis=axis)
    std = np.std(values, axis=axis)
    if eps is not None:
        std = np.where(std < eps, eps, std)
    standardised = (values - np.expand_dims(mean, axis)) / np.expand_dims(std, axis)
    return standardised, mean, std


def _eigen(covariance, eps):
    """Eigendecomposition of one covariance matrix, or a stack of them.

    Parameters
    ----------
    covariance : array, shape (..., n_vars, n_vars)
    eps : float
        Floor applied to the eigenvalues, keeping them strictly positive so
        that the inverse square root below is well defined.

    Returns
    -------
    eigenvalues : array, shape (..., n_vars)
        Ascending, floored at ``eps``.
    eigenvectors : array, shape (..., n_vars, n_vars)
        Columns ordered to match ``eigenvalues``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return np.maximum(eigenvalues, eps), eigenvectors


def _covariance_eigen(standardised, eps):
    """Covariance across variables for one ``(n_times, n_vars)`` block, decomposed."""
    return _eigen(np.cov(standardised, rowvar=False), eps)


def _batched_covariance(standardised):
    """``np.cov(block, rowvar=False)`` for a stack of blocks, bit for bit.

    This mirrors ``np.cov``'s internals deliberately and in order, because the
    result is sensitive to every one of them. ``np.cov`` promotes to float64,
    transposes to ``(n_vars, n_times)`` as a *view*, centres along the last
    axis, takes the product, and scales by the reciprocal of the degrees of
    freedom. Doing any of it differently, including centring a contiguous copy
    rather than a view, changes the summation order and so the last bits.

    Those last bits matter here more than they usually would. :func:`_eigen`
    can turn a one-ulp difference in the covariance into an eigenvector sign
    flip, and that sign does not cancel in :func:`_align`, so it surfaces as a
    difference of hundreds of percent rather than of rounding. Reproducing
    ``np.cov`` exactly is what lets the batched and looped paths agree.

    Parameters
    ----------
    standardised : array, shape (n_blocks, n_times, n_vars)

    Returns
    -------
    array, shape (n_blocks, n_vars, n_vars)
    """
    n_times = standardised.shape[1]
    blocks = np.asarray(standardised, dtype=np.float64).transpose(0, 2, 1)
    blocks = blocks - blocks.mean(axis=2)[:, :, None]
    return (blocks @ blocks.transpose(0, 2, 1)) * np.true_divide(1, n_times - 1)


def _batched_covariance_eigen(standardised, eps):
    """Covariance across variables for a stack of blocks, decomposed."""
    return _eigen(_batched_covariance(standardised), eps)


def _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors):
    """Whiten in the model's eigenbasis and recolour in the observed one.

    Works on one block or a stack of them: every operation broadcasts over
    leading axes, so ``(n_times, n_vars)`` and ``(n_blocks, n_times, n_vars)``
    both go through unchanged.

    Parameters
    ----------
    mod_st : array, shape (..., n_times, n_vars)
        Standardised model block, or a stack of them.
    mod_eigenvalues, mod_eigenvectors : array
        Eigendecomposition of the standardised model covariance, shaped
        ``(..., n_vars)`` and ``(..., n_vars, n_vars)``.
    obs_eigenvalues, obs_eigenvectors : array
        The same for the standardised observed covariance.

    Returns
    -------
    aligned : array, same shape as ``mod_st``
        Still standardised; the caller returns it to physical units.

    Notes
    -----
    Model and observed principal components are paired by eigenvalue rank,
    both decompositions being returned in ascending order.

    The eigenvalue scalings are applied by elementwise multiplication rather
    than as ``np.diag`` matrices. That is bit-identical to the matrix form,
    since the off-diagonal terms contribute exact zeros, and it is what lets
    the same code serve a stack.
    """
    gamma_inv_sqrt = np.expand_dims(1.0 / np.sqrt(mod_eigenvalues), -2)
    lambda_sqrt = np.expand_dims(np.sqrt(obs_eigenvalues), -2)

    aligned = np.asarray(mod_st) @ mod_eigenvectors
    aligned = aligned * gamma_inv_sqrt
    aligned = aligned * lambda_sqrt
    return aligned @ np.swapaxes(obs_eigenvectors, -1, -2)



def _to_blocks(field):
    """Stack every ``(n_times, n_vars)`` block onto a leading batch axis.

    Accepts either an observed field ``(n_times, n_lons, n_lats, n_vars)`` or a
    model field ``(n_times, n_ensembles, n_lons, n_lats, n_vars)``; the batch
    axis then runs over cells, or over members and cells respectively, in C
    order.

    Parameters
    ----------
    field : array, shape (n_times, ..., n_vars)

    Returns
    -------
    array, shape (n_blocks, n_times, n_vars)
    """
    n_times, n_vars = field.shape[0], field.shape[-1]
    return np.moveaxis(field, 0, -2).reshape(-1, n_times, n_vars)


def _from_blocks(blocks, leading):
    """Invert :func:`_to_blocks`, restoring the axes it collapsed.

    Parameters
    ----------
    blocks : array, shape (n_blocks, n_times, n_vars)
    leading : tuple of int
        The axes between time and variables in the original field, e.g.
        ``(n_ensembles, n_lons, n_lats)``.

    Returns
    -------
    array, shape (n_times, *leading, n_vars)
    """
    n_times, n_vars = blocks.shape[1], blocks.shape[2]
    return np.moveaxis(blocks.reshape(*leading, n_times, n_vars), -2, 0)


def _repeat_per_member(values, n_ensembles):
    """Tile per-cell quantities so they line up with the model's block order.

    Observed quantities are computed once per grid cell, but the model's blocks
    run over members and cells. Repeating along the member axis puts them in
    the same C order that :func:`_to_blocks` produces.

    Parameters
    ----------
    values : array, shape (n_cells, ...)
    n_ensembles : int

    Returns
    -------
    array, shape (n_ensembles * n_cells, ...)
    """
    return np.broadcast_to(values, (n_ensembles, *values.shape)).reshape(-1, *values.shape[1:])


def _to_ndarray(values, fill_value=np.nan):
    """Convert a masked array to a plain one, filling masked entries.

    Missing data is carried as NaN throughout, which every step of the
    transform already isolates to the block it appears in. A mask means the
    same thing, so it is translated on the way in rather than interpreted
    separately.

    Without this, a masked array would be adjusted as though its masked
    entries were data, and the result would come back wrapped in a
    ``MaskedArray`` whose mask came from ``np.empty_like`` and so was never
    initialised.

    Parameters
    ----------
    values : array or None
        Masked or plain. ``None`` passes through, so ``mod_future`` can be
        handed here unconditionally.
    fill_value : float, optional
        Substituted for masked entries.

    Returns
    -------
    ndarray or None
    """
    if values is None:
        return None
    if np.ma.isMaskedArray(values):
        return values.filled(fill_value)
    return np.asarray(values)


def _check_future(mod_calibration, mod_future):
    """Reject a ``mod_future`` that does not line up with the calibration data.

    Only the leading time axis may differ: the transform is derived per
    ensemble member (and per grid cell), so every other axis has to match for
    the calibration to apply. Without this check a ``mod_future`` carrying more
    ensemble members than ``mod_calibration`` would return silently, with the
    surplus members never written and left holding whatever the output buffer
    was allocated over.
    """
    if mod_future is None:
        return

    if mod_future.ndim != mod_calibration.ndim:
        raise ValueError(
            f"mod_future has {mod_future.ndim} dimensions but mod_calibration has "
            f"{mod_calibration.ndim}. They must have the same layout, differing "
            "only in the length of the leading time axis."
        )

    if mod_future.shape[1:] != mod_calibration.shape[1:]:
        raise ValueError(
            f"mod_future has shape {mod_future.shape} but mod_calibration has "
            f"{mod_calibration.shape}. Only the leading time axis may differ; "
            f"got {mod_future.shape[1:]} against {mod_calibration.shape[1:]} "
            "for the remaining axes."
        )


def _add_ensemble_axis(mod_calibration, mod_future, ndim):
    """Insert a length-1 ensemble axis when the caller did not supply one.

    A single realisation is just an ensemble of one, so rather than duplicate
    the transform for that case the arrays are reshaped into the ensemble
    layout, adjusted unchanged, and reshaped back by :func:`_drop_ensemble_axis`.

    Parameters
    ----------
    mod_calibration, mod_future : array
        As passed by the caller. ``mod_future`` may be ``None``.
    ndim : int
        Rank of the ensemble layout this function expects: 3 for area-mean,
        5 for gridded.

    Returns
    -------
    mod_calibration, mod_future : array
        Reshaped to ``ndim`` dimensions if an axis was inserted.
    inserted : bool
        Whether an axis was inserted, for :func:`_drop_ensemble_axis`.
    """
    if mod_calibration.ndim == ndim:
        return mod_calibration, mod_future, False

    if mod_calibration.ndim == ndim - 1:
        mod_calibration = np.expand_dims(mod_calibration, 1)
        if mod_future is not None:
            mod_future = np.expand_dims(mod_future, 1)
        return mod_calibration, mod_future, True

    raise ValueError(
        f"mod_calibration has {mod_calibration.ndim} dimensions; expected {ndim} "
        f"with an ensemble axis, or {ndim - 1} without one."
    )


def _drop_ensemble_axis(mod_corrected, inserted):
    """Undo :func:`_add_ensemble_axis`, so the result matches what was passed in."""
    return mod_corrected[:, 0] if inserted else mod_corrected


def _area_mean_loop(mod_calibration, obs_calibration, mod_future, eps, scale_by_obs_std):
    """Area-mean EMBCCA, looping over ensemble members.

    The two public variants differ only in ``scale_by_obs_std``, which selects
    the standard deviation used to return the aligned block to physical units.
    """
    mod_calibration = _to_ndarray(mod_calibration)
    obs_calibration = _to_ndarray(obs_calibration)
    mod_future = _to_ndarray(mod_future)
    _check_future(mod_calibration, mod_future)
    mod_calibration, mod_future, inserted = _add_ensemble_axis(mod_calibration, mod_future, 3)
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    obs_st, obs_mean, obs_std = _standardise(obs_calibration, eps)
    obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

    for member in range(mod_calibration.shape[1]):
        mod_st, mod_mean, mod_std = _standardise(mod_calibration[:, member, :], eps)
        mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
        if mod_future is not None:
            mod_st = (mod_future[:, member, :] - mod_mean) / mod_std

        aligned = _align(
            mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors
        )
        scale = obs_std if scale_by_obs_std else mod_std
        mod_corrected[:, member, :] = aligned * scale + obs_mean

    return _drop_ensemble_axis(mod_corrected, inserted)


def _gridded_loop(mod_calibration, obs_calibration, mod_future, eps, scale_by_obs_std):
    """Gridded EMBCCA, looping over grid cells and then ensemble members.

    Kept alongside :func:`_gridded_vectorised` because the manuscript's timing
    comparison against the SBCK methods is only like for like when EMBCCA is
    applied the same way they are, one cell and member at a time.
    """
    mod_calibration = _to_ndarray(mod_calibration)
    obs_calibration = _to_ndarray(obs_calibration)
    mod_future = _to_ndarray(mod_future)
    _check_future(mod_calibration, mod_future)
    mod_calibration, mod_future, inserted = _add_ensemble_axis(mod_calibration, mod_future, 5)
    _, n_ensembles, n_lons, n_lats, _ = mod_calibration.shape
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            obs_st, obs_mean, obs_std = _standardise(obs_calibration[:, ilon, ilat, :], eps)
            obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

            for member in range(n_ensembles):
                mod_st, mod_mean, mod_std = _standardise(
                    mod_calibration[:, member, ilon, ilat, :], eps
                )
                mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
                if mod_future is not None:
                    mod_st = (mod_future[:, member, ilon, ilat, :] - mod_mean) / mod_std

                aligned = _align(
                    mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors
                )
                scale = obs_std if scale_by_obs_std else mod_std
                mod_corrected[:, member, ilon, ilat, :] = aligned * scale + obs_mean

    return _drop_ensemble_axis(mod_corrected, inserted)


def _vectorised(mod_calibration, obs_calibration, mod_future, eps, scale_by_obs_std, ndim):
    """EMBCCA with every block decomposed in one batch, for either layout.

    Each ensemble member, and for gridded data each member and cell, is an
    independent ``(n_times, n_vars)`` problem. Stacking them onto a leading
    batch axis lets ``np.linalg.eigh`` decompose the whole stack in one call.
    The arithmetic is unchanged and the result is bit-identical to the looped
    forms; only the Python loops disappear.

    The area-mean layout is the same problem with a single cell, so it goes
    through this function unaltered: ``_to_blocks`` turns ``(n_times, n_vars)``
    observations into a batch of one, and the observed quantities are repeated
    across members exactly as they would be across members and cells.

    Parameters
    ----------
    mod_calibration, obs_calibration, mod_future, eps
        As for the public functions.
    scale_by_obs_std : bool
        Return to physical units with the observed standard deviation rather
        than the model's, which is the only difference between the variants.
    ndim : int
        Rank of the ensemble layout: 3 for area-mean, 5 for gridded.

    Returns
    -------
    mod_corrected : array

    Notes
    -----
    Speed is bought with memory: several arrays of
    ``n_blocks * n_times * n_vars`` elements are live at once, where the looped
    forms hold one block at a time.
    """
    mod_calibration = _to_ndarray(mod_calibration)
    obs_calibration = _to_ndarray(obs_calibration)
    mod_future = _to_ndarray(mod_future)
    _check_future(mod_calibration, mod_future)
    mod_calibration, mod_future, inserted = _add_ensemble_axis(mod_calibration, mod_future, ndim)
    leading = mod_calibration.shape[1:-1]
    n_ensembles = leading[0]

    obs_st, obs_mean, obs_std = _standardise(_to_blocks(obs_calibration), eps, axis=1)
    obs_eigenvalues, obs_eigenvectors = _batched_covariance_eigen(obs_st, eps)

    mod_st, mod_mean, mod_std = _standardise(_to_blocks(mod_calibration), eps, axis=1)
    mod_eigenvalues, mod_eigenvectors = _batched_covariance_eigen(mod_st, eps)

    if mod_future is not None:
        mod_st = (_to_blocks(mod_future) - mod_mean[:, None, :]) / mod_std[:, None, :]

    # Observed quantities are shared by every member at a given cell.
    obs_eigenvalues = _repeat_per_member(obs_eigenvalues, n_ensembles)
    obs_eigenvectors = _repeat_per_member(obs_eigenvectors, n_ensembles)
    obs_mean = _repeat_per_member(obs_mean, n_ensembles)
    obs_std = _repeat_per_member(obs_std, n_ensembles)

    aligned = _align(
        mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors
    )
    scale = obs_std if scale_by_obs_std else mod_std
    blocks = aligned * scale[:, None, :] + obs_mean[:, None, :]

    # The looped forms write into np.empty_like(mod), so the result carries the
    # input dtype however the intermediates were promoted.
    source = mod_calibration if mod_future is None else mod_future
    blocks = blocks.astype(source.dtype, copy=False)

    return _drop_ensemble_axis(_from_blocks(blocks, leading), inserted)


def bias_adjust_area_mean_unseen(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA to area-mean (non-spatial) data, adjusting the mean only.

    Transfers the observed mean and correlation structure onto each ensemble
    member while keeping that member's own standard deviation, which is the
    behaviour UNSEEN analyses require.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently. The ensemble axis may be omitted, giving
        ``(n_years, n_vars)``, in which case the result omits it too.
    obs_calibration : array, shape (n_years, n_vars)
        Observed data for the calibration period, providing the target mean
        and correlation structure.
    mod_future : array, shape (n_future_years, n_ensembles, n_vars), optional
        Model data to adjust using the transform calibrated above. It is
        standardised with the calibration period's model mean and standard
        deviation. If omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero.
    vectorised : bool, optional
        Decompose every block in one batch (the default) rather than looping
        over them. The two produce bit-identical results; the batched form is
        substantially faster but holds every block in memory at once. Pass
        ``False`` to reproduce the manuscript's timing comparison against the
        SBCK methods, which is only like for like when EMBCCA is applied one
        block at a time.

    Returns
    -------
    mod_corrected : array
        Adjusted model data.

    See Also
    --------
    bias_adjust_area_mean : Same layout, adjusting the standard deviation too.
    bias_adjust_gridded_unseen : Gridded equivalent of this function.
    """
    if vectorised:
        return _vectorised(mod_calibration, obs_calibration, mod_future, eps, False, 3)
    return _area_mean_loop(mod_calibration, obs_calibration, mod_future, eps, False)


def bias_adjust_gridded_unseen(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA to gridded data at each grid cell, adjusting the mean only.

    Transfers the observed mean and correlation structure onto each ensemble
    member while keeping that member's own standard deviation, which is the
    behaviour UNSEEN analyses require. Each grid cell is treated independently.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently at each grid cell. The ensemble axis may be
        omitted, giving ``(n_years, n_lons, n_lats, n_vars)``, in which case
        the result omits it too.
    obs_calibration : array, shape (n_years, n_lons, n_lats, n_vars)
        Observed data for the calibration period, providing the target mean
        and correlation structure.
    mod_future : array, optional
        Shape ``(n_future_years, n_ensembles, n_lons, n_lats, n_vars)``. Model
        data to adjust using the transform calibrated above, standardised with
        the calibration period's model mean and standard deviation. If
        omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero on cells with little or no variance.
    vectorised : bool, optional
        Decompose every cell and member in one batch (the default) rather than
        looping over them. The two produce bit-identical results; the batched
        form is faster but holds every block in memory at once. Pass ``False``
        to reproduce the manuscript's timing comparison against the SBCK methods,
        which is only like for like when EMBCCA is applied one cell and member
        at a time.

    Returns
    -------
    mod_corrected : array
        Adjusted model data.

    See Also
    --------
    bias_adjust_gridded : Same layout, adjusting the standard deviation too.
    bias_adjust_area_mean_unseen : Area-mean equivalent of this function.
    """
    if vectorised:
        return _vectorised(mod_calibration, obs_calibration, mod_future, eps, False, 5)
    return _gridded_loop(mod_calibration, obs_calibration, mod_future, eps, False)


def bias_adjust_area_mean(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA to area-mean (non-spatial) data, adjusting mean and standard deviation.

    Transfers the observed mean, standard deviation and correlation structure
    onto each ensemble member. Use :func:`bias_adjust_area_mean_unseen` where the
    model's own spread must be preserved.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently. The ensemble axis may be omitted, giving
        ``(n_years, n_vars)``, in which case the result omits it too.
    obs_calibration : array, shape (n_years, n_vars)
        Observed data for the calibration period, providing the target mean,
        standard deviation and correlation structure.
    mod_future : array, shape (n_future_years, n_ensembles, n_vars), optional
        Model data to adjust using the transform calibrated above. It is
        standardised with the calibration period's model mean and standard
        deviation. If omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero.
    vectorised : bool, optional
        Decompose every block in one batch (the default) rather than looping
        over them. The two produce bit-identical results; the batched form is
        substantially faster but holds every block in memory at once. Pass
        ``False`` to reproduce the manuscript's timing comparison against the
        SBCK methods, which is only like for like when EMBCCA is applied one
        block at a time.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_future`` if given, otherwise as
        ``mod_calibration``.

    See Also
    --------
    bias_adjust_area_mean_unseen : Same layout, keeping the model's standard deviation.
    bias_adjust_gridded : Gridded equivalent of this function.
    """
    if vectorised:
        return _vectorised(mod_calibration, obs_calibration, mod_future, eps, True, 3)
    return _area_mean_loop(mod_calibration, obs_calibration, mod_future, eps, True)


def bias_adjust_gridded(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA to gridded data at each grid cell, adjusting mean and standard 
    deviation.

    Transfers the observed mean, standard deviation and correlation structure
    onto each ensemble member, treating each grid cell independently. Use
    :func:`bias_adjust_gridded_unseen` where the model's own spread must be
    preserved.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently at each grid cell. The ensemble axis may be
        omitted, giving ``(n_years, n_lons, n_lats, n_vars)``, in which case
        the result omits it too.
    obs_calibration : array, shape (n_years, n_lons, n_lats, n_vars)
        Observed data for the calibration period, providing the target mean,
        standard deviation and correlation structure.
    mod_future : array, optional
        Shape ``(n_future_years, n_ensembles, n_lons, n_lats, n_vars)``. Model
        data to adjust using the transform calibrated above, standardised with
        the calibration period's model mean and standard deviation. If
        omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero on cells with little or no variance.
    vectorised : bool, optional
        Decompose every cell and member in one batch (the default) rather than
        looping over them. The two produce bit-identical results; the batched
        form is faster but holds every block in memory at once. Pass ``False``
        to reproduce the manuscript's timing comparison against the SBCK methods,
        which is only like for like when EMBCCA is applied one cell and member
        at a time.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_calibration``.

    See Also
    --------
    bias_adjust_gridded_unseen : Same layout, keeping the model's standard deviation.
    bias_adjust_area_mean : Area-mean equivalent of this function.
    """
    if vectorised:
        return _vectorised(mod_calibration, obs_calibration, mod_future, eps, True, 5)
    return _gridded_loop(mod_calibration, obs_calibration, mod_future, eps, True)


def bias_adjust(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, 
        vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA (adjust mean and standard deviation), dispatching on the layout of 
    ``mod_calibration`` and whether ``mod_future`` is provided.

    Parameters
    ----------
    mod_calibration : array
        Either ``(n_years, n_ensembles, n_vars)`` for area-mean data or
        ``(n_years, n_ensembles, n_lons, n_lats, n_vars)`` for gridded data.
        The ensemble axis may be omitted in either case, giving 2-D or 4-D
        input for a single realisation; the result then omits it too.
    obs_calibration : array
        Correspondingly ``(n_years, n_vars)`` or
        ``(n_years, n_lons, n_lats, n_vars)``.
    mod_future : array, optional
        If given, adjusted using the calibration data's observed mean and
        correlation structure. If not given, the calibration data is adjusted
        in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero.
    vectorised : bool, optional
        Decompose every block in one batch (the default) rather than looping
        over them; see :func:`bias_adjust_gridded`. Applies to both layouts.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_calibration``.

    See Also
    --------
    bias_adjust_area_mean : Area-mean form, called for 3-D input.
    bias_adjust_gridded : Gridded form, called for 5-D input.

    Notes
    -----
    The two spatial axes of the gridded form are treated symmetrically -- the
    adjustment is applied independently at each cell -- so their order does not
    affect the result.
    """
    ndim = np.ndim(mod_calibration)
    if ndim in (2, 3):
        return bias_adjust_area_mean(
            mod_calibration, obs_calibration, mod_future, eps, vectorised
        )
    if ndim in (4, 5):
        return bias_adjust_gridded(
            mod_calibration, obs_calibration, mod_future, eps, vectorised
        )
    raise ValueError(
        f"Unsupported dimensions: mod.ndim={ndim}. Expected 3 for area-mean "
        "(n_years, n_ensembles, n_vars) or 5 for gridded "
        "(n_years, n_ensembles, n_lons, n_lats, n_vars) input, or 2 and 4 "
        "respectively for a single realisation with no ensemble axis."
    )



def bias_adjust_unseen(
        mod_calibration: np.ndarray, 
        obs_calibration: np.ndarray, 
        mod_future: Optional[np.ndarray] = None, 
        eps: float = 1e-6, 
        vectorised: bool = True
    ) -> np.ndarray:
    """Apply EMBCCA for UNSEEN (adjust mean only), dispatching on the layout of 
    ``mod_calibration`` and whether ``mod_future`` is provided.

    Parameters
    ----------
    mod_calibration : array
        Either ``(n_years, n_ensembles, n_vars)`` for area-mean data or
        ``(n_years, n_ensembles, n_lons, n_lats, n_vars)`` for gridded data.
        The ensemble axis may be omitted in either case, giving 2-D or 4-D
        input for a single realisation; the result then omits it too.
    obs_calibration : array
        Correspondingly ``(n_years, n_vars)`` or
        ``(n_years, n_lons, n_lats, n_vars)``.
    mod_future : array, optional
        If given, adjusted using the calibration data's observed mean and
        correlation structure. If not given, the calibration data is adjusted
        in place.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero.
    vectorised : bool, optional
        Decompose every block in one batch (the default) rather than looping
        over them; see :func:`bias_adjust_gridded`. Applies to both layouts.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_calibration``.

    See Also
    --------
    bias_adjust_area_mean_unseen : Area-mean form, called for 3-D input.
    bias_adjust_gridded_unseen : Gridded form, called for 5-D input.

    Notes
    -----
    The two spatial axes of the gridded form are treated symmetrically -- the
    adjustment is applied independently at each cell -- so their order does not
    affect the result.
    """
    ndim = np.ndim(mod_calibration)
    if ndim in (2, 3):
        return bias_adjust_area_mean_unseen(
            mod_calibration, obs_calibration, mod_future, eps, vectorised
        )
    if ndim in (4, 5):
        return bias_adjust_gridded_unseen(
            mod_calibration, obs_calibration, mod_future, eps, vectorised
        )
    raise ValueError(
        f"Unsupported dimensions: mod.ndim={ndim}. Expected 3 for area-mean "
        "(n_years, n_ensembles, n_vars) or 5 for gridded "
        "(n_years, n_ensembles, n_lons, n_lats, n_vars) input, or 2 and 4 "
        "respectively for a single realisation with no ensemble axis."
    )
