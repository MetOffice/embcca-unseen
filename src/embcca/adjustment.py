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
cell is adjusted independently as well.
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


def _standardise(values, eps=None):
    """Standardise a ``(n_times, n_vars)`` block over its leading axis.

    Parameters
    ----------
    values : array, shape (n_times, n_vars)
    eps : float, optional
        If given, standard deviations below ``eps`` are raised to ``eps`` so
        that a variable with little or no variance does not divide by zero.
        If ``None``, no floor is applied.

    Returns
    -------
    standardised : array, shape (n_times, n_vars)
    mean : array, shape (n_vars,)
    std : array, shape (n_vars,)
        The standard deviation actually used, i.e. after any ``eps`` floor.
    """
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    if eps is not None:
        std = np.where(std < eps, eps, std)
    return (values - mean) / std, mean, std


def _covariance_eigen(standardised, eps):
    """Eigendecomposition of the covariance across variables.

    Parameters
    ----------
    standardised : array, shape (n_times, n_vars)
    eps : float
        Floor applied to the eigenvalues, keeping them strictly positive so
        that the inverse square root below is well defined.

    Returns
    -------
    eigenvalues : array, shape (n_vars,)
        Ascending, floored at ``eps``.
    eigenvectors : array, shape (n_vars, n_vars)
        Columns ordered to match ``eigenvalues``.
    """
    covariance = np.cov(standardised, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return np.maximum(eigenvalues, eps), eigenvectors


def _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors):
    """Whiten in the model's eigenbasis and recolour in the observed one.

    Parameters
    ----------
    mod_st : array, shape (n_times, n_vars)
        Standardised model block.
    mod_eigenvalues, mod_eigenvectors : array
        Eigendecomposition of the standardised model covariance.
    obs_eigenvalues, obs_eigenvectors : array
        Eigendecomposition of the standardised observed covariance.

    Returns
    -------
    aligned : array, shape (n_times, n_vars)
        Still standardised; the caller returns it to physical units.

    Notes
    -----
    Model and observed principal components are paired by eigenvalue rank,
    both decompositions being returned in ascending order.
    """
    gamma_inv_sqrt = np.diag(1.0 / np.sqrt(mod_eigenvalues))
    lambda_sqrt = np.diag(np.sqrt(obs_eigenvalues))

    aligned = np.asarray(mod_st) @ mod_eigenvectors
    aligned = aligned @ gamma_inv_sqrt
    aligned = aligned @ lambda_sqrt
    return aligned @ obs_eigenvectors.T



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


def bias_adjust_area_mean_unseen(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA to area-mean (non-spatial) data, adjusting the mean only.

    Transfers the observed mean and correlation structure onto each ensemble
    member while keeping that member's own standard deviation, which is the
    behaviour UNSEEN analyses require.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently.
    obs_calibration : array, shape (n_years, n_vars)
        Observed data for the calibration period, providing the target mean
        and correlation structure.
    mod_future : array, shape (n_future_years, n_ensembles, n_vars), optional
        Model data to adjust using the transform calibrated above. It is
        standardised with the calibration period's model mean and standard
        deviation. If omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to the eigenvalues, keeping them strictly positive so
        that the inverse square root is well defined.

    Returns
    -------
    mod_corrected : array
        Adjusted model data.

    See Also
    --------
    bias_adjust_area_mean : Same layout, adjusting the standard deviation too.
    bias_adjust_gridded_unseen : Gridded equivalent of this function.

    Notes
    -----
    Standard deviations are not floored here, so a variable with zero variance
    across time will divide by zero. :func:`bias_adjust_gridded_unseen` applies the
    ``eps`` floor to standard deviations as well, since an all-sea or otherwise
    constant grid cell makes that a realistic possibility.
    """
    _check_future(mod_calibration, mod_future)
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    obs_st, obs_mean, _ = _standardise(obs_calibration)
    obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

    for member in range(mod_calibration.shape[1]):
        mod_st, mod_mean, mod_std = _standardise(mod_calibration[:, member, :])
        mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
        if mod_future is not None:
            future_mod_st = (mod_future[:, member, :] - mod_mean) / mod_std
            mod_st = future_mod_st

        aligned = _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors)
        mod_corrected[:, member, :] = aligned * mod_std + obs_mean

    return mod_corrected


def bias_adjust_gridded_unseen(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA to gridded data at each grid cell, adjusting the mean only.

    Transfers the observed mean and correlation structure onto each ensemble
    member while keeping that member's own standard deviation, which is the
    behaviour UNSEEN analyses require. Each grid cell is treated independently.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently at each grid cell.
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

    Returns
    -------
    mod_corrected : array
        Adjusted model data.

    See Also
    --------
    bias_adjust_gridded : Same layout, adjusting the standard deviation too.
    bias_adjust_area_mean_unseen : Area-mean equivalent of this function.
    """
    _check_future(mod_calibration, mod_future)
    _, n_ensembles, n_lons, n_lats, _ = mod_calibration.shape
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            obs_st, obs_mean, _ = _standardise(obs_calibration[:, ilon, ilat, :], eps)
            obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

            for member in range(n_ensembles):
                mod_st, mod_mean, mod_std = _standardise(mod_calibration[:, member, ilon, ilat, :], eps)
                mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
                if mod_future is not None:
                    future_mod_st = (mod_future[:, member, ilon, ilat, :] - mod_mean) / mod_std
                    mod_st = future_mod_st

                aligned = _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors)
                mod_corrected[:, member, ilon, ilat, :] = aligned * mod_std + obs_mean

    return mod_corrected


def bias_adjust_area_mean(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA to area-mean (non-spatial) data, adjusting mean and standard deviation.

    Transfers the observed mean, standard deviation and correlation structure
    onto each ensemble member. Use :func:`bias_adjust_area_mean_unseen` where the
    model's own spread must be preserved.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently.
    obs_calibration : array, shape (n_years, n_vars)
        Observed data for the calibration period, providing the target mean,
        standard deviation and correlation structure.
    mod_future : array, shape (n_future_years, n_ensembles, n_vars), optional
        Model data to adjust using the transform calibrated above. It is
        standardised with the calibration period's model mean and standard
        deviation. If omitted, ``mod_calibration`` is adjusted in place.
    eps : float, optional
        Floor applied to the eigenvalues, keeping them strictly positive so
        that the inverse square root is well defined.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_future`` if given, otherwise as
        ``mod_calibration``.

    See Also
    --------
    bias_adjust_area_mean_unseen : Same layout, keeping the model's standard deviation.
    bias_adjust_gridded : Gridded equivalent of this function.

    Notes
    -----
    Standard deviations are not floored here, so a variable with zero variance
    across time will divide by zero. :func:`bias_adjust_gridded` applies the ``eps``
    floor to standard deviations as well.
    """
    _check_future(mod_calibration, mod_future)
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    obs_st, obs_mean, obs_std = _standardise(obs_calibration)
    obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

    for member in range(mod_calibration.shape[1]):
        mod_st, mod_mean, mod_std = _standardise(mod_calibration[:, member, :])
        mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
        if mod_future is not None:
            future_mod_st = (mod_future[:, member, :] - mod_mean) / mod_std
            mod_st = future_mod_st

        aligned = _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors)
        mod_corrected[:, member, :] = aligned * obs_std + obs_mean

    return mod_corrected



def bias_adjust_gridded(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA to gridded data at each grid cell, adjusting mean and standard deviation.

    Transfers the observed mean, standard deviation and correlation structure
    onto each ensemble member, treating each grid cell independently. Use
    :func:`bias_adjust_gridded_unseen` where the model's own spread must be
    preserved.

    Parameters
    ----------
    mod_calibration : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
        Model data for the calibration period. Each ensemble member is
        adjusted independently at each grid cell.
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

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod_calibration``.

    See Also
    --------
    bias_adjust_gridded_unseen : Same layout, keeping the model's standard deviation.
    bias_adjust_area_mean : Area-mean equivalent of this function.
    """
    _check_future(mod_calibration, mod_future)
    _, n_ensembles, n_lons, n_lats, _ = mod_calibration.shape
    mod_corrected = np.empty_like(mod_calibration if mod_future is None else mod_future)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            obs_st, obs_mean, obs_std = _standardise(obs_calibration[:, ilon, ilat, :], eps)
            obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

            for member in range(n_ensembles):
                mod_st, mod_mean, mod_std = _standardise(mod_calibration[:, member, ilon, ilat, :], eps)
                mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
                if mod_future is not None:
                    future_mod_st = (mod_future[:, member, ilon, ilat, :] - mod_mean) / mod_std
                    mod_st = future_mod_st

                aligned = _align(mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors)
                mod_corrected[:, member, ilon, ilat, :] = aligned * obs_std + obs_mean

    return mod_corrected


def bias_adjust(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA (adjust mean and standard deviation), dispatching on the layout of ``mod_calibration`` and whether ``mod_future`` is provided.

    Parameters
    ----------
    mod_calibration : array
        Either ``(n_years, n_ensembles, n_vars)`` for area-mean data or
        ``(n_years, n_ensembles, n_lons, n_lats, n_vars)`` for gridded data.
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
    if ndim == 3:
        return bias_adjust_area_mean(mod_calibration, obs_calibration, mod_future, eps)
    if ndim == 5:
        return bias_adjust_gridded(mod_calibration, obs_calibration, mod_future, eps)
    raise ValueError(
        f"Unsupported dimensions: mod.ndim={ndim}. Expected 3 for area-mean "
        "(n_years, n_ensembles, n_vars) or 5 for gridded "
        "(n_years, n_ensembles, n_lons, n_lats, n_vars) input."
    )



def bias_adjust_unseen(mod_calibration: np.ndarray, obs_calibration: np.ndarray, mod_future: Optional[np.ndarray] = None, eps: float = 1e-6) -> np.ndarray:
    """Apply EMBCCA for UNSEEN (adjust mean only), dispatching on the layout of ``mod_calibration`` and whether ``mod_future`` is provided.

    Parameters
    ----------
    mod_calibration : array
        Either ``(n_years, n_ensembles, n_vars)`` for area-mean data or
        ``(n_years, n_ensembles, n_lons, n_lats, n_vars)`` for gridded data.
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
    if ndim == 3:
        return bias_adjust_area_mean_unseen(mod_calibration, obs_calibration, mod_future, eps)
    if ndim == 5:
        return bias_adjust_gridded_unseen(mod_calibration, obs_calibration, mod_future, eps)
    raise ValueError(
        f"Unsupported dimensions: mod.ndim={ndim}. Expected 3 for area-mean "
        "(n_years, n_ensembles, n_vars) or 5 for gridded "
        "(n_years, n_ensembles, n_lons, n_lats, n_vars) input."
    )
