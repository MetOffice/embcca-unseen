# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""EMBCCA multivariate bias adjustment.

Eigen-decomposition-based Multivariate Bias Correction with Covariance
Alignment. Model and observed data are standardised, the model is whitened in
its own principal-component basis, and then recoloured in the observed one::

    Zm = mod_st @ V @ Gamma^-1/2 @ Lambda^1/2 @ W.T

where ``V, Gamma`` and ``W, Lambda`` are the eigenvectors and eigenvalues of
the standardised model and observed covariance matrices. The result is
returned to physical units as ``Zm * mod_std + obs_mean``.

That final rescaling is the defining property of the method: the correction
transfers the observed *mean* and *correlation structure* onto the model while
retaining the model's own *variance*. Methods that transfer the observed
variance as well shrink the ensemble's sampled spread, which is precisely the
quantity UNSEEN analyses depend on.

Each ensemble member is corrected independently, and for gridded data each grid
cell is corrected independently as well.
"""

from __future__ import annotations

import numpy as np

__all__ = ["correct", "correct_area_mean", "correct_gridded"]


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


def correct_area_mean(mod: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Apply EMBCCA to area-mean (non-spatial) data.

    Parameters
    ----------
    mod : array, shape (n_years, n_ensembles, n_vars)
        Model data. Each ensemble member is corrected independently.
    obs : array, shape (n_years, n_vars)
        Observed data providing the target mean and correlation structure.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod``.

    Notes
    -----
    Standard deviations are not floored, so a variable with zero variance
    across time will divide by zero. Use :func:`correct_gridded`, which takes
    an ``eps`` floor, where that is a possibility.
    """
    eps = 1e-6
    mod_corrected = np.empty_like(mod)

    obs_st, obs_mean, _ = _standardise(obs)
    obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

    for member in range(mod.shape[1]):
        mod_st, _, mod_std = _standardise(mod[:, member, :])
        mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
        aligned = _align(
            mod_st, mod_eigenvalues, mod_eigenvectors, obs_eigenvalues, obs_eigenvectors
        )
        mod_corrected[:, member, :] = aligned * mod_std + obs_mean

    return mod_corrected



def correct_gridded(mod: np.ndarray, obs: np.ndarray, eps=1e-6) -> np.ndarray:
    """Apply EMBCCA to gridded data, independently at each grid cell.

    Parameters
    ----------
    mod : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
        Model data. Each ensemble member is corrected independently at each
        grid cell.
    obs : array, shape (n_years, n_lons, n_lats, n_vars)
        Observed data providing the target mean and correlation structure.
    eps : float, optional
        Floor applied to standard deviations and eigenvalues, to avoid
        division by zero on cells with little or no variance.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod``.
    """
    _, n_ensembles, n_lons, n_lats, _ = mod.shape
    mod_corrected = np.empty_like(mod)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            obs_st, obs_mean, _ = _standardise(obs[:, ilon, ilat, :], eps)
            obs_eigenvalues, obs_eigenvectors = _covariance_eigen(obs_st, eps)

            for member in range(n_ensembles):
                mod_st, _, mod_std = _standardise(mod[:, member, ilon, ilat, :], eps)
                mod_eigenvalues, mod_eigenvectors = _covariance_eigen(mod_st, eps)
                aligned = _align(
                    mod_st,
                    mod_eigenvalues,
                    mod_eigenvectors,
                    obs_eigenvalues,
                    obs_eigenvectors,
                )
                mod_corrected[:, member, ilon, ilat, :] = aligned * mod_std + obs_mean

    return mod_corrected



def correct(mod: np.ndarray, obs: np.ndarray, **kwargs) -> np.ndarray:
    """Apply EMBCCA, dispatching on the layout of ``mod``.

    Parameters
    ----------
    mod : array
        Either ``(n_years, n_ensembles, n_vars)`` for area-mean data or
        ``(n_years, n_ensembles, n_lons, n_lats, n_vars)`` for gridded data.
    obs : array
        Correspondingly ``(n_years, n_vars)`` or
        ``(n_years, n_lons, n_lats, n_vars)``.
    **kwargs
        Passed through to the selected implementation. Only the gridded form
        accepts ``eps``; supplying it with area-mean input raises
        ``TypeError``.

    Returns
    -------
    mod_corrected : array
        Same shape and dtype as ``mod``.

    See Also
    --------
    correct_area_mean : Area-mean form, called for 3-D input.
    correct_gridded : Gridded form, called for 5-D input.

    Notes
    -----
    The two spatial axes of the gridded form are treated symmetrically -- the
    correction is applied independently at each cell -- so their order does not
    affect the result.
    """
    ndim = np.ndim(mod)
    if ndim == 3:
        return correct_area_mean(mod, obs, **kwargs)
    if ndim == 5:
        return correct_gridded(mod, obs, **kwargs)
    raise ValueError(
        f"Unsupported dimensions: mod.ndim={ndim}. Expected 3 for area-mean "
        "(n_years, n_ensembles, n_vars) or 5 for gridded "
        "(n_years, n_ensembles, n_lons, n_lats, n_vars) input."
    )
