#!/usr/bin/env python3
# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""How much faster EMBCCA becomes when the per-cell loops are vectorised.

The `embcca` package applies the transform with an explicit loop over grid
cells and ensemble members. That is deliberate: the SBCK methods compared
against in the manuscript (dOTC, MRec, R2D2) can only be applied that way, so
EMBCCA uses the same structure to keep the published timing comparison fair.

This script asks the separate question of what EMBCCA costs once that
constraint is lifted. It holds a batched implementation alongside the packaged
one, checks the two agree, and times:

    EMBCCA-UNSEEN (vectorised)   the inline implementation below
    EMBCCA-UNSEEN (loop)         embcca.bias_adjust_gridded_unseen
    dOTC, MRec, R2D2             SBCK, per cell and member

Everything runs on one grid, the size of the China domain in the manuscript.
Both EMBCCA implementations are linear in the number of (cell, member) blocks,
so vectorising buys a constant factor rather than a better scaling; one
representative grid size therefore says as much as a sweep would.

    python paper/vectorisation.py

Each method is timed once. At this grid size the SBCK methods take one to
three minutes each, so repeated runs would add little beyond runtime. SBCK is
optional, and without it the two EMBCCA implementations are still compared, in
seconds.
"""

from __future__ import annotations

import time

import numpy as np

import embcca

# =============================================================================
# USER SETTINGS (edit these to run the workflow)
# =============================================================================

N_YEARS = 30        # length of the calibration period
N_MEMBERS = 20      # ensemble members, halved by the train/test split below 
                    # (100 would be the same as the manuscript)
N_VARS = 2          # variables, e.g. (precipitation, temperature)

# The China domain used in the manuscript.
N_LONS = 69
N_LATS = 74

SEED = 42

# Methods to include. Drop entries to shorten the run; these dominate it.
SBCK_METHODS = ["dOTC", "MRec", "R2D2"]

# Tolerance for the agreement check between the two EMBCCA implementations.
RTOL = 1e-9

# =============================================================================

EPS = 1e-6

VECTORISED = "EMBCCA-UNSEEN (vectorised)"
LOOP = "EMBCCA-UNSEEN (loop)"


# ---------------------------------------------------------------------------
# Vectorised EMBCCA
# ---------------------------------------------------------------------------


def _batched_cov(standardised, n_years):
    """``np.cov(block, rowvar=False)`` for a whole stack of blocks.

    ``np.cov`` re-centres its input before taking the product. Doing the same
    here is not cosmetic: skipping it changes the covariance in the last bits,
    and ``np.linalg.eigh`` can amplify that into an eigenvector sign flip,
    which does not cancel in the whitening/recolouring below.
    """
    centred = standardised - standardised.mean(axis=1, keepdims=True)
    return centred.transpose(0, 2, 1) @ centred / (n_years - 1)


def embcca_vectorised(mod, obs, eps=EPS):
    """Batched equivalent of ``embcca.bias_adjust_gridded_unseen``.

    Each grid cell and ensemble member is an independent ``(n_years, n_vars)``
    problem, so they can be stacked along a leading batch axis and handed to
    ``np.linalg.eigh``, which decomposes a whole stack of matrices in one call.
    The Python loops disappear; the arithmetic is unchanged.

    Variable names follow the original per-cell implementation so the two can
    be read side by side. The differences are that every quantity carries a
    leading batch axis, and that ``Lambda_sqrt`` and ``Gamma_inv_sqrt`` are
    held as diagonals rather than as matrices built with ``np.diag``, so they
    apply by elementwise multiplication.

    Parameters
    ----------
    mod : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
    obs : array, shape (n_years, n_lons, n_lats, n_vars)
    eps : float
        Small number to avoid divide-by-zero and negative/zero eigenvalues.

    Returns
    -------
    mod_corrected : array, same shape as mod

    Notes
    -----
    Speed is bought with memory. The looped implementation holds one
    ``(n_years, n_vars)`` block at a time, so its footprint is essentially the
    input and output arrays. This version instead materialises every block at
    once, several times over: ``mod_cells``, ``mod_st``, the centred copy
    inside :func:`_batched_cov`, each intermediate ``Zm``, and the result are
    all ``n_ensembles * n_cells * n_years * n_vars`` elements. At eight bytes
    per element that is roughly 120 MB apiece for the China domain with 50
    adjusted members, and several are live simultaneously.

    If that does not fit, process the ensemble axis in chunks and concatenate
    the results: each chunk is still batched, so most of the speedup survives,
    and peak memory falls by roughly the number of chunks. Grid cells could be
    chunked instead, though the ensemble axis is usually the longer one and
    splits without touching the observed quantities, which are shared across
    members at a given cell.
    """
    n_years, n_ensembles, n_lons, n_lats, n_vars = mod.shape
    n_cells = n_lons * n_lats

    # --- OBS at every grid cell at once: (n_cells, time, vars) ---
    obs_cells = np.moveaxis(obs, 0, -2).reshape(n_cells, n_years, n_vars)

    # mean/std over time for each variable
    obs_mean = np.mean(obs_cells, axis=1)
    obs_std = np.std(obs_cells, axis=1)

    # avoid divide-by-zero
    obs_std = np.where(obs_std < eps, eps, obs_std)

    # standardise
    obs_st = (obs_cells - obs_mean[:, None, :]) / obs_std[:, None, :]

    # covariance across variables, per cell (n_cells, vars, vars)
    Cov_obs = _batched_cov(obs_st, n_years)

    # eigen-decomp, batched over cells
    eigenvalues_obs, W = np.linalg.eigh(Cov_obs)
    eigenvalues_obs = np.maximum(eigenvalues_obs, eps)
    Lambda_sqrt = np.sqrt(eigenvalues_obs)

    # --- MOD at every grid cell and ensemble at once ---
    mod_cells = np.moveaxis(mod, 0, -2).reshape(n_ensembles * n_cells, n_years, n_vars)

    mod_mean = np.mean(mod_cells, axis=1)
    mod_std = np.std(mod_cells, axis=1)
    mod_std = np.where(mod_std < eps, eps, mod_std)

    mod_st = (mod_cells - mod_mean[:, None, :]) / mod_std[:, None, :]

    Cov_mod = _batched_cov(mod_st, n_years)

    eigenvalues_mod, V = np.linalg.eigh(Cov_mod)
    eigenvalues_mod = np.maximum(eigenvalues_mod, eps)
    Gamma_inv_sqrt = 1.0 / np.sqrt(eigenvalues_mod)

    # The observed quantities are shared by every ensemble member at a given
    # cell, so repeat them along the ensemble axis to line the batches up.
    batch = (n_ensembles, n_cells)
    Lambda_sqrt = np.broadcast_to(Lambda_sqrt, (*batch, n_vars)).reshape(-1, n_vars)
    W = np.broadcast_to(W, (*batch, n_vars, n_vars)).reshape(-1, n_vars, n_vars)
    obs_mean = np.broadcast_to(obs_mean, (*batch, n_vars)).reshape(-1, n_vars)

    # Whitening + recoloring:
    # Zm = mod_st @ V @ Gamma^-1/2 @ Lambda^1/2 @ W.T
    Zm = mod_st @ V
    Zm = Zm * Gamma_inv_sqrt[:, None, :]
    Zm = Zm * Lambda_sqrt[:, None, :]
    Zm = Zm @ W.transpose(0, 2, 1)

    # Back to observed scale (observed mean, model standard deviation)
    mod_corrected = Zm * mod_std[:, None, :] + obs_mean[:, None, :]

    return np.moveaxis(
        mod_corrected.reshape(n_ensembles, n_lons, n_lats, n_years, n_vars), -2, 0
    )


# ---------------------------------------------------------------------------
# SBCK, applied the way the manuscript applies it
# ---------------------------------------------------------------------------


def sbck_gridded(obs, mod_train, mod_test, handler):
    """Fit and predict per grid cell and ensemble member.

    Mirrors the spatial branch of ``apply_sbck_to_ensemble`` in the China
    analysis: one fit per (cell, member), trained on the held-in half of the
    ensemble and applied to the held-out half.
    """
    _, n_ensembles, n_lons, n_lats, _ = mod_test.shape
    corrected = np.empty_like(mod_test)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            Y0 = obs[:, ilon, ilat, :]
            for ens in range(n_ensembles):
                X0 = mod_train[:, ens, ilon, ilat, :]
                X1 = mod_test[:, ens, ilon, ilat, :]
                bc = handler()
                bc.fit(Y0, X0, X1)
                corrected[:, ens, ilon, ilat, :] = bc.predict(X1)

    return corrected


# ---------------------------------------------------------------------------
# Data, timing and reporting
# ---------------------------------------------------------------------------


def make_data(seed=SEED):
    """Synthetic model and observed fields on the China-sized grid.

    Synthetic rather than the manuscript data so the script runs without the
    restricted input files, and so the grid can be resized from the settings
    above.
    """
    rng = np.random.default_rng(seed)
    scale = np.array([40.0, 1.5])[:N_VARS] if N_VARS <= 2 else np.ones(N_VARS)
    offset = np.array([300.0, 22.0])[:N_VARS] if N_VARS <= 2 else np.zeros(N_VARS)

    mod = rng.normal(size=(N_YEARS, N_MEMBERS, N_LONS, N_LATS, N_VARS)) * scale + offset
    obs = rng.normal(size=(N_YEARS, N_LONS, N_LATS, N_VARS)) * scale * 0.7 + offset - 5.0
    return mod, obs


def split_ensemble(mod, seed=SEED):
    """Half the ensemble to train on, half to adjust, as the analysis does."""
    rng = np.random.default_rng(seed=seed)
    n_total = mod.shape[1]
    idx = rng.choice(n_total, size=n_total // 2, replace=False)
    rest = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)
    return mod[:, idx, ...], mod[:, rest, ...]


def time_once(fn):
    """Seconds for a single call.

    Nothing is measured cold: the agreement check above has already run both
    EMBCCA implementations on this data, so NumPy and its BLAS are warm before
    any timing starts.
    """
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def check_agreement(mod_test, obs):
    """The speedup only means anything if the answer is unchanged."""
    print("Agreement between the two EMBCCA implementations")
    reference = embcca.bias_adjust_gridded_unseen(mod_test, obs)
    batched = embcca_vectorised(mod_test, obs)

    difference = np.abs(batched - reference)
    relative = difference / np.maximum(np.abs(reference), 1e-300)
    blocks = relative.max(axis=(0, -1)).ravel()

    print(f"  bit-identical            : {np.array_equal(batched, reference)}")
    print(f"  max absolute difference  : {difference.max():.3e}")
    print(f"  max relative difference  : {relative.max():.3e}")
    print(f"  blocks differing > {RTOL:.0e}  : {int((blocks > RTOL).sum())}/{blocks.size}")

    if not np.allclose(batched, reference, rtol=RTOL, atol=0.0):
        raise SystemExit("\nVectorised output does not match the package. Not timing.")
    print("  -> agree within tolerance\n")


def get_sbck_handlers():
    """Return {name: constructor}, or an empty dict if SBCK is unavailable."""
    try:
        import SBCK
    except ImportError:
        print("SBCK not installed; comparing the two EMBCCA implementations only.\n")
        return {}
    return {name: getattr(SBCK, name) for name in SBCK_METHODS if hasattr(SBCK, name)}


def report(results, blocks):
    """One row per method, fastest first."""
    width = max(len(name) for name in results)
    base = min(results.values())

    header = f"{'method':{width}}  {'time':>10}  {'per block':>10}  {'relative':>9}"
    print(header)
    print("-" * len(header))
    for name, seconds in sorted(results.items(), key=lambda kv: kv[1]):
        print(
            f"{name:{width}}  {seconds * 1000:>9.1f}ms"
            f"  {seconds / blocks * 1e6:>9.1f}us  {seconds / base:>8.1f}x"
        )


def main():
    n_cells = N_LONS * N_LATS
    mod, obs = make_data()
    mod_train, mod_test = split_ensemble(mod)
    blocks = n_cells * mod_test.shape[1]

    print(f"grid {N_LONS}x{N_LATS} = {n_cells} cells (China domain) | {N_YEARS} years | "
          f"{N_MEMBERS} members, {mod_test.shape[1]} adjusted")
    print(f"{blocks} (cell, member) blocks per method\n")

    check_agreement(mod_test, obs)

    results = {
        VECTORISED: time_once(lambda: embcca_vectorised(mod_test, obs)),
        LOOP: time_once(lambda: embcca.bias_adjust_gridded_unseen(mod_test, obs)),
    }

    for name, handler in get_sbck_handlers().items():
        print(f"timing {name} (minutes)...", flush=True)
        results[name] = time_once(lambda h=handler: sbck_gridded(obs, mod_train, mod_test, h))

    print()
    report(results, blocks)


if __name__ == "__main__":
    main()
