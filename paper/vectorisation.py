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

It reads the same four files as `Multi-DePreSys4-Paper-area_full_final_China.py`
and prepares them the same way, so the timings correspond to the manuscript's
China domain. The observations must already be on the model grid; run
`paper/regrid_obs.py` first if the `*_regridded.nc` files do not exist.

    python paper/vectorisation.py

Each method is timed once. The SBCK methods take minutes each on this domain,
so repeated runs would add little beyond runtime. SBCK is optional, and without
it the two EMBCCA implementations are still compared, in seconds.
"""

from __future__ import annotations

import time
from pathlib import Path

import netCDF4 as nc
import numpy as np

import embcca

# =============================================================================
# USER SETTINGS (edit these to run the workflow)
# =============================================================================

DATA_DIR = "example-data/"          # directory containing input NetCDF files

START_YEAR = 1992
END_YEAR_EXCL = 2022   # -> 1992-2021 inclusive

# Filenames (relative to DATA_DIR)
TAS_MODEL_FILE = "China_1992_2023_summer_tas_model_DePreSys4.nc"
PR_MODEL_FILE = "China_1992_2023_summer_pr_model_DePreSys4.nc"
TAS_OBS_FILE = "China_1992_2023_summer_tas_obs_ERA5_Land_regridded.nc"
PR_OBS_FILE = "China_1992_2023_summer_pr_obs_ERA5_Land_regridded.nc"

# Variable names inside NetCDF
TAS_MODEL_VAR = "mean_jja_temperature"
PR_MODEL_VAR = "total_jja_precipitation"
TAS_OBS_VAR = "t2m"
PR_OBS_VAR = "tp"

# The China domain, as in the manuscript. Checked against the files on load.
N_LONS = 69
N_LATS = 74
N_ENSEMBLES = 100

# Methods to include. Drop entries to shorten the run; these dominate it.
SBCK_METHODS = ["dOTC", "MRec", "R2D2"]

# Tolerance for the agreement check between the two EMBCCA implementations.
RTOL = 1e-9

# =============================================================================

DATA_PATH = Path(DATA_DIR)

EPS = 1e-6
SEED = 42

VECTORISED = "EMBCCA-UNSEEN (vectorised)"
LOOP = "EMBCCA-UNSEEN (loop)"


# ---------------------------------------------------------------------------
# Vectorised EMBCCA
# ---------------------------------------------------------------------------


def _batched_cov(standardised, n_years):
    """``np.cov(block, rowvar=False)`` for a whole stack of blocks.

    Two details of ``np.cov`` have to be reproduced exactly, and neither is
    cosmetic. It promotes its input to float64 before doing anything, and it
    re-centres before taking the product. Departing from either changes the
    covariance in the last bits, and ``np.linalg.eigh`` can amplify that into
    an eigenvector sign flip, which does not cancel in the whitening and
    recolouring below. The model data is float32, so the promotion matters
    most: without it the two implementations disagree by more than 100% on a
    large fraction of blocks.
    """
    centred = standardised.astype(np.float64, copy=False)
    centred = centred - centred.mean(axis=1, keepdims=True)
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

    Masked cells arrive as NaN and stay NaN: ``np.linalg.eigh`` propagates them
    rather than raising, so ocean cells cost time but produce no output.
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

    # The package writes into np.empty_like(mod), so the result carries the
    # input dtype however the intermediates were promoted.
    mod_corrected = mod_corrected.astype(mod.dtype, copy=False)

    return np.moveaxis(
        mod_corrected.reshape(n_ensembles, n_lons, n_lats, n_years, n_vars), -2, 0
    )


# ---------------------------------------------------------------------------
# SBCK, applied the way the manuscript applies it
# ---------------------------------------------------------------------------


def sbck_gridded(obs, mod_train, mod_test, handler):
    """Fit and predict per grid cell and ensemble member.

    Mirrors the spatial branch of ``apply_sbck_to_ensemble`` in the China
    analysis, including its handling of missing data: a cell is skipped
    entirely when the observations have fewer than five valid timesteps, and a
    member is skipped when the overlap between valid observed and model
    timesteps is that short. Ocean cells therefore cost SBCK nothing, whereas
    the EMBCCA implementations run over every cell and return NaN for them.
    """
    _, n_ensembles, n_lons, n_lats, _ = mod_test.shape
    corrected = np.full_like(mod_test, np.nan)

    for ilon in range(n_lons):
        for ilat in range(n_lats):
            Y0_full = obs[:, ilon, ilat, :]

            valid_obs = np.all(np.isfinite(Y0_full), axis=-1)
            if valid_obs.sum() < 5:
                continue

            for ens in range(n_ensembles):
                X0_full = mod_train[:, ens, ilon, ilat, :]
                X1_full = mod_test[:, ens, ilon, ilat, :]

                valid = (
                    valid_obs
                    & np.all(np.isfinite(X0_full), axis=-1)
                    & np.all(np.isfinite(X1_full), axis=-1)
                )
                if valid.sum() < 5:
                    continue

                bc = handler()
                bc.fit(Y0_full[valid, :], X0_full[valid, :], X1_full[valid, :])
                corrected[valid, ens, ilon, ilat, :] = bc.predict(X1_full[valid, :])

    return corrected


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _read(fname, var):
    ds = nc.Dataset(str(DATA_PATH / fname), mode="r")
    try:
        return ds.variables[var][:]
    finally:
        ds.close()


def load_data():
    """Load and prepare the China inputs, as the manuscript analysis does.

    Returns ``(obs_combined, mod_raw, mask)`` with the variable axis ordered
    (precipitation, temperature), masked cells filled with NaN, and years
    restricted to the analysis period.
    """
    tas_model = _read(TAS_MODEL_FILE, TAS_MODEL_VAR)
    pr_model = _read(PR_MODEL_FILE, PR_MODEL_VAR)
    tas_obs = _read(TAS_OBS_FILE, TAS_OBS_VAR)
    pr_obs = _read(PR_OBS_FILE, PR_OBS_VAR)

    if tas_obs.shape[1:] != tas_model.shape[-2:]:
        raise SystemExit(
            f"Observations are on a {tas_obs.shape[1:]} grid but the model is on "
            f"{tas_model.shape[-2:]}. Run paper/regrid_obs.py first."
        )

    # restrict to years of interest
    all_years = np.arange(1992, 2023)
    years = np.arange(START_YEAR, END_YEAR_EXCL)
    years_ind = np.where(np.isin(all_years, years))[0]

    tas_model, pr_model = tas_model[years_ind, :], pr_model[years_ind, :]
    tas_obs, pr_obs = tas_obs[years_ind], pr_obs[years_ind]

    # collapse realisation x leadtime into a single ensemble axis
    def reshape(simulations):
        return simulations.reshape(
            simulations.shape[0], N_ENSEMBLES, simulations.shape[3], simulations.shape[4]
        )

    tas_flat, pr_flat = reshape(tas_model), reshape(pr_model)

    # combined mask, from the observations' own mask and the model's zero fill
    mask_obs = np.all(np.ma.getmaskarray(tas_obs), axis=0)
    mask_model = (tas_flat == 0.0)[0, 0, :, :]
    mask = mask_obs | mask_model

    obs_combined = np.stack((pr_obs, tas_obs), axis=-1)
    obs_combined = np.ma.array(
        obs_combined,
        mask=np.broadcast_to(mask[None, ..., None], obs_combined.shape),
        copy=False,
    )

    mod_raw = np.ma.stack((pr_flat, tas_flat), axis=-1)
    mod_raw = np.ma.array(
        mod_raw,
        mask=np.broadcast_to(mask[None, None, ..., None], mod_raw.shape),
        copy=False,
    )

    return obs_combined.filled(np.nan), mod_raw.filled(np.nan), mask


def split_ensemble(mod, seed=SEED):
    """Half the ensemble to train on, half to adjust, as the analysis does."""
    rng = np.random.default_rng(seed=seed)
    n_total = mod.shape[1]
    idx = rng.choice(n_total, size=n_total // 2, replace=False)
    rest = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)
    return mod[:, idx, ...], mod[:, rest, ...]


# ---------------------------------------------------------------------------
# Timing and reporting
# ---------------------------------------------------------------------------


def time_once(fn):
    """Seconds for a single call.
    """
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def check_agreement(mod_test, obs):
    """The speedup only means anything if the answer is unchanged."""
    print("Agreement between the two EMBCCA implementations")
    reference = embcca.bias_adjust_gridded_unseen(mod_test, obs)
    batched = embcca_vectorised(mod_test, obs)

    finite = np.isfinite(reference)
    same_pattern = np.array_equal(finite, np.isfinite(batched))

    difference = np.abs(batched[finite] - reference[finite])
    relative = difference / np.maximum(np.abs(reference[finite]), 1e-300)

    print(f"  matching NaN pattern     : {same_pattern}")
    print(f"  finite values compared   : {finite.sum()}")
    print(f"  max absolute difference  : {difference.max():.3e}")
    print(f"  max relative difference  : {relative.max():.3e}")
    print(f"  values differing > {RTOL:.0e}  : {int((relative > RTOL).sum())}")

    if not (same_pattern and np.allclose(batched[finite], reference[finite], rtol=RTOL)):
        raise SystemExit("\nVectorised output does not match the package.")
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
    """One row per method, fastest first.

    `blocks` gives the (cell, member) count each method actually processed,
    which differs between EMBCCA and SBCK because SBCK skips masked cells.
    """
    width = max(len(name) for name in results)
    base = results[LOOP]

    header = (
        f"{'method':{width}}  {'time':>10}  {'blocks':>9}  {'per block':>10}  {'relative':>9}"
    )
    print(header)
    print("-" * len(header))
    for name, seconds in sorted(results.items(), key=lambda kv: kv[1]):
        n = blocks[name]
        print(
            f"{name:{width}}  {seconds * 1000:>9.1f}ms  {n:>9}"
            f"  {seconds / n * 1e6:>9.1f}us  {seconds / base:>8.3f}x"
        )


def main():
    obs, mod, mask = load_data()
    mod_train, mod_test = split_ensemble(mod)

    n_years, n_adjusted = mod_test.shape[0], mod_test.shape[1]
    n_cells = mask.size
    n_valid = int((~mask).sum())

    print(f"grid {mask.shape[0]}x{mask.shape[1]} = {n_cells} cells, "
          f"{n_valid} unmasked ({n_valid / n_cells:.0%} land)")
    print(f"{n_years} years | {N_ENSEMBLES} members, {n_adjusted} adjusted\n")

    results = {
        VECTORISED: time_once(lambda: embcca_vectorised(mod_test, obs)),
        LOOP: time_once(lambda: embcca.bias_adjust_gridded_unseen(mod_test, obs)),
    }

    check_agreement(mod_test, obs)

    # EMBCCA runs over every cell; SBCK skips the masked ones.
    blocks = {name: n_cells * n_adjusted for name in results}

    for name, handler in get_sbck_handlers().items():
        print(f"timing {name} (minutes)...", flush=True)
        results[name] = time_once(lambda h=handler: sbck_gridded(obs, mod_train, mod_test, h))
        blocks[name] = n_valid * n_adjusted

    print()
    report(results, blocks)


if __name__ == "__main__":
    main()
