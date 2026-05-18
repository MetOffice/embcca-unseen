# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

import netCDF4 as nc
import numpy as np
import seaborn as sns
import random
import matplotlib as mpl


# =============================================================================
# USER SETTINGS (edit these to run the workflow)
# =============================================================================

DATA_DIR = "/path/to/data"          # directory containing input NetCDF files
OUTDIR   = "/path/to/output/China"  # directory where figures will be saved
SEED     = 42                       # reproducibility seed

START_YEAR    = 1992
END_YEAR_EXCL = 2022   # -> 1992–2021 inclusive

# Filenames (relative to DATA_DIR)
TAS_MODEL_FILE = "China_1992_2023_summer_tas_model_DePreSys4.nc"
PR_MODEL_FILE  = "China_1992_2023_summer_pr_model_DePreSys4.nc"
TAS_OBS_FILE   = "China_1992_2023_summer_tas_obs_ERA5_Land_regridded.nc"
PR_OBS_FILE    = "China_1992_2023_summer_pr_obs_ERA5_Land_regridded.nc"

# Variable names inside NetCDF
TAS_MODEL_VAR = "mean_jja_temperature"
PR_MODEL_VAR  = "total_jja_precipitation"
TAS_OBS_VAR   = "t2m"
PR_OBS_VAR    = "tp"

# Assumed ensemble size per year (used in reshape)
N_ENSEMBLE_PER_YEAR = 100

# Plotting backend (use Agg for headless/HPC)
MPL_BACKEND = "Agg"

# =============================================================================
mpl.use(MPL_BACKEND)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth, Reader
from cartopy.feature import ShapelyFeature

from matplotlib.lines import Line2D
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Derived paths (do not edit)
DATA_PATH = Path(DATA_DIR)
import iris
import time
from SBCK import QDM, CDFt, R2D2, dOTC, MRec

# import functions
from fidelity_test_cube import FidelityTestCube
ftc = FidelityTestCube()
outdir = OUTDIR
# getting model info
def _get_extent(template_cube):
    lats = template_cube.coord("latitude").points
    lons = template_cube.coord("longitude").points
    return [float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))]

dir = '/data/users/cst/Projects/CSSP/CSSP_China/FY2526/Yiwei_paper/data/'
cube = iris.load_cube(str(DATA_PATH / TAS_OBS_FILE))
lats = cube.coord('latitude').points
lons = cube.coord('longitude').points

# load model data
tas_model_nc = nc.Dataset(str(DATA_PATH / TAS_MODEL_FILE), mode='r')
pr_model_nc = nc.Dataset(str(DATA_PATH / PR_MODEL_FILE), mode='r')
tas_obs_nc = nc.Dataset(str(DATA_PATH / TAS_OBS_FILE), mode='r')
pr_obs_nc = nc.Dataset(str(DATA_PATH / PR_OBS_FILE), mode='r')

tas_model = tas_model_nc.variables['mean_jja_temperature'][:]
pr_model = pr_model_nc.variables['total_jja_precipitation'][:]
tas_obs = tas_obs_nc.variables['t2m'][:]
pr_obs = pr_obs_nc.variables['tp'][:]

# # average over lat and lon for model data
# tas_model = tas_model.mean(axis=(-2,-1))
# pr_model = pr_model.mean(axis=(-2,-1))
# tas_obs = tas_obs.mean(axis=(-2,-1))
# pr_obs = pr_obs.mean(axis=(-2,-1))

# restrict to years of interest
all_years = np.arange(1992, 2023)
years = np.arange(START_YEAR, END_YEAR_EXCL)
hist_years = np.arange(1962, 1992)
years_ind = np.where(np.isin(all_years, years))[0]
hist_years_ind = np.where(np.isin(all_years, hist_years))[0]
date_dict = {'all_dates': all_years, 'dates': years, 'hist_dates': hist_years, 'dates_ind': years_ind, 'hist_dates_ind': hist_years_ind}

tas_model = tas_model[date_dict['dates_ind'],:]
pr_model = pr_model[date_dict['dates_ind'],:]
tas_obs = tas_obs[date_dict['dates_ind']]
pr_obs = pr_obs[date_dict['dates_ind']]

# new reshape simulations from 3 dimensions to 2 dimensions
def new_reshape_model(simulations):
    n_years = simulations.shape[0]
    n_lats = simulations.shape[3]
    n_lons = simulations.shape[4]
    new_model = simulations.reshape(n_years, N_ENSEMBLE_PER_YEAR, n_lats, n_lons)
    return new_model

tas_flat = new_reshape_model(tas_model)
pr_flat = new_reshape_model(pr_model)

# Create combined mas, removing all missing values from both obs and model
mask_obs = np.all(np.ma.getmaskarray(tas_obs), axis=0) #copying mask over
mask_model = tas_flat == 0.0
mask_model = mask_model[0,0,:,:]
mask = mask_obs | mask_model

## Multivariate data ##
# 1. combine obs
obs_combined = np.stack((pr_obs, tas_obs), axis=-1)
mask4d = np.broadcast_to(mask[None, ..., None], obs_combined.shape)
# Applying mask
obs_combined = np.ma.array(obs_combined, mask=mask4d, copy=False)

# 2a. combine model data
mod_raw = np.ma.stack((pr_flat, tas_flat), axis=-1)
mask5d = np.broadcast_to(mask[None, None, ..., None], mod_raw.shape)
mod_raw = np.ma.array(mod_raw, mask=mask5d, copy=False)

## Bias Correction: Yiweh's method ##
# correct mean and correlation, preserve the variance

def multi_correction_eigen_per_ensemble(mod, obs, eps=1e-6):
    """
    Eigen-based multivariate correction per ensemble member, applied independently
    at each (lon, lat) grid cell.

    Parameters
    ----------
    mod : array, shape (n_years, n_ensembles, n_lons, n_lats, n_vars)
    obs : array, shape (n_years, n_lons, n_lats, n_vars)
    eps : float
        Small number to avoid divide-by-zero and negative/zero eigenvalues.

    Returns
    -------
    mod_corrected : array, same shape as mod
    """

    n_years, n_ensembles, n_lons, n_lats, n_vars = mod.shape
    mod_corrected = np.empty_like(mod)

    # Loop over grid
    for ilon in range(n_lons):
        for ilat in range(n_lats):

            # --- OBS at this grid cell: (time, vars) ---
            obs_cell = obs[:, ilon, ilat, :]  # shape (n_years, n_vars)

            # mean/std over time for each variable
            obs_mean = np.mean(obs_cell, axis=0)
            obs_std  = np.std(obs_cell, axis=0)

            # avoid divide-by-zero
            obs_std = np.where(obs_std < eps, eps, obs_std)

            # standardise
            obs_st = (obs_cell - obs_mean) / obs_std

            # covariance across variables (vars x vars)
            Cov_obs = np.cov(obs_st, rowvar=False)

            # eigen-decomp
            eigenvalues_obs, W = np.linalg.eigh(Cov_obs)
            eigenvalues_obs = np.maximum(eigenvalues_obs, eps)
            Lambda_sqrt = np.diag(np.sqrt(eigenvalues_obs))

            # Loop over ensembles for this grid cell
            for ens in range(n_ensembles):

                # --- MOD at this grid cell and ensemble: (time, vars) ---
                mod_cell = mod[:, ens, ilon, ilat, :]  # (n_years, n_vars)

                mod_mean = np.mean(mod_cell, axis=0)
                mod_std  = np.std(mod_cell, axis=0)
                mod_std  = np.where(mod_std < eps, eps, mod_std)

                mod_st = (mod_cell - mod_mean) / mod_std

                Cov_mod = np.cov(mod_st, rowvar=False)

                eigenvalues_mod, V = np.linalg.eigh(Cov_mod)
                eigenvalues_mod = np.maximum(eigenvalues_mod, eps)
                Gamma_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_mod))

                # Whitening + recoloring:
                # Zm = mod_st @ V @ Gamma^-1/2 @ Lambda^1/2 @ W.T
                Zm = mod_st.data @ V
                Zm = Zm @ Gamma_inv_sqrt
                Zm = Zm @ Lambda_sqrt
                Zm = Zm @ W.T

                # Back to observed scale (note: your original uses obs_mean + mod_std scaling)
                mod_corrected[:, ens, ilon, ilat, :] = Zm * mod_std + obs_mean

    return mod_corrected

# mod_corrected_eigen = multi_correction_eigen_per_ensemble(mod_raw, obs_combined)

## Comparison with other multivariate bias-adjustment methods using SBCK tool ##

def get_bc_handler(bc_type):
    if bc_type == 'MBCn':
        return MBCn()
    elif bc_type == 'R2D2':
        return R2D2()
    elif bc_type == 'dOTC':
        return dOTC()
    elif bc_type == 'MRec':
        return MRec()
    else:
        return None


def _to_ndarray(a, fill_value=np.nan):
    """Convert masked arrays to ndarray, filling masked values if needed."""
    if np.ma.isMaskedArray(a):
        return a.filled(fill_value)
    return np.asarray(a)


def apply_sbck_to_ensemble(obs, model_ensemble, bc_type, seed=42, eps=1e-6):
    """
    Apply SBCK bias correction per ensemble member.
    Supports:
      - non-spatial:
          obs  : (T, V)
          model: (T, E, V)
      - spatial:
          obs  : (T, X, Y, V)   (lon, lat)
          model: (T, E, X, Y, V)

    Returns corrected arrays shaped like X_test (i.e. the "held-out half"):
      - non-spatial: (T, E/2, V)
      - spatial    : (T, E/2, X, Y, V)
    """

    obs = _to_ndarray(obs, fill_value=np.nan)
    model_ensemble = _to_ndarray(model_ensemble, fill_value=np.nan)

    rng = np.random.default_rng(seed=seed)

    n_total = model_ensemble.shape[1]
    n_select = int(n_total / 2)

    if 2 * n_select != n_total:
        raise ValueError(
            f"Expected an even number of ensembles so train/test halves match. "
            f"Got n_total={n_total}."
        )

    idx = rng.choice(n_total, size=n_select, replace=False)
    idx_comp = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)

    # Split ensemble dimension
    X_train = model_ensemble[:, idx, ...]
    X_test  = model_ensemble[:, idx_comp, ...]

    # ---- If EMBCCA-UNSEEN: use your eigen method (already works for spatial) ----
    if bc_type == 'EMBCCA-UNSEEN':
        # Your multi_correction_eigen_per_ensemble expects (T,E,lon,lat,V) and obs (T,lon,lat,V)
        # and returns same shape as X_test
        return multi_correction_eigen_per_ensemble(X_test, obs)

    SBCK = get_bc_handler(bc_type)
    if SBCK is None:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    # ---- Non-spatial case: (T,E,V) ----
    if model_ensemble.ndim == 3:
        T, E_half, V = X_test.shape
        corrected = np.empty_like(X_test, dtype=float)

        Y0 = np.asarray(obs)  # (T,V)

        for e in range(E_half):
            X0 = np.asarray(X_train[:, e, :])
            X1 = np.asarray(X_test[:,  e, :])

            SBCK = get_bc_handler(bc_type)  # new instance each time
            SBCK.fit(Y0, X0, X1)
            corrected[:, e, :] = SBCK.predict(X1)

        return corrected

    # ---- Spatial case: (T,E,lon,lat,V) ----
    if model_ensemble.ndim == 5:
        T, E_half, n_lons, n_lats, V = X_test.shape
        corrected = np.full_like(X_test, np.nan, dtype=float)  # default NaNs

        for ilon in range(n_lons):
            for ilat in range(n_lats):

                Y0_full = obs[:, ilon, ilat, :]  # (T,V)

                # timesteps where obs is finite for ALL variables
                valid_obs = np.all(np.isfinite(Y0_full), axis=-1)
                if valid_obs.sum() < 5:  # or a higher threshold
                    continue

                for e in range(E_half):
                    X0_full = X_train[:, e, ilon, ilat, :]
                    X1_full = X_test[:, e, ilon, ilat, :]

                    valid = (
                            valid_obs
                            & np.all(np.isfinite(X0_full), axis=-1)
                            & np.all(np.isfinite(X1_full), axis=-1)
                    )

                    if valid.sum() < 5:
                        continue

                    Y0 = Y0_full[valid, :]
                    X0 = X0_full[valid, :]
                    X1 = X1_full[valid, :]

                    SBCK = get_bc_handler(bc_type)
                    SBCK.fit(Y0, X0, X1)

                    # predict only for valid timesteps, leave others NaN
                    corrected[valid, e, ilon, ilat, :] = SBCK.predict(X1)

        return corrected

    raise ValueError(
        f"Unsupported dimensions: obs.ndim={obs.ndim}, model_ensemble.ndim={model_ensemble.ndim}. "
        "Expected (T,V)/(T,E,V) or (T,X,Y,V)/(T,E,X,Y,V)."
    )


# bc_method_names = ['EMBCCA-UNSEEN', 'dOTC', 'MRec', 'R2D2', 'MBCn']
bc_method_names = ['EMBCCA-UNSEEN', 'dOTC', 'MRec', 'R2D2']
mod_corrected_sbck_list = list()
mod_corrected_sbck_list_mean = list()
timings = {}  # Timing how long each bias-adjustment method takes

for bc_method_name in bc_method_names:
    # apply bias-adjustment
    start = time.perf_counter()
    mod_corrected_sbck = apply_sbck_to_ensemble(obs_combined, mod_raw, bc_method_name)
    timings[bc_method_name] = time.perf_counter() - start
    mod_corrected_sbck_list.append(mod_corrected_sbck)

for k, v in timings.items():
    print(f"{k:10s}: {v:8.2f} s")

# Averaging across time and ensemble member (for models) for plotting of maps further below
for bc_method_name in bc_method_names:
    mod_corrected_sbck_time_avg = mod_corrected_sbck.mean(axis=0)
    mod_corrected_sbck_time_avg_ensemble_mean = mod_corrected_sbck_time_avg.mean(axis=0)
    mod_corrected_sbck_list_mean.append(mod_corrected_sbck_time_avg_ensemble_mean)

obs_combined_time_avg = obs_combined.mean(axis=0)
mod_raw_time_avg = mod_raw.mean(axis=0)
mod_raw_time_avg_ensemble_mean = mod_raw_time_avg.mean(axis=0)


def china_boundary_feature(linewidth=1.2, edgecolor='k'):
    # Admin-0 country boundaries from Natural Earth
    shp = natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
    geoms = []
    for rec in Reader(shp).records():
        # Natural Earth uses NAME_LONG / ADMIN depending on the file/version
        name = (rec.attributes.get('ADMIN') or rec.attributes.get('NAME_LONG') or '')
        if name in ('China', "People's Republic of China"):
            geoms.append(rec.geometry)

    return ShapelyFeature(geoms, ccrs.PlateCarree(),
                         facecolor='none', edgecolor=edgecolor, linewidth=linewidth)


def add_gridlabels(ax, left=False, bottom=False, fontsize=9, max_n=5):
    """
    Add cartopy gridlines, but only label left and/or bottom sides.
    """
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = bool(left)
    gl.bottom_labels = bool(bottom)

    gl.xlabel_style = {"size": fontsize}
    gl.ylabel_style = {"size": fontsize}

    # Keep labels from getting too dense
    gl.xlocator = mticker.MaxNLocator(max_n)
    gl.ylocator = mticker.MaxNLocator(max_n)

    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    return gl


def draw_map_panel(ax, Lon, Lat, data2d, title, *, proj=None, transform=None, cmap="viridis", vmin=None, vmax=None,
                   china_feat=None, extent=None, show_coastlines=True, coastline_res="10m", coastline_lw=0.6,
                   label_left=False, label_bottom=False):
    """
    Draw a single pcolormesh panel on a Cartopy axis and optionally overlay China boundary.
    Returns the QuadMesh from pcolormesh for colourbar creation.
    """
    if proj is None:
        proj = ccrs.PlateCarree()
    if transform is None:
        transform = proj

    m = ax.pcolormesh(Lon, Lat, data2d, cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)
    ax.set_title(title)

    if show_coastlines:
        ax.coastlines(resolution=coastline_res, linewidth=coastline_lw)
        ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, edgecolor="0.2")

    if china_feat is not None:
        ax.add_feature(china_feat)

    if extent is not None:
        ax.set_extent(extent, crs=transform)

    add_gridlabels(ax, left=label_left, bottom=label_bottom)
    return m

def plot_mean_map(obs, mod_raw, mod_sbck_list, var_name, lats, lons):

    if var_name == "precipitation":
        cbar_label = "(mm)"
        title = f"Total summer (JJA) precipitation {cbar_label}"
        cmap = "YlGnBu"
        vmax = 1200
        vmin = 0
        ind = 0
    elif var_name == "temperature":
        cbar_label = f"(\N{DEGREE SIGN}C)"
        title = f"Summer (JJA) mean temperature {cbar_label}"
        cmap = "YlOrRd"
        vmax = 32
        vmin = 0
        ind = 1
    else:
        raise ValueError(f"Unknown var_name: {var_name}")

    Lon, Lat = np.meshgrid(lons, lats)

    proj = ccrs.PlateCarree()
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 8),
        subplot_kw={"projection": proj},
        sharey=False
    )

    # Build feature once
    china_feat = china_boundary_feature(linewidth=1.3, edgecolor="black")

    # extent in lon/lat
    extent = [float(np.min(lons)), float(np.max(lons)), float(np.min(lats)), float(np.max(lats))]

    # helper to decide label placement
    def is_left_col(c): return c == 0
    def is_bottom_row(r): return r == (nrows - 1)

    # Panel calls: pass 2D slices explicitly
    m0 = draw_map_panel(
        axes[0, 0], Lon, Lat, obs[:, :, ind], "ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(0)
    )
    draw_map_panel(
        axes[0, 1], Lon, Lat, mod_raw[:, :, ind], "DePreSys model (raw)",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(0)
    )
    draw_map_panel(
        axes[0, 2], Lon, Lat, mod_sbck_list[0][:, :, ind], "EMBCCA-UNSEEN",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(2), label_bottom=is_bottom_row(0)
    )
    draw_map_panel(
        axes[1, 0], Lon, Lat, mod_sbck_list[1][:, :, ind], "dOTC",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(1)
    )
    draw_map_panel(
        axes[1, 1], Lon, Lat, mod_sbck_list[2][:, :, ind], "MRec",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(1)
    )
    draw_map_panel(
        axes[1, 2], Lon, Lat, mod_sbck_list[3][:, :, ind], "R2D2",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(2), label_bottom=is_bottom_row(1)
    )

    plt.tight_layout()

    cbar = fig.colorbar(m0, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(f"{title}", fontsize=18, y=1.02)

    fname = Path(f"{outdir}Maps/{var_name}.png")
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {fname}")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

plot_mean_map(obs_combined_time_avg, mod_raw_time_avg_ensemble_mean, mod_corrected_sbck_list_mean, 'precipitation', lats, lons)
plot_mean_map(obs_combined_time_avg, mod_raw_time_avg_ensemble_mean, mod_corrected_sbck_list_mean, 'temperature', lats, lons)

## Plot correlation between temperature and precipitation
def calc_corr(data):
    # Calculate for obs with shape (30, 11, 8, 2)
    x = data[..., 0]  # (30, 11, 8)
    y = data[..., 1]  # (30, 11, 8)

    # Pearson r across axis=0 (the 30 samples)
    xm = x - x.mean(axis=0, keepdims=True)
    ym = y - y.mean(axis=0, keepdims=True)
    corr = (xm * ym).sum(axis=0) / np.sqrt((xm**2).sum(axis=0) * (ym**2).sum(axis=0))

    return corr

# Calculate correlation
obs_corr = calc_corr(obs_combined)
mod_raw_corr = calc_corr(mod_raw)
mod_raw_corr = mod_raw_corr.mean(axis=0)

mod_corr_list = []
for mod in mod_corrected_sbck_list:
    mod_corr = calc_corr(mod)
    mod_corr = mod_corr.mean(axis=0)
    mod_corr_list.append(mod_corr)


def plot_corr_map(obs, mod_raw, mod_sbck_list, lats, lons):

    cbar_label = "Pearson correlation coefficient ($r$)"
    title = "Correlation between summer (JJA) temperature and precipitation ($r$)"
    cmap = "PuOr"
    vmax, vmin = 1, -1

    Lon, Lat = np.meshgrid(lons, lats)

    proj = ccrs.PlateCarree()
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 8),
        subplot_kw={"projection": proj},
        sharey=False
    )

    # Feature once
    china_feat = china_boundary_feature(linewidth=1.3, edgecolor="black")

    # extent in lon/lat
    extent = [
        float(np.min(lons)), float(np.max(lons)),
        float(np.min(lats)), float(np.max(lats))
    ]

    # helpers for label placement
    def is_left_col(c): return c == 0
    def is_bottom_row(r): return r == (nrows - 1)

    # --- top row ---
    m0 = draw_map_panel(
        axes[0, 0], Lon, Lat, obs, "ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(0)
    )

    draw_map_panel(
        axes[0, 1], Lon, Lat, mod_raw, "DePreSys model (raw)",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(0)
    )

    draw_map_panel(
        axes[0, 2], Lon, Lat, mod_sbck_list[0], "EMBCCA-UNSEEN",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(2), label_bottom=is_bottom_row(0)
    )

    # --- bottom row ---
    draw_map_panel(
        axes[1, 0], Lon, Lat, mod_sbck_list[1], "dOTC",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(1)
    )

    draw_map_panel(
        axes[1, 1], Lon, Lat, mod_sbck_list[2], "MRec",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(1)
    )

    draw_map_panel(
        axes[1, 2], Lon, Lat, mod_sbck_list[3], "R2D2",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(2), label_bottom=is_bottom_row(1)
    )

    plt.tight_layout()

    cbar = fig.colorbar(m0, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=18, y=1.02)

    fname = Path(f"{outdir}Maps/correlation.png")
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {fname}")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

plot_corr_map(obs_corr, mod_raw_corr, mod_corr_list, lats, lons)


def plot_corr_diff_map(obs, mod_raw, mod_sbck_list, lats, lons):
    """
    Plot differences in correlation coefficient relative to ERA5-Land obs:
        (model correlation) - (obs correlation)
    """

    # Differences relative to obs
    mod_raw_diff = mod_raw - obs
    mod_diff = [(m - obs) for m in mod_sbck_list]

    cbar_label = "Difference in Pearson correlation coefficient ($r$)"
    title = "Difference between observed and modelled correlation ($r$)"
    cmap = "seismic"
    vmin, vmax = -1, 1

    Lon, Lat = np.meshgrid(lons, lats)

    proj = ccrs.PlateCarree()
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(20, 8),
        subplot_kw={"projection": proj},
        sharey=False
    )

    # China outline + borders (as defined in your helpers)
    china_feat = china_boundary_feature(linewidth=1.3, edgecolor="black")

    # extent in lon/lat
    extent = [
        float(np.min(lons)), float(np.max(lons)),
        float(np.min(lats)), float(np.max(lats))
    ]

    # helpers for label placement
    def is_left_col(c): return c == 0
    def is_bottom_row(r): return r == (nrows - 1)

    # --- top row: 3 panels ---
    m0 = draw_map_panel(
        axes[0, 1], Lon, Lat, mod_raw_diff, "DePreSys model (raw) − ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(0)
    )

    draw_map_panel(
        axes[0, 2], Lon, Lat, mod_diff[0], "EMBCCA-UNSEEN − ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(0)
    )

    draw_map_panel(
        axes[1, 0], Lon, Lat, mod_diff[1], "dOTC − ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(2), label_bottom=is_bottom_row(0)
    )

    # --- bottom row: 2 panels + one blank ---
    draw_map_panel(
        axes[1, 1], Lon, Lat, mod_diff[2], "MRec − ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(0), label_bottom=is_bottom_row(1)
    )

    draw_map_panel(
        axes[1, 2], Lon, Lat, mod_diff[3], "R2D2 − ERA5-Land",
        proj=proj, transform=proj, cmap=cmap, vmin=vmin, vmax=vmax,
        china_feat=china_feat, extent=extent,
        label_left=is_left_col(1), label_bottom=is_bottom_row(1)
    )

    # Blank first panel (since only five entries)
    axes[0, 0].set_axis_off()

    plt.tight_layout()

    cbar = fig.colorbar(m0, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(title, fontsize=18, y=1.02)

    fname = Path(f"{outdir}Maps/correlation_diff.png")
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {fname}")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()


plot_corr_diff_map(obs_corr, mod_raw_corr, mod_corr_list, lats, lons)
