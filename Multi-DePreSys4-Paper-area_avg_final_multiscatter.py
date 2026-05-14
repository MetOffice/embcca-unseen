
import netCDF4 as nc
import numpy as np
import seaborn as sns
import random
import matplotlib as mpl


# =============================================================================
# USER SETTINGS (edit these to run the workflow)
# =============================================================================

DATA_DIR = "/path/to/data"          # directory containing input NetCDF files
OUTDIR   = "/path/to/output"        # directory where figures will be saved
SEED     = 42                       # reproducibility seed

START_YEAR    = 1992
END_YEAR_EXCL = 2022   # -> 1992–2021 inclusive

# Filenames (relative to DATA_DIR)
TAS_MODEL_FILE = "Hunan_1960_2024_summer_tas_model_DePreSys4.nc"
PR_MODEL_FILE  = "Hunan_1960_2024_summer_pr_model_DePreSys4.nc"
TAS_OBS_FILE   = "Hunan_1960_2024_summer_tas_obs_ERA5_Land.nc"
PR_OBS_FILE    = "Hunan_1960_2024_summer_pr_obs_ERA5_Land.nc"

# Variable names inside NetCDF
TAS_MODEL_VAR = "mean_jja_temperature"
PR_MODEL_VAR  = "total_jja_precipitation"
TAS_OBS_VAR   = "t2m"
PR_OBS_VAR    = "tp"

# Plotting backend (use Agg for headless/HPC)
MPL_BACKEND = "Agg"

# =============================================================================
mpl.use(MPL_BACKEND)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Derived paths (do not edit)
DATA_PATH = Path(DATA_DIR)
# --- SBCK imports (REMOVED MBCn) ---
from SBCK import QDM, CDFt, R2D2, dOTC, MRec

# import functions
from fidelity_test_cube import FidelityTestCube
ftc = FidelityTestCube()

outdir = OUTDIR
seed = SEED
# load model data
dir = DATA_DIR  # input data directory
tas_model_nc = nc.Dataset(str(DATA_PATH / TAS_MODEL_FILE), mode='r')
pr_model_nc = nc.Dataset(str(DATA_PATH / PR_MODEL_FILE), mode='r')
tas_obs_nc = nc.Dataset(str(DATA_PATH / TAS_OBS_FILE), mode='r')
pr_obs_nc = nc.Dataset(str(DATA_PATH / PR_OBS_FILE), mode='r')

tas_model = tas_model_nc.variables[TAS_MODEL_VAR][:]
pr_model = pr_model_nc.variables[PR_MODEL_VAR][:]
tas_obs = tas_obs_nc.variables[TAS_OBS_VAR][:]
pr_obs = pr_obs_nc.variables[PR_OBS_VAR][:]

# average over lat and lon for model data
tas_model = tas_model.mean(axis=(-2, -1))
pr_model = pr_model.mean(axis=(-2, -1))
tas_obs = tas_obs.mean(axis=(-2, -1))
pr_obs = pr_obs.mean(axis=(-2, -1))

# restrict to years of interest
all_years = np.arange(1960, 2025)
years = np.arange(START_YEAR, END_YEAR_EXCL)
hist_years = np.arange(1962, 1992)
years_ind = np.where(np.isin(all_years, years))[0]
hist_years_ind = np.where(np.isin(all_years, hist_years))[0]
date_dict = {
    'all_dates': all_years,
    'dates': years,
    'hist_dates': hist_years,
    'dates_ind': years_ind,
    'hist_dates_ind': hist_years_ind
}

tas_model = tas_model[date_dict['dates_ind'], :]
pr_model = pr_model[date_dict['dates_ind'], :]
tas_obs = tas_obs[date_dict['dates_ind']]
pr_obs = pr_obs[date_dict['dates_ind']]

# new reshape simulations from 3 dimensions to 2 dimensions
def new_reshape_model(simulations):
    n_years = simulations.shape[0]
    new_model = simulations.reshape(n_years, -1)
    return new_model

tas_flat = new_reshape_model(tas_model)
pr_flat = new_reshape_model(pr_model)

# Splitting data so that it works with SBCK's workflow, which assumes a past and future dataset
rng_split = np.random.default_rng(seed=seed)
n_total = tas_flat.shape[1]  # number of ensemble members
n_train = n_total // 2

idx_train = rng_split.choice(n_total, size=n_train, replace=False)
idx_test = np.setdiff1d(np.arange(n_total), idx_train, assume_unique=False)

# # Optional: keep deterministic ordering (nice for debugging/plots)
# idx_train = np.sort(idx_train)
# idx_test = np.sort(idx_test)

print(f"Ensemble split: n_total={n_total}, n_train={len(idx_train)}, n_test={len(idx_test)}")


## 1. Plotting original data vs. univariate correction of mean ##

def plotting_obs_mod(obs, mod, mod_corrected, var_name):
    obs_data = obs.data if hasattr(obs, "data") else obs
    mod_data = mod.data if hasattr(mod, "data") else mod
    mod_corrected_data = mod_corrected

    mod_mean = mod_data.mean(axis=1)
    mod_std = mod_data.std(axis=1)
    mod_corrected_mean = mod_corrected_data.mean(axis=1)
    mod_corrected_std = mod_corrected_data.std(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].plot(years, obs_data, label='Observations', color='black', linewidth=2, linestyle='--')
    for i in range(mod_data.shape[1]):
        axes[0].plot(years, mod_data[:, i], color='grey', alpha=0.2)
    axes[0].plot(years, mod_mean, label='DePreSys4_mean', color='red', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel(var_name)
    axes[0].set_title('Original Model')
    axes[0].legend()

    axes[1].plot(years, obs_data, label='Observations', color='black', linewidth=2, linestyle='--')
    for i in range(mod_corrected_data.shape[1]):
        axes[1].plot(years, mod_corrected_data[:, i], color='grey', alpha=0.2)
    axes[1].plot(years, mod_corrected_mean, label='DePreSys4_corrected_mean', color='blue', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel(var_name)
    axes[1].set_title('Corrected Model')
    axes[1].legend()

    plt.tight_layout()
    fname = Path(f'{outdir}/Line_plots/Original_vs_mean_adjusted_{var_name}.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()


def mean_bias_correction(obs, model, idx_train, idx_test):
    """
    Train mean-shift on training ensemble members, apply to test ensemble members.

    obs: (nyrs,)
    model: (nyrs, n_members)
    Returns:
        model_test_corrected: (nyrs, n_test)
    """
    # training subset (nyrs, n_train)
    model_train = model[:, idx_train]
    # test subset (nyrs, n_test)
    model_test = model[:, idx_test]

    # mean over years AND training members (scalar)
    model_mean_train = np.mean(model_train)
    obs_mean = np.mean(obs)

    correction_factor = obs_mean - model_mean_train
    model_test_corrected = model_test + correction_factor

    return model_test_corrected


tas_meancor = mean_bias_correction(tas_obs, tas_flat, idx_train, idx_test)
pr_meancor = mean_bias_correction(pr_obs,  pr_flat,  idx_train, idx_test)
plotting_obs_mod(tas_obs, tas_flat, tas_meancor, 'Temperature')
plotting_obs_mod(pr_obs, pr_flat, pr_meancor, 'Precipitation')

## Multivariate data ##

# 1. combine obs
obs_combined = np.column_stack((pr_obs, tas_obs))

# 2a. combine model data
mod_raw_full = np.empty((tas_flat.shape[0], n_total, 2))
mod_raw_full[:, :, 0] = pr_flat
mod_raw_full[:, :, 1] = tas_flat

mod_raw = np.empty((tas_flat.shape[0], len(idx_test), 2))
mod_raw[:, :, 0] = pr_flat[:, idx_test]
mod_raw[:, :, 1] = tas_flat[:, idx_test]

# 2b. combine univariate corrected data
meancor_combined = np.empty_like(mod_raw)
meancor_combined[:, :, 0] = pr_meancor
meancor_combined[:, :, 1] = tas_meancor


# 3. statistical info
print("\nObservation statistics:")
print("Mean (pr, tas):", obs_combined.mean(axis=0))
print("Std (pr, tas):", obs_combined.std(axis=0))

print("\nMean-corrected model statistics:")
print("Mean (pr, tas):", meancor_combined.mean(axis=(0, 1)))
print("Std (pr, tas):", meancor_combined.std(axis=(0, 1)))

print("\nRaw model statistics:")
print("Mean (pr, tas):", mod_raw.mean(axis=(0, 1)))
print("Std (pr, tas):", mod_raw.std(axis=(0, 1)))

# 4. correlation
obs_corr = np.corrcoef(obs_combined[:, 0], obs_combined[:, 1])[0, 1]
print("\nObservation correlation between pr and tas:", obs_corr)

n_members = mod_raw.shape[1]
mod_corrs = np.array([np.corrcoef(meancor_combined[:, i, 0], mod_raw[:, i, 1])[0, 1] for i in range(n_members)])
print("Model_cor correlation range:", mod_corrs.min(), "to", mod_corrs.max())

mod_rawcor = np.array([np.corrcoef(mod_raw[:, i, 0], mod_raw[:, i, 1])[0, 1] for i in range(n_members)])
print("Model_raw correlation range:", mod_rawcor.min(), "to", mod_rawcor.max())

obs_r = np.corrcoef(obs_combined[:, 0], obs_combined[:, 1])[0, 1]

def pearson_r(dat):
    """
    Pearson correlation between precip and temp for a (nyrs, n_members, 2) dataset,
    computed on the pooled cloud (flattened over years and ensemble members).
    """
    x = dat[:, :, 0].ravel()
    y = dat[:, :, 1].ravel()

    # remove NaNs if present
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan

    return np.corrcoef(x[m], y[m])[0, 1]



def plot_scatter_sixpanel(obs_combined, mod_raw, meancor_combined,
                          sbck_list, sbck_names, outdir,
                          xlim=(0, 1300), ylim=(20, 34),
                          fname='Scatter_plots/Scatter_sixpanel.png',
                          auto_limits=True, pad_frac=0.02):
    """
    Panel order:
      1) raw (test)
      2) univariate mean shift (test)
      3-6) SBCK methods (test), in sbck_names order

    If auto_limits=True, compute global x/y limits from min/max across ALL datasets
    included in the six-panel plot (including obs), and apply to all panels.
    pad_frac adds a small padding fraction of the data range on each side.
    """
    titles = ['DePreSys model (raw)', 'Univariate mean shift'] + list(sbck_names)
    data_list = [mod_raw, meancor_combined] + list(sbck_list)

    # ---- compute global limits across all datasets (and obs) ----
    if auto_limits:
        # Collect x/y from all model datasets
        xs = []
        ys = []
        for dat in data_list:
            xs.append(dat[:, :, 0].ravel())
            ys.append(dat[:, :, 1].ravel())

        # Include observations too (so obs never falls outside the panel)
        xs.append(obs_combined[:, 0].ravel())
        ys.append(obs_combined[:, 1].ravel())

        x_all = np.concatenate(xs)
        y_all = np.concatenate(ys)

        # Handle NaNs safely
        x_min = np.nanmin(x_all)
        x_max = np.nanmax(x_all)
        y_min = np.nanmin(y_all)
        y_max = np.nanmax(y_all)

        # Add small padding so points don't sit on the border
        x_rng = x_max - x_min
        y_rng = y_max - y_min

        # Guard against zero range (unlikely, but safe)
        if x_rng == 0:
            x_rng = 1.0
        if y_rng == 0:
            y_rng = 1.0

        x_pad = pad_frac * x_rng
        y_pad = pad_frac * y_rng

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        print(f"[sixpanel scatter] auto xlim={xlim}, auto ylim={ylim}")

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # Obs points
    obs_x = obs_combined[:, 0]
    obs_y = obs_combined[:, 1]

    for i in range(nrows * ncols):
        ax = axes[i]
        if i >= len(data_list):
            ax.axis("off")
            continue

        dat = data_list[i]  # (nyrs, n_members, 2)

        ax.scatter(dat[:, :, 0], dat[:, :, 1],
                   alpha=0.3, c='orange', s=36, edgecolors='none', label='Model (members)')
        ax.scatter(obs_x, obs_y,
                   alpha=0.8, c='#1f77b4', s=36, edgecolors='none', label='ERA5-Land')

        # --- Pearson r annotation (model pooled everywhere; obs only on panel 1) ---
        r_model = pearson_r(dat)

        if i == 0:
            txt = (rf"$\it{{r}}$ = {r_model:.2f}" + "\n" +
                   rf"$\it{{r}}$ (ERA5-Land) = {obs_r:.2f}")
        else:
            txt = rf"$\it{{r}}$ = {r_model:.2f}"

        ax.text(
            0.98, 0.98, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
        )

        ax.set_title(titles[i], fontsize=13)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel('Precipitation (mm)', fontsize=12)
        if col == 0:
            ax.set_ylabel('Temperature (°C)', fontsize=12)

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = Path(outdir) / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {outpath}')
    plt.savefig(outpath, dpi=200)
    plt.close()


## Bias Correction: Yiweh's method ##
# correct mean and correlation, preserve the variance
def multi_correction_eigen_per_ensemble(mod, obs):
    n_years, n_ensembles, n_vars = mod.shape
    mod_corrected = np.empty_like(mod)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0)
    obs_st = (obs - obs_mean) / obs_std
    Covariance_obs = np.cov(obs_st, rowvar=False)
    eigenvalues_obs, W = np.linalg.eigh(Covariance_obs)
    eigenvalues_obs = np.maximum(eigenvalues_obs, 1e-6)

    for i in range(n_ensembles):
        mod_mean = np.mean(mod[:, i, :], axis=0)
        mod_std = np.std(mod[:, i, :], axis=0)
        mod_st = (mod[:, i, :] - mod_mean) / mod_std
        Covariance_mod = np.cov(mod_st, rowvar=False)
        eigenvalues_mod, V = np.linalg.eigh(Covariance_mod)
        eigenvalues_mod = np.maximum(eigenvalues_mod, 1e-6)
        Gamma_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues_mod))
        Lambda_sqrt = np.diag(np.sqrt(eigenvalues_obs))

        Zm = np.einsum('ij,jk->ik', mod_st, V)                 # mod_st * V
        Zm = np.einsum('ik,kl->il', Zm, Gamma_inv_sqrt)        # * Gamma^-1/2
        Zm = np.einsum('il,lm->im', Zm, Lambda_sqrt)           # * Lambda^1/2
        Zm = np.einsum('im,mn->in', Zm, W.T)                   # * W^T
        mod_corrected[:, i, :] = Zm * mod_std + obs_mean
    return mod_corrected

mod_corrected_eigen = multi_correction_eigen_per_ensemble(mod_raw, obs_combined)


## Comparison with other multivariate bias-adjustment methods using SBCK tool ##

def get_bc_handler(bc_type):
    # --- REMOVED MBCn ---
    if bc_type == 'R2D2':
        return R2D2()
    elif bc_type == 'dOTC':
        return dOTC()
    elif bc_type == 'MRec':
        return MRec()
    else:
        return None


def apply_sbck_to_ensemble(obs, model_ensemble, bc_type, idx_train, idx_test):
    """
    Fit on idx_train, predict on idx_test.
    """
    X_train = model_ensemble[:, idx_train, :]
    X_test  = model_ensemble[:, idx_test, :]

    n_samples, n_ensembles, n_features = X_train.shape
    corrected = np.empty_like(X_test)

    if bc_type == 'EMBCCA-UNSEEN':
        corrected = multi_correction_eigen_per_ensemble(X_test, obs)
    else:
        for i in range(n_ensembles):
            Y0 = np.asarray(obs)
            X0 = np.asarray(X_train[:, i, :])
            X1 = np.asarray(X_test[:, i, :])

            SBCK = get_bc_handler(bc_type)
            SBCK.fit(Y0, X0, X1)
            corrected[:, i, :] = SBCK.predict(X1)

    return corrected


# --- REMOVED MBCn from methods ---
bc_method_names = ['EMBCCA-UNSEEN', 'dOTC', 'MRec', 'R2D2']
mod_corrected_sbck_list = []

for bc_method_name in bc_method_names:
    # apply bias-adjustment
    mod_corrected_sbck = apply_sbck_to_ensemble(obs_combined, mod_raw_full, bc_method_name, idx_train, idx_test)
    mod_corrected_sbck_list.append(mod_corrected_sbck)

# Create the 6-panel scatter plot once (raw, univariate, then SBCK methods)
plot_scatter_sixpanel(
    obs_combined=obs_combined,
    mod_raw=mod_raw,
    meancor_combined=meancor_combined,
    sbck_list=mod_corrected_sbck_list,
    sbck_names=bc_method_names,
    outdir=outdir,
    xlim=(0, 1300),
    ylim=(20, 34),
    fname='Scatter_plots/Scatter_sixpanel.png'
)


## Fidelity Testing for mean and correlation corrected data ##

def mardia_skewness_kurtosis(X):
    n_samples, n_features = X.shape
    mean_vector = np.mean(X, axis=0)
    covariance_matrix = np.cov(X, rowvar=False)
    cov_inv = np.linalg.pinv(covariance_matrix)

    skewness = 0
    for i in range(n_samples):
        for j in range(n_samples):
            diff_i = X[i] - mean_vector
            diff_j = X[j] - mean_vector
            skewness += (np.dot(np.dot(diff_i.T, cov_inv), diff_j))**3
    skewness /= n_samples**2

    kurtosis = 0
    for i in range(n_samples):
        diff = X[i] - mean_vector
        kurtosis += (np.dot(np.dot(diff.T, cov_inv), diff))**2
    kurtosis /= n_samples

    return skewness, kurtosis

## Bootstrapping ##

def calcDistrStatistic(obs, mod_corrected, n_samples=10000, n_months=1, seed=None):
    rng = random.Random(seed)
    nyrs, n_members, _ = mod_corrected.shape

    # Observed statistics
    obs_means = np.mean(obs, axis=0)
    obs_stds = np.std(obs, axis=0)
    obs_corr = np.corrcoef(obs[:, 0], obs[:, 1])[0, 1]
    obs_skewness, obs_kurtosis = mardia_skewness_kurtosis(obs)

    obs_stats_dict = {
        "Mean": obs_means,
        "Standard Deviation": obs_stds,
        "Correlation": obs_corr,
        "Skewness": obs_skewness,
        "Kurtosis": obs_kurtosis,
    }

    bootstrap_means = []
    bootstrap_stds = []
    bootstrap_corrs = []
    bootstrap_skewness = []
    bootstrap_kurtosis = []

    for _ in range(n_samples):
        model_sample = []
        for iyr in range(nyrs):
            sample_indices = rng.sample(range(n_members), n_months)
            model_sample.append(mod_corrected[iyr, sample_indices, :])
        model_sample = np.concatenate(model_sample, axis=0)

        means = np.mean(model_sample, axis=0)
        stds = np.std(model_sample, axis=0)
        corr = np.corrcoef(model_sample[:, 0], model_sample[:, 1])[0, 1]
        skewness, kurtosis_val = mardia_skewness_kurtosis(model_sample)

        bootstrap_means.append(means)
        bootstrap_stds.append(stds)
        bootstrap_corrs.append(corr)
        bootstrap_skewness.append(skewness)
        bootstrap_kurtosis.append(kurtosis_val)

    mod_stats_dict = {
        "Mean": np.array(bootstrap_means),
        "Standard Deviation": np.array(bootstrap_stds),
        "Correlation": np.array(bootstrap_corrs),
        "Skewness": np.array(bootstrap_skewness),
        "Kurtosis": np.array(bootstrap_kurtosis),
    }

    return obs_stats_dict, mod_stats_dict

def calcPercentile(obs, model):
    model = sorted(model)
    ind = min(range(len(model)), key=lambda i: abs(model[i] - obs))
    percentile = ind / float(len(model)) * 100
    return percentile

# cutting raw data to same sample size as bias-adjusted data
mod_raw_full = mod_raw.copy()
rng = np.random.default_rng(seed=seed)
n_total = mod_raw.shape[1]
n_select = int(n_total / 2)
idx = rng.choice(n_total, size=n_select, replace=False)
idx_comp = np.setdiff1d(np.arange(n_total), idx, assume_unique=True)
mod_raw = mod_raw[:, idx_comp, :]

# adding to our model list
mod_corrected_sbck_list.insert(0, mod_raw)
bc_method_names.insert(0, 'DePreSys model (raw)')

# --- dynamic colours to match number of datasets ---
default_colors = ["lightblue", "red", "purple", "green", "yellow", "navy", "orange", "grey"]
colors = default_colors[:len(bc_method_names)]

# create empty lists to append stats to
mod_stats_dict_list = []
mean_perc_list = []
std_perc_list = []
corr_perc_list = []
skew_perc_list = []
kurt_perc_list = []

n_samples = 10000

for model_data in mod_corrected_sbck_list:
    obs_stats_dict, mod_stats_dict = calcDistrStatistic(
        obs_combined, model_data, n_samples=n_samples, n_months=1, seed=seed
    )
    mod_stats_dict_list.append(mod_stats_dict)

    # Calculate percentiles
    mean_perc = [calcPercentile(obs_stats_dict["Mean"][i], mod_stats_dict["Mean"][:, i]) for i in range(2)]
    std_perc = [calcPercentile(obs_stats_dict["Standard Deviation"][i], mod_stats_dict["Standard Deviation"][:, i]) for i in range(2)]
    corr_perc = calcPercentile(obs_stats_dict["Correlation"], mod_stats_dict["Correlation"])
    skew_perc = calcPercentile(obs_stats_dict["Skewness"], mod_stats_dict["Skewness"])
    kurt_perc = calcPercentile(obs_stats_dict["Kurtosis"], mod_stats_dict["Kurtosis"])

    mean_perc_list.append(mean_perc)
    std_perc_list.append(std_perc)
    corr_perc_list.append(corr_perc)
    skew_perc_list.append(skew_perc)
    kurt_perc_list.append(kurt_perc)

def plot_single_variable_comparison(
    obs_stats, mod_stats_list,
    percentiles_list,
    statistic_key, variable_name, color, variable
):
    datasets = []
    for i, mod_stats in enumerate(mod_stats_list):
        datasets.append((bc_method_names[i], mod_stats_list[i], percentiles_list[i]))

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharey=False)

    for j, (label, stats_dict, percentiles_set) in enumerate(datasets):
        if j >= nrows * ncols:
            break

        row, col = divmod(j, ncols)
        ax = axes[row, col]

        obs_stat = obs_stats[statistic_key][variable]
        model_stat = stats_dict[statistic_key][:, variable]

        ax.hist(
            model_stat,
            bins=30,
            density=True,
            alpha=0.5,
            histtype="stepfilled",
            color=color[j],
            label=f"Model {statistic_key}",
        )
        ax.axvline(
            obs_stat,
            color="k",
            linestyle="--",
            label=(
                f"Observed {statistic_key}: {obs_stat:.2f}\n"
                f"(Percentile: {percentiles_set[variable]:.2f}%)"
            ),
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(f"{statistic_key} of {variable_name}", fontsize=12)

        if col == 0:
            ax.set_ylabel(f"{variable_name} Density", fontsize=12)

        ax.legend(loc="upper right", fontsize=11)

    # Hide unused axes
    for k in range(len(datasets), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    plt.tight_layout()

    key_for_fname = 'Standard_Deviation' if statistic_key == 'Standard Deviation' else statistic_key.replace(" ", "_")
    fname = Path(f'{outdir}Statistical_Comparison/{variable_name}_{key_for_fname}.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()

# mean comparison
plot_single_variable_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=mean_perc_list,
    statistic_key="Mean",
    variable_name="Precipitation",
    color=colors,
    variable=0
)

plot_single_variable_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=mean_perc_list,
    statistic_key="Mean",
    variable_name="Temperature",
    color=colors,
    variable=1
)

# std comparison
plot_single_variable_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=std_perc_list,
    statistic_key="Standard Deviation",
    variable_name="Precipitation",
    color=colors,
    variable=0
)

plot_single_variable_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=std_perc_list,
    statistic_key="Standard Deviation",
    variable_name="Temperature",
    color=colors,
    variable=1
)

def plot_corr_skew_kurtosis_comparison(
    obs_stats, mod_stats_list,
    percentiles_list,
    statistic_key, color
):
    datasets = []
    for i, mod_stats in enumerate(mod_stats_list):
        datasets.append((bc_method_names[i], mod_stats_list[i], percentiles_list[i]))

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharey=False)

    for j, (label, stats_dict, percentiles_set) in enumerate(datasets):
        if j >= nrows * ncols:
            break

        row, col = divmod(j, ncols)
        ax = axes[row, col]

        obs_stat = obs_stats[statistic_key]
        model_stat = stats_dict[statistic_key]

        ax.hist(
            model_stat,
            bins=30,
            density=True,
            alpha=0.5,
            histtype="stepfilled",
            color=color[j],
            label=f"Model {statistic_key}",
        )
        ax.axvline(
            obs_stat,
            color="k",
            linestyle="--",
            label=(
                f"Observed {statistic_key}: {obs_stat:.2f}\n"
                f"(Percentile: {percentiles_set:.2f}%)"
            ),
        )
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(f"{statistic_key}", fontsize=13)
        if j == 0:
            ax.set_ylabel("Density", fontsize=13)
        ax.legend(loc="upper right", fontsize=11)

    # Hide unused axes
    for k in range(len(datasets), nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis('off')

    plt.tight_layout()

    fname = Path(f'{outdir}Statistical_Comparison/{statistic_key}.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()

# Correlation / Skewness / Kurtosis comparisons (dynamic colours)
plot_corr_skew_kurtosis_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=corr_perc_list,
    statistic_key="Correlation",
    color=colors,
)

plot_corr_skew_kurtosis_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=skew_perc_list,
    statistic_key="Skewness",
    color=colors,
)

plot_corr_skew_kurtosis_comparison(
    obs_stats=obs_stats_dict,
    mod_stats_list=mod_stats_dict_list,
    percentiles_list=kurt_perc_list,
    statistic_key="Kurtosis",
    color=colors,
)

## SVM ##

def compare_with_svm(obs_data, mod_data):
    X = np.vstack((obs_data, mod_data))
    y = np.hstack((np.zeros(obs_data.shape[0]), np.ones(mod_data.shape[0])))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm_model = svm.SVC(kernel='rbf', probability=True, random_state=seed)
    svm_model.fit(X_scaled, y)
    y_scores = svm_model.decision_function(X_scaled)
    obs_decision_mean = np.mean(y_scores[:obs_data.shape[0]])
    mod_decision_mean = np.mean(y_scores[obs_data.shape[0]:])
    y_pred = svm_model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    fpr, tpr, _ = roc_curve(y, svm_model.predict_proba(X_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)
    return accuracy, obs_decision_mean, mod_decision_mean, fpr, tpr, roc_auc



def SVM_plot_roc_sixpanel(
    obs_combined,
    mod_list,
    names,
    outdir,
    fname="SVM_Comparison/SVM_ROC_sixpanel.png",
    seed=42,
    show_member_curves=True,
    member_alpha=0.06,
    member_lw=1.0,
    mean_lw=2.5,
    band_alpha=0.20,
):
    """
    Plot ROC curves only in a 2x3 multi-panel figure.
    - Up to 6 datasets; if fewer, unused panels are hidden.
    - Each panel shows: member ROC curves (optional), mean ROC, ±1 std band, random baseline.
    - Uses compare_with_svm() defined in your script.

    Parameters
    ----------
    obs_combined : array, shape (T, 2)
    mod_list : list of arrays, each shape (T, E, 2)
    names : list of str, same length as mod_list
    outdir : str
    fname : str (relative to outdir)
    seed : int (passed through to svm random_state via your global `seed` usage)
    """

    # --- styling knobs ---
    title_fs = 12  # subplot title
    label_fs = 13  # axis labels
    tick_fs = 12  # tick labels
    legend_fs = 11  # legend text
    box_fs = 12  # AUC box
    suptitle_fs = 18  # overall title

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 13), sharex=True, sharey=True)
    axes = axes.ravel()

    # Common ROC grid
    mean_fpr = np.linspace(0, 1, 200)

    for k in range(nrows * ncols):
        ax = axes[k]

        if k >= len(mod_list):
            ax.axis("off")
            continue

        mod_cor = mod_list[k]
        name = names[k]

        obs_data = obs_combined
        num_ensembles = mod_cor.shape[1]

        # Run SVM per ensemble member
        results = [compare_with_svm(obs_data, mod_cor[:, j, :]) for j in range(num_ensembles)]
        accuracies, obs_means, mod_means, fprs, tprs, aucs = zip(*results)

        interp_tprs = []
        for fpr, tpr in zip(fprs, tprs):
            interp = np.interp(mean_fpr, fpr, tpr)
            interp[0] = 0.0
            interp_tprs.append(interp)

            if show_member_curves:
                ax.plot(fpr, tpr, color="lightblue", alpha=member_alpha, linewidth=member_lw)

        interp_tprs = np.asarray(interp_tprs)
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        std_tpr = np.std(interp_tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Mean + band + random
        ax.plot(mean_fpr, mean_tpr, color="blue", linewidth=mean_lw, label=f"Mean ROC (AUC={mean_auc:.2f})")
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=band_alpha, label=r"$\pm$ 1 std. dev.")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.6, label="Random")

        # Panel cosmetics
        ax.set_title(name, fontsize=title_fs, pad=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.tick_params(axis="both", labelsize=tick_fs)

        # Labels only on left + bottom like your other multi-panels
        row, col = divmod(k, ncols)
        if row == nrows - 1:
            ax.set_xlabel("False Positive Rate", fontsize=label_fs)
        if col == 0:
            ax.set_ylabel("True Positive Rate", fontsize=label_fs)

        # Small summary box
        ax.text(
            0.02, 0.98,
            f"AUC={mean_auc:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=box_fs,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none")
        )

        # Legend only on first panel to reduce clutter
        if k == 0:
            leg = ax.legend(loc="lower right", fontsize=legend_fs, frameon=True)
            leg.get_frame().set_alpha(0.9)

    fig.suptitle("SVM ROC Curves", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = Path(outdir) / fname
    outpath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {outpath}")
    plt.savefig(outpath, dpi=200)
    plt.close()

SVM_plot_roc_sixpanel(
    obs_combined=obs_combined,
    mod_list=mod_corrected_sbck_list,   # SBCK only
    names=bc_method_names,              # SBCK only
    outdir=outdir,
    fname="SVM_Comparison/SVM_ROC_sixpanel.png",
    seed=seed,
    show_member_curves=True
)


def plot_comparison(obs, mod_corrected_sbck_list, bc_method_names):
    obs_extreme_temp = obs[:, 1].max()
    obs_extreme_precip = obs[:, 0].min()

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharey=False)

    plotted = 0
    for i, mod in enumerate(mod_corrected_sbck_list):
        if i >= nrows * ncols:
            break

        mod_flat = mod.reshape(-1, 2)

        mod_exceeding = mod_flat[
            (mod_flat[:, 1] >= obs_extreme_temp) | (mod_flat[:, 0] <= obs_extreme_precip)
        ]

        row, col = divmod(i, ncols)
        ax = axes[row, col]

        sns.kdeplot(x=obs[:, 1], y=obs[:, 0], cmap="Blues", fill=True, alpha=0.6, ax=ax, label="Observations")
        sns.kdeplot(x=mod_flat[:, 1], y=mod_flat[:, 0], cmap="Oranges", fill=True, alpha=0.4, ax=ax)
        ax.axvline(obs_extreme_temp, color="blue", linestyle="--", linewidth=1.5)
        ax.axhline(obs_extreme_precip, color="green", linestyle="--", linewidth=1.5)
        ax.scatter(mod_exceeding[:, 1], mod_exceeding[:, 0], color="red", alpha=0.6)
        ax.set_title(f"Obs vs. {bc_method_names[i]}")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Precipitation")

        plotted += 1

    # Hide unused axes
    for k in range(plotted, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    custom_lines = [
        Line2D([0], [0], color="blue", linestyle="--", lw=1.5, label="Observed Temp Extreme"),
        Line2D([0], [0], color="green", linestyle="--", lw=1.5, label="Observed Precip Extreme"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=13, label="Record-breaking Events"),
    ]
    fig.legend(handles=custom_lines, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95))

    fname = Path(f'{outdir}Exceedance_Comparison/exceedance_comparison.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()

plot_comparison(obs_combined, mod_corrected_sbck_list, bc_method_names)


def calculate_and_compare_probabilities(obs, mod_corrected_sbck_list, bc_method_names, colors):
    obs_extreme_precip = obs[:, 0].min()
    obs_extreme_temp = obs[:, 1].max()

    # --- styling knobs (tweak here) ---
    title_fs = 18
    label_fs = 16
    tick_fs = 14
    legend_fs = 12
    annot_fs = 12

    fig, ax = plt.subplots(figsize=(11, 6))  # wider + less tall than 10x10
    labels = ['Dry', 'Hot']
    x = np.arange(len(labels))

    def centered_offsets(n, width):
        return (np.arange(n) - (n - 1) / 2) * width

    width = 0.12 if len(mod_corrected_sbck_list) <= 6 else 0.09
    offsets = centered_offsets(len(mod_corrected_sbck_list), width)

    # collect probabilities for setting y-limits
    all_probs = []

    for i, mod in enumerate(mod_corrected_sbck_list):
        mod_flat = mod.reshape(-1, 2)

        dry_count = np.sum(mod_flat[:, 0] <= obs_extreme_precip)
        hot_count = np.sum(mod_flat[:, 1] >= obs_extreme_temp)
        total_count = len(mod_flat)

        P_dry = dry_count / total_count
        P_hot = hot_count / total_count
        probs = [P_dry, P_hot]
        all_probs.extend(probs)

        bars = ax.bar(
            x + offsets[i],
            probs,
            width,
            label=bc_method_names[i],
            color=colors[i],
            alpha=0.75,
            edgecolor='black',
            linewidth=0.6,
            zorder=3
        )

        # annotations (percent)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=annot_fs
            )

    # axes labels + title
    ax.set_ylabel('Probability (%)', fontsize=label_fs)
    ax.set_title('Probability Comparison of Dry and Hot Events', fontsize=title_fs, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fs)

    # y-axis as percent
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='y', labelsize=tick_fs)

    # y-limits: tighten to data with a bit of headroom
    ymax = max(all_probs) if all_probs else 1.0
    ax.set_ylim(0, min(1.0, ymax * 1.15 + 0.01))

    # grid
    ax.grid(True, linestyle='--', alpha=0.35, axis='y', zorder=0)

    # legend outside right
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=legend_fs,
        title='Dataset',
        title_fontsize=legend_fs
    )

    plt.tight_layout()

    fname = Path(f'{outdir}Exceedance_Comparison/exceedance_comparison_bar.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname, dpi=200)
    plt.close()

calculate_and_compare_probabilities(obs_combined, mod_corrected_sbck_list, bc_method_names, colors=colors)


def calculate_and_compare_joint_probabilities(obs, mod_corrected_sbck_list, bc_method_names, colors):
    obs_extreme_precip = obs[:, 0].min()
    obs_extreme_temp = obs[:, 1].max()

    # --- styling knobs ---
    title_fs = 18
    label_fs = 16
    tick_fs = 14
    legend_fs = 12
    annot_fs = 12

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ['High Temp & Low Precip']
    x = np.arange(len(labels))

    def centered_offsets(n, width):
        return (np.arange(n) - (n - 1) / 2) * width

    width = 0.14 if len(mod_corrected_sbck_list) <= 6 else 0.10
    offsets = centered_offsets(len(mod_corrected_sbck_list), width)

    all_probs = []

    for i, mod in enumerate(mod_corrected_sbck_list):
        mod_flat = mod.reshape(-1, 2)

        joint_count = np.sum(
            (mod_flat[:, 0] <= obs_extreme_precip) &
            (mod_flat[:, 1] >= obs_extreme_temp)
        )
        total_count = len(mod_flat)
        joint_prob = joint_count / total_count
        all_probs.append(joint_prob)

        bars = ax.bar(
            x + offsets[i],
            [joint_prob],
            width,
            label=bc_method_names[i],
            color=colors[i],
            alpha=0.75,
            edgecolor='black',
            linewidth=0.6,
            zorder=3
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=annot_fs,
                fontweight='bold'
            )

    ax.set_ylabel('Probability (%)', fontsize=label_fs)
    ax.set_title('Joint Probability of High Temp & Low Precip', fontsize=title_fs, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=label_fs)

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis='y', labelsize=tick_fs)

    ymax = max(all_probs) if all_probs else 1.0
    ax.set_ylim(0, min(1.0, ymax * 1.20 + 0.005))

    ax.grid(True, linestyle='--', alpha=0.35, axis='y', zorder=0)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=legend_fs,
        title='Dataset',
        title_fontsize=legend_fs
    )

    plt.tight_layout()

    fname = Path(f'{outdir}Exceedance_Comparison/joint_exceedance_comparison_bar.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname, dpi=200)
    plt.close()


calculate_and_compare_joint_probabilities(obs_combined, mod_corrected_sbck_list, bc_method_names, colors=colors)

def calculate_below_minimum_compound_probability(data, temp_threshold, precip_threshold, precip_decrements):
    probs = []
    for precip_dec in precip_decrements:
        prob = np.mean((data[:, :, 1] >= temp_threshold) &
                       (data[:, :, 0] <= (precip_threshold - precip_dec))) * 100
        probs.append(prob)
    return probs

def plot_below_minimum_compound_probability(ax, mod, obs, title):
    obs_min_precip = np.min(obs[:, 0])
    obs_high_temp = np.percentile(obs[:, 1], 100)
    max_precip_decrement = obs_min_precip * 0.5
    precip_decrements = np.linspace(0, max_precip_decrement, 150)
    probabilities = calculate_below_minimum_compound_probability(mod, obs_high_temp, obs_min_precip, precip_decrements)

    ax.plot(precip_decrements, probabilities, color='blue', linewidth=2, label="Probability")
    ax.set_xlabel('Precipitation Decrement')
    ax.set_ylabel('Probability of High Temp & Low Precip Event (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 10)
    ax.set_xlim(0, max_precip_decrement)
    ax.set_yticks([0.2, 1, 2.5, 10])
    ax.set_yticklabels(['0.2%', '1%', '2.5%', '10%'])
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_title(title)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, label="1% Threshold")
    ax.legend()
    return probabilities, precip_decrements

nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharey=False)

plotted = 0
for i, mod in enumerate(mod_corrected_sbck_list):
    if i >= nrows * ncols:
        break
    row, col = divmod(i, ncols)
    ax = axes[row, col]
    prob, precip_decrements = plot_below_minimum_compound_probability(ax, mod, obs_combined, bc_method_names[i])
    plotted += 1

for k in range(plotted, nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].axis("off")

plt.tight_layout()
fname = Path(f'{outdir}Exceedance_Comparison/joint_exceedance_by_precipitation_decrement.png')
fname.parent.mkdir(parents=True, exist_ok=True)
print(f'Saving {fname}')
plt.savefig(fname)
plt.close()

def prob_compound_probability_bytemp(data, precip_threshold, temp_threshold, temp_increments):
    probs = []
    for temp_inc in temp_increments:
        prob = np.mean((data[:, :, 1] >= (temp_threshold + temp_inc)) &
                       (data[:, :, 0] <= precip_threshold)) * 100
        probs.append(prob)
    return probs

def plot_compound_probability_bytemp(ax, mod, obs, title):
    obs_min_precip = np.min(obs[:, 0])
    obs_temp_thr = np.percentile(obs[:, 1], 100)
    max_temp_increment = obs_temp_thr * 0.08
    temp_increments = np.linspace(0, max_temp_increment, 100)
    probabilities = prob_compound_probability_bytemp(mod, obs_min_precip, obs_temp_thr, temp_increments)

    ax.plot(temp_increments, probabilities, color='blue', linewidth=2, label="Probability")
    ax.set_xlabel('Temperature Increment (°C)')
    ax.set_ylabel('Probability of High Temp & Low Precip Event (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.25, 10)
    ax.set_xlim(0, max_temp_increment)
    ax.set_yticks([0.25, 1, 2.5, 10])
    ax.set_yticklabels(['0.25%', '1%', '2.5%', '10%'])
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.set_title(title)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=0.5, label="1% Threshold")
    ax.legend()
    return probabilities, temp_increments

nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 12), sharey=False)

plotted = 0
for i, mod in enumerate(mod_corrected_sbck_list):
    if i >= nrows * ncols:
        break
    row, col = divmod(i, ncols)
    ax = axes[row, col]
    prob, temp_increments = plot_compound_probability_bytemp(ax, mod, obs_combined, bc_method_names[i])
    plotted += 1

for k in range(plotted, nrows * ncols):
    r, c = divmod(k, ncols)
    axes[r, c].axis("off")

plt.tight_layout()
fname = Path(f'{outdir}Exceedance_Comparison/joint_exceedance_by_temperature_increment.png')
fname.parent.mkdir(parents=True, exist_ok=True)
print(f'Saving {fname}')
plt.savefig(fname)
plt.close()

# ## FIDELITY TESTING ##
fid_method_names = ['Univariate mean shift'] + list(bc_method_names)
fid_model_list = [meancor_combined] + list(mod_corrected_sbck_list)

for i, mod in enumerate(fid_model_list):
    # Temperature
    stats_measures_temp = ftc.timeseries_fid_test(obs_combined[:, 1], mod[:, :, 1], seed=seed)
    ftc.plot_fidelity_testing(obs_combined[:, 1], mod[:, :, 1], stats_measures_temp, 0.1, "", "1.png")
    fname = Path(f'{outdir}Fidelity_Testing/{fid_method_names[i]}_temperature.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()

    # Precipitation
    stats_measures_pr = ftc.timeseries_fid_test(obs_combined[:, 0], mod[:, :, 0], seed=seed)
    ftc.plot_fidelity_testing(obs_combined[:, 0], mod[:, :, 0], stats_measures_pr, 0.1, "", "1.png")
    fname = Path(f'{outdir}Fidelity_Testing/{fid_method_names[i]}_precipitation.png')
    fname.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving {fname}')
    plt.savefig(fname)
    plt.close()

