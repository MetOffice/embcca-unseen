(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE.txt in the root of the repository for full licensing details.
# Manuscript analyses

The two analyses from:

> *A new fast multivariate bias correction technique: a case study for compound
> events in Hunan Province, China, using the UNSEEN approach*

| File | Analysis |
|---|---|
| `Multi-DePreSys4-Paper-area_avg_final_multiscatter.py` | Hunan Province, area-mean. |
| `Multi-DePreSys4-Paper-area_full_final_China.py` | China-wide spatial analysis. |
| `fidelity_test_cube.py` | Helper module for UNSEEN-style fidelity testing, imported by both scripts. |

Both scripts apply the bias adjustment by calling `embcca.bias_adjust_unseen`
from the `embcca-unseen` package. That is the variant which keeps the model's
own standard deviation, as the UNSEEN approach requires.

## Requirements

Use the conda environment at the repository root. It provides everything these
scripts need: matplotlib, iris, cartopy, netCDF4, scikit-learn, and SBCK for
the dOTC / MRec / R2D2 comparison.

From the repository root:

```bash
conda env create -f environment.yml
conda activate embcca-unseen
pip install -e .
```

Conda is used rather than a plain pip environment because SBCK has no Linux or
Windows wheel and must be compiled, which needs a C++ compiler and the Eigen
headers. `environment.yml` supplies both, and SBCK finds them automatically
inside an active conda environment. Activate the environment before installing
anything with pip, or that lookup will fail.

## Input data

Not committed; available on reasonable request. Both scripts read NetCDF from
`DATA_DIR`.

**Model (DePreSys4)**, one file each for temperature and precipitation:
dimensions `(year, realisation, leadtime, latitude, longitude)`, with
realisation × leadtime collapsed to the ensemble axis at load time.
Variables: `mean_jja_temperature`, `total_jja_precipitation`.

**Observations (ERA5-Land)**: dimensions `(year, latitude, longitude)`.
Variables: `t2m`, `tp`.

The China script needs observations **already regridded onto the model grid**,
which is what the `_regridded` suffix in its default filenames refers to, because
it stacks obs and model per grid cell. The Hunan script averages over latitude
and longitude first, so its inputs do not need to share a grid.

## How to run

This code is designed to be run by **editing constants at the top of each script**.

Each script contains:

```python
# =============================================================================
# USER SETTINGS (edit these to run the workflow)
# =============================================================================
```

You must edit:

* `DATA_DIR`
* `OUTDIR`

\---

### 1\. Hunan case study (main results)

```bash
cd paper
python Multi-DePreSys4-Paper-area_avg_final_multiscatter.py
```

Produces:

* Line plots (not shown in manuscript)
* Scatter plots comparing joint distributions across methods
* Statistical fidelity plots
* SVM ROC curves
* Extreme event probabilities

Outputs saved in:

```
OUTDIR/
```

subfolders:

* `Line_plots`
* `Scatter_plots`
* `Statistical_Comparison`
* `SVM_Comparison`
* `Exceedance_Comparison`
* `Fidelity_Testing/`

\---

### 2\. China spatial analysis

```bash
cd paper
python Multi-DePreSys4-Paper-area_full_final_China.py
```

Produces:

* Spatial correlation maps
* Correlation anomaly maps
* Calculation of time taken for each multivariate bias adjustment method to be applied across China

Outputs saved in:

```
OUTDIR/China/
```

\---

## Figure mapping (script outputs → manuscript figures)

This table helps reproduce key figures from the paper.

### Main figures

|Manuscript Figure|Description|Script output|
|-|-|-|
|Figure 3|Univariate fidelity (temperature mean shift)|`Fidelity_Testing/Original model data_temperature.png` and `Fidelity_Testing/Univariate mean shift_temperature.png`|
|Figure 4|Six-panel temperature–precip scatter|`Scatter_plots/Scatter_sixpanel.png`|
|Figure 5a|China correlation maps|`China/maps/correlation'.png`|
|Figure 5b|Correlation anomaly maps|`China/maps/correlation_diff'.png`|
|Figure 6|Correlation fidelity distributions|`Statistical_Comparison/Correlation.png`|
|Figure 7|SVM ROC curves|`SVM_Comparison/SVM_ROC_sixpanel.png`|
|Figure 8 (left)|Dry / hot probabilities|`Exceedance_Comparison/exceedance_comparison_bar.png`|
|Figure 8 (right)|Joint probability|`Exceedance_Comparison/joint_exceedance_comparison_bar.png`|
|Figure 9|Threshold sensitivity plots||
|→ precip decrement||`Exceedance_Comparison/joint_exceedance_by_precipitation_decrement.png`|
|→ temperature increment||`Exceedance_Comparison/joint_exceedance_by_temperature_increment.png`|

### Appendix B figures (SFC testing)

|Appendix Figure|Description|Script output|
|-|-|-|
|Figures 10–11|Mean distributions|`Statistical_Comparison/*_Mean.png` where * = temperature or precipitation
|Figures 12–13|Standard deviation|`Statistical_Comparison/*_Standard_Deviation.png` where * = temperature or precipitation
|Figure 14|Skewness|`Statistical_Comparison/Skewness.png`|
|Figure 15|Kurtosis|`Statistical_Comparison/Kurtosis.png`|

\---

## Data requirements

This repository does **not** include data. Data can be provided on request in .nc format.

You need:

### Hunan (area-mean)

* DePreSys4 JJA temperature + precipitation
* ERA5-Land JJA temperature + precipitation

### China (gridded)

* Regridded ERA5-Land
* DePreSys4 gridded output

### Expected variable names

|Variable|Name|
|-|-|
|Model temperature|`mean_jja_temperature`|
|Model precipitation|`total_jja_precipitation`|
|Obs temperature|`t2m`|
|Obs precipitation|`tp`|

\---

## Reproducibility

* Seeds are used for bootstrapping during SFC testing, during SVM resampling, and when splitting the data into training and testing, so that results are reproducible.

Analysis period:

```
1992–2021 (30 years)
```

\---

## Plotting / HPC usage

The scripts use:

```python
MPL_BACKEND = "Agg"
```

Switch to `"Qt5Agg"` only for interactive use.

\---

## Code and data availability

* Data: DePreSys data available from CEDA (https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast for data up to 2018 and https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast for data from 2019-2024). Shapefiles for country and administrative boundaries available from Natural Earth (https://www.naturalearthdata.com/downloads/)
* Code: provided in this repository

\---
## Outputs

Figure directories are created as needed.

Hunan, under `OUTDIR`: `Line_plots/`, `Scatter_plots/`,
`Statistical_Comparison/`, `SVM_Comparison/`, `Exceedance_Comparison/`,
`Fidelity_Testing/`.

China, under `OUTDIR`: `Maps/` (mean maps, correlation maps, and
correlation-difference maps). Method timings are printed to stdout, not saved.
