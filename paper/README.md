(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE in the root of the repository for full licensing details.
# Manuscript analyses

The two analyses from:

> *A new fast multivariate bias correction technique: a case study for compound
> events in Hunan Province, China, using the UNSEEN approach*

| File | Purpose                                                                                                                                                   |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Multi-DePreSys4-Paper-area_avg_final_multiscatter.py` | Hunan Province area-mean analysis used for the main manuscript results.                                                                                   |
| `Multi-DePreSys4-Paper-area_full_final_China.py` | China-wide spatial analysis used for correlation-map and computational-cost comparisons.                                                                  |
| `prepare_depresys4_data_for_EMBCCA-UNSEEN.py` | Extracts seasonal DePreSys4 temperature or precipitation from CMIP6 DCPP archives and writes NetCDF files which are used by the first two scripts listed. |
| `prepare_era5-land_data_for_EMBCCA-UNSEEN.py` | Extracts seasonal ERA5-Land temperature or precipitation and writes NetCDF files which are used by the first two scripts listed.                          |
| `regrid_obs_to_depresys_grid.py` | Regrids ERA5-Land observations onto the DePreSys4 grid for gridded China analyses.                                                                        |
| `fidelity_test_cube.py` | Statistical fidelity test module (an established part of UNSEEN analysis), used by the first two scripts listed.                                          |

The preprocessing scripts listed above can be used to regenerate the manuscript input datasets from the publicly available source data.

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
**Model (DePreSys4)**, one file each for temperature and precipitation:
dimensions `(year, realisation, leadtime, latitude, longitude)`, with
realisation × leadtime collapsed to the ensemble axis at load time.
Variables: `mean_jja_temperature`, `total_jja_precipitation`.

**Observations (ERA5-Land)**: dimensions `(year, latitude, longitude)`.
Variables: `t2m`, `tp`.

The gridded China analysis requires ERA5-Land observations to be regridded onto the DePreSys4 grid before bias adjustment. The script `regrid_obs_to_depresys_grid.py` performs this step.

The area-mean Hunan analysis averages over latitude and longitude before bias adjustment and therefore does not require the observations and model data to share the same grid.

## Workflow overview

The manuscript analyses depend on processed NetCDF inputs.

To fully reproduce the workflow:

1. Download DePreSys4 data from the CEDA archive.
2. Download ERA5-Land data from the Copernicus Climate Data Store.
3. Run:
   * `prepare_depresys4_data_for_EMBCCA-UNSEEN.py`
   * `prepare_era5-land_data_for_EMBCCA-UNSEEN.py`
   * `regrid_obs_to_depresys_grid.py`
4. Run the manuscript analysis scripts.

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
* Calculation of computational cost for each multivariate bias adjustment method to be applied across China

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

## Reproducibility

All stochastic elements (bootstrap resampling, SVM testing, and ensemble
splitting) use fixed random seeds so results are reproducible.

\---

## Plotting / HPC usage

The scripts use:

```python
MPL_BACKEND = "Agg"
```

Switch to `"Qt5Agg"` only for interactive use.

\---

## Code and data availability

### Code

The EMBCCA-UNSEEN package, manuscript analysis scripts, preprocessing workflows,
and example datasets are provided in this repository.

### Source data

The underlying model data are from the Met Office DePreSys4 decadal prediction
system and are available through the CEDA CMIP6 archive:

* dcppA-hindcast (available up to 2018):
  https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast

* dcppB-forecast (available from 2019 onwards):
  https://data.ceda.ac.uk/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast

ERA5-Land observational data are available from the Copernicus Climate Data Store:

* https://cds.climate.copernicus.eu/

Country and administrative boundary shapefiles are available from Natural Earth:

* https://www.naturalearthdata.com/downloads/

The preprocessing scripts listed in the repository overview can be used to
recreate the manuscript input datasets from these source data.
\---
## Outputs

Figure directories are created automatically.

Hunan outputs:
* Line_plots/
* Scatter_plots/
* Statistical_Comparison/
* SVM_Comparison/
* Exceedance_Comparison/
* Fidelity_Testing/

China outputs:
* Maps/

Method timings for the China workflow are printed to stdout.
