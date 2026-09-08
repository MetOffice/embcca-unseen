(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE.txt in the root of the repository for full licensing details.

# EMBCCA‑UNSEEN: multivariate bias correction for UNSEEN compound extremes

This repository contains two things:

* **`embcca`**, an installable Python package providing the EMBCCA‑UNSEEN multivariate bias correction method. It depends only on NumPy and can be applied to your own data.
* The analyses that produced the results in the manuscript:

> *A new fast multivariate bias correction technique: a case study for compound events in Hunan Province, China, using the UNSEEN approach*

Those analyses apply the **EMBCCA‑UNSEEN** bias correction method to DePreSys4 initialised hindcasts, using ERA5‑Land as the observational reference, and evaluate fidelity using:

* Multivariate statistical feature consistency (SFC) testing
* Support Vector Machine (SVM)-based separability testing

\---

## Repository structure

* `src/embcca/`  
→ The EMBCCA bias-adjustment method, as an installable Python package  
→ Depends only on NumPy
* `examples/`  
→ Worked example applying the package to the sample data in `example-data/`  
→ Needs only the package itself
* `example-data/`  
→ Small 2x2 grid cut from the China domain, stored ready to use
* `paper/`  
→ The manuscript analyses, with their own README covering the extra dependencies they need
* `paper/Multi-DePreSys4-Paper-area_avg_final_multiscatter.py`  
→ Hunan Province (area-mean) analysis  
→ Reproduces main manuscript figures
* `paper/Multi-DePreSys4-Paper-area_full_final_China.py`  
→ China-wide spatial analysis  
→ Produces correlation maps and comparison of time taken for the different multivariate methods when applied China-wide
* `paper/fidelity_test_cube.py`  
→ Helper module for UNSEEN-style fidelity testing
* `tests/`  
→ Test suite for the package
* `pyproject.toml`  
→ Package definition
* `environment.yml`  
→ Conda environment for the `paper/` analyses

\---

## Installation

### The package

The bias-adjustment method needs only NumPy and installs with pip on any platform:

```bash
pip install -e .
```

To see it applied to the sample data in `example-data/`:

```bash
python examples/example.py
```

That script corrects a small 2x2 grid in both the gridded and the area-mean form, and reports the mean, correlation and variance before and after. See `example-data/README.md` for the array layouts it expects.

### The manuscript analyses

The analyses in `paper/` need considerably more. Due to the SBCK dependency, use the conda environment for those:

```bash
conda env create -f environment.yml
conda activate embcca-unseen
pip install -e .
```

See `paper/README.md` for what the analyses require and how to run them.

Tested with:

* Python 3.12.10
* SBCK 1.4.2 (https://github.com/yrobink/SBCK-python)

\---

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

* Data: available on reasonable request
* Code: provided in this repository

\---

## Citation

Please cite:

* The associated manuscript

\---

## Notes

* SBCK methods included: **dOTC, MRec, R2D2**
* Unlike these methods, EMBCCA‑UNSEEN preserves variance (critical for UNSEEN applications)

