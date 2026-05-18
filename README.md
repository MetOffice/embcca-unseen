(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE.txt in the root of the repository for full licensing details.

# EMBCCA‑UNSEEN: multivariate bias correction for UNSEEN compound extremes

This repository contains the analysis code used in the manuscript:

> \*A new fast multivariate bias correction technique: a case study for compound events in Hunan Province, China, using the UNSEEN approach\*

The code applies the **EMBCCA‑UNSEEN** bias correction method to DePreSys4 initialised hindcasts, using ERA5‑Land as the observational reference, and evaluates fidelity using:

* Multivariate statistical feature consistency (SFC) testing
* Support Vector Machine (SVM)-based separability testing

\---

## Repository structure

* `Multi-DePreSys4-Paper-area\_avg\_final\_multiscatter.py`  
→ Hunan Province (area-mean) analysis  
→ Reproduces main manuscript figures
* `Multi-DePreSys4-Paper-area\_full\_final\_China.py`  
→ China-wide spatial analysis  
→ Produces correlation maps and timing comparisons
* `fidelity\_test\_cube.py`  
→ Helper module for UNSEEN-style fidelity testing

\---

## Installation (conda)

```bash
conda env create -f environment.yml
conda activate embcca-unseen
```

Tested with:

* Python 3.12.10
* SBCK 1.4.2

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

* `DATA\_DIR`
* `OUTDIR`

\---

### 1\. Hunan case study (main results)

```bash
python Multi-DePreSys4-Paper-area\_avg\_final\_multiscatter.py
```

Produces:

* Line plots (not shown in manuscript)
* Scatter plots comparing joint distributions across methods
* Statistical fidelity plots
* SVM ROC curves
* Extreme-event probabilities

Outputs saved in:

```
OUTDIR/
```

subfolders:

* `Line\_plots/`
* `Scatter\_plots/`
* `Statistical\_Comparison/`
* `SVM\_Comparison/`
* `Exceedance\_Comparison/`
* `Fidelity\_Testing/`

\---

### 2\. China spatial analysis

```bash
python Multi-DePreSys4-Paper-area\_full\_final\_China.py
```

Produces:

* Spatial correlation maps
* Correlation anomaly maps
* Timing comparisons

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
|Figure 3|Univariate fidelity (temperature mean shift)|`Fidelity\_Testing/\*temperature.png`|
|Figure 4|Six-panel temperature–precip scatter|`Scatter\_plots/Scatter\_sixpanel.png`|
|Figure 5a|China correlation maps|China script (`plot\_corr\_map`)|
|Figure 5b|Correlation anomaly maps|China script (`plot\_corr\_diff\_map`)|
|Figure 6|Correlation fidelity distributions|`Statistical\_Comparison/Correlation.png`|
|Figure 7|SVM ROC curves|`SVM\_Comparison/SVM\_ROC\_sixpanel.png`|
|Figure 8 (left)|Dry / hot probabilities|`Exceedance\_Comparison/exceedance\_comparison\_bar.png`|
|Figure 8 (right)|Joint probability|`Exceedance\_Comparison/joint\_exceedance\_comparison\_bar.png`|
|Figure 9|Threshold sensitivity plots||
|→ precip decrement||`Exceedance\_Comparison/joint\_exceedance\_by\_precipitation\_decrement.png`|
|→ temperature increment||`Exceedance\_Comparison/joint\_exceedance\_by\_temperature\_increment.png`|

### Appendix B figures (SFC testing)

|Appendix Figure|Description|Script output|
|-|-|-|
|Figures 10–11|Mean distributions|`Statistical\_Comparison/\*Mean\*.png`|
|Figures 12–13|Standard deviation|`Statistical\_Comparison/\*Standard\_Deviation\*.png`|
|Figure 14|Skewness|`Statistical\_Comparison/Skewness.png`|
|Figure 15|Kurtosis|`Statistical\_Comparison/Kurtosis.png`|

\---

## Data requirements

This repository does **not** include data.

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
|Model temperature|`mean\_jja\_temperature`|
|Model precipitation|`total\_jja\_precipitation`|
|Obs temperature|`t2m`|
|Obs precipitation|`tp`|

\---

## Reproducibility

* Seed controlled via `SEED`
* Bootstrapping reproducible
* SVM reproducible
* Ensemble splitting reproducible

Analysis period:

```
1992–2021 (30 years)
```

\---

## Plotting / HPC usage

The scripts use:

```python
MPL\_BACKEND = "Agg"
```

This ensures compatibility with:

* HPC systems
* batch jobs
* headless environments

Switch to `"Qt5Agg"` only for interactive use.

\---

## Code and data availability

* Data: available on reasonable request
* Code: provided in this repository

\---

## Citation

Please cite:

* The associated manuscript
* This repository (`CITATION.cff`)

\---

## Notes

* SBCK methods included: **dOTC, MRec, R2D2**
* EMBCCA‑UNSEEN preserves variance (critical for UNSEEN)

