(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE in the root of the repository for full licensing details.
# EMBCCA-UNSEEN

A fast multivariate bias-adjustment method for UNSEEN climate-risk analyses.


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
→ Produces correlation maps and compares computational cost across multivariate bias-adjustment methods when applied China-wide
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

The package provides two related methods:

* `bias_adjust` (EMBCCA) adjusts the mean, standard deviation and correlation structure to match the observations.
* `bias_adjust_unseen` (EMBCCA-UNSEEN) adjusts the mean and correlation structure while preserving the model's own standard deviation. This is the version used for UNSEEN applications.
To see it applied to the sample data in `example-data/`:

```bash
python examples/example.py
```

This script corrects a small 2x2 grid in both the gridded and the area-mean form, and reports the mean, correlation and variance before and after. See `example-data/README.md` for the array layouts it expects.

### The manuscript analyses

The analyses in `paper/` depends on further packages. Use the conda environment:

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

## Citation

If you use this software, please cite:

* the associated manuscript
* the software release described in `CITATION.cff`

\---

## Notes

* SBCK methods included: **dOTC, MRec, R2D2**
* Unlike these methods, EMBCCA‑UNSEEN preserves variance (critical for UNSEEN applications)

