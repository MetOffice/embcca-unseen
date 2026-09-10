(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE.txt in the root of the repository for full licensing details.
# EMBCCA
# A new fast multivariate bias correction technique: a case study for compound events in Hunan Province, China, using the UNSEEN approach

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

After installation the user has the choice between bias adjustment with EMBCCA (adjust mean and standard deviation of the distribution) or EMBCCA-UNSEEN (only adjust the mean). The bias adjustment can be appled to a numpy array grid of latitudes and longitudes or on an area average. See `USERGUIDE.md` for full details.

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

## Citation

Please cite:

* The associated manuscript

\---

## Notes

* SBCK methods included: **dOTC, MRec, R2D2**
* Unlike these methods, EMBCCA‑UNSEEN preserves variance (critical for UNSEEN applications)

