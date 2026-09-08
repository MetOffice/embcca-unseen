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

Both scripts apply the bias adjustment by calling `embcca.correct` from the
`embcca-unseen` package.

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

## Running

Both scripts are configured by editing the constants under
`USER SETTINGS` at the top of each file. At minimum set `DATA_DIR` and
`OUTDIR`; also there are `SEED`, the year range, input filenames and variable
names, and `MPL_BACKEND` (leave as `"Agg"` for headless or HPC use, `"Qt5Agg"`
for interactive).

> **`OUTDIR` must end with a trailing slash.** Most output paths are built by
> string concatenation, so `"/path/to/output"` writes to `/path/to/outputMaps/`
> rather than into the directory you meant. Use `"/path/to/output/"`.

Run them from this directory, so `fidelity_test_cube` is importable:

```bash
cd paper
python Multi-DePreSys4-Paper-area_avg_final_multiscatter.py
python Multi-DePreSys4-Paper-area_full_final_China.py
```

The China script times each bias-adjustment method and prints the results, so
avoid running anything else heavy on the machine if those timings matter.

## Outputs

Figure directories are created as needed.

Hunan, under `OUTDIR`: `Line_plots/`, `Scatter_plots/`,
`Statistical_Comparison/`, `SVM_Comparison/`, `Exceedance_Comparison/`,
`Fidelity_Testing/`.

China, under `OUTDIR`: `Maps/` (mean maps, correlation maps, and
correlation-difference maps). Method timings are printed to stdout, not saved.
