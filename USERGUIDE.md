(C) Crown Copyright, Met Office. All rights reserved.
See LICENCE in the root of the repository for full licensing details.

# EMBCCA user guide

A guide to applying the `embcca` package to your own data. For the manuscript
analyses see [`paper/README.md`](paper/README.md); for the repository as a
whole see [`README.md`](README.md).

## Contents

- [What the method does](#what-the-method-does)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Choosing a function](#choosing-a-function)
- [Array layouts](#array-layouts)
- [The two variants](#the-two-variants)
- [Adjusting a different period](#adjusting-a-different-period)
- [Missing data](#missing-data)
- [Numerical notes](#numerical-notes)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## What the method does

EMBCCA (Eigenvalue-based Multivariate Bias
Correction with Correlation Alignment) adjusts a model ensemble so that it carries the observed mean and
the observed *correlation between variables*, rather than treating each
variable separately.

Model and observed data are standardised, the model is whitened in its own
principal-component basis, and recoloured in the observed one:

```
Zm = mod_st @ V @ Gamma^-1/2 @ Lambda^1/2 @ W.T
```

where `V, Gamma` and `W, Lambda` are the eigenvectors and eigenvalues of the
standardised model and observed covariance matrices. The result is returned to
physical units.

Each ensemble member is adjusted independently, and for gridded data each grid
cell is adjusted independently as well. Nothing is shared between cells, so the
method imposes no spatial smoothing.

## Installation

The package needs only NumPy and installs on any platform, with no compiler
and no system libraries:

```bash
pip install -e .
```

Python 3.10 or newer. Tested on 3.10, 3.11, 3.12 and 3.13.

The heavier dependencies elsewhere in this repository (SBCK, iris, cartopy)
belong to the manuscript analyses and are not needed to use the method.

## Quickstart

```python
import numpy as np
from embcca import bias_adjust_unseen

rng = np.random.default_rng(0)
obs = rng.normal(size=(30, 2)) * [3.0, 0.5] + [400.0, 22.0]   # (years, vars)
mod = rng.normal(size=(30, 50, 2)) * [5.0, 2.0] + [380.0, 24.0]  # (years, members, vars)

corrected = bias_adjust_unseen(mod, obs)
corrected.shape
# (30, 50, 2)
```

Every member now carries the observed mean and the observed correlation, while
keeping its own standard deviation:

```python
np.allclose(corrected.mean(axis=0), obs.mean(axis=0))   # True
np.allclose(corrected.std(axis=0), mod.std(axis=0))     # True
```

## Choosing a function

Six functions are exported. Two dispatch on the shape of the input; the other
four are the implementations they dispatch to, if you would rather be explicit.

| | area-mean input | gridded input | dispatches on shape |
|---|---|---|---|
| **adjust mean and standard deviation** | `bias_adjust_area_mean` | `bias_adjust_gridded` | `bias_adjust` |
| **adjust mean only** | `bias_adjust_area_mean_unseen` | `bias_adjust_gridded_unseen` | `bias_adjust_unseen` |

All six take the same arguments:

```python
f(mod_calibration, obs_calibration, mod_future=None, eps=1e-6)
```

The dispatchers pick the area-mean form for 2-D or 3-D input and the gridded
form for 4-D or 5-D, and raise `ValueError` for anything else. They forward
`mod_future` and `eps` unchanged, so `bias_adjust_unseen(mod, obs)` and
`bias_adjust_area_mean_unseen(mod, obs)` return exactly the same array. Use the
dispatchers unless you want the layout enforced by the function name.

## Array layouts

Make sure the inputs are in the correct format detailed below.

**Area-mean (non-spatial)**

| argument | shape |
|---|---|
| `mod_calibration` | `(n_years, n_members, n_vars)` |
| `obs_calibration` | `(n_years, n_vars)` |

**Gridded**

| argument | shape |
|---|---|
| `mod_calibration` | `(n_years, n_members, n_lons, n_lats, n_vars)` |
| `obs_calibration` | `(n_years, n_lons, n_lats, n_vars)` |

Three things to note:

- **Time is always first, variables always last.** The ensemble axis is second
  for model data and absent for observations.
- **The two spatial axes are interchangeable.** They are named `n_lons` and
  `n_lats`, but each cell is adjusted independently, so passing them the other
  way round gives the same answer with the same axis order back. The names are
  a convention, not a requirement.
- **`n_vars` can be any number**, not just two. The method generalises to any
  number of jointly adjusted variables.
- **The ensemble axis is optional.** A single realisation can be passed
  without it, as `(n_years, n_vars)` or `(n_years, n_lons, n_lats, n_vars)`.
  It is treated as an ensemble of one and the result omits the axis too, so
  the shape you pass is the shape you get back. `mod_future` must use the same
  layout as `mod_calibration`.

The result always has the same shape and dtype as the array that was adjusted.

> **`obs_calibration` is only partly checked.** `mod_future` is validated
> against `mod_calibration` on every axis, but the observations are not, and
> two mismatches pass silently rather than raising:
>
> - an observation grid **larger** than the model grid is indexed over the
>   model's ranges, so only its leading corner is used and the rest is
>   discarded;
> - a different number of years is accepted, because each block is
>   standardised over its own time axis.
>
> A *smaller* observation grid raises `IndexError`, and a different number of
> variables raises `ValueError`. If your observations are not already on the
> model grid by construction, assert it yourself:
>
> ```python
> assert obs.shape[1:-1] == mod.shape[2:-1]   # gridded: spatial axes agree
> assert obs.shape[0] == mod.shape[0]         # same number of years
> assert obs.shape[-1] == mod.shape[-1]       # same variables
> ```

To go from gridded to area-mean, average the spatial axes first:

```python
mod_area = mod.mean(axis=(2, 3))   # (years, members, vars)
obs_area = obs.mean(axis=(1, 2))   # (years, vars)
```

## The two variants

The only difference is how the adjusted values are returned to physical units:

| | rescaling | mean | standard deviation | correlation |
|---|---|---|---|---|
| `bias_adjust` | `Zm * obs_std + obs_mean` | observed | **observed** | observed |
| `bias_adjust_unseen` | `Zm * mod_std + obs_mean` | observed | **model's own** | observed |

Use `bias_adjust_unseen` for UNSEEN work. UNSEEN draws its statistical power
from the ensemble's larger sampled spread, so replacing that spread with the
observed one removes the property the approach depends on.

Use `bias_adjust` when you want the model to match the observed distribution in
mean and spread as well as correlation.

The two coincide when the model and observed standard deviations already agree.

## Adjusting a different period

Pass `mod_future` to calibrate the transform on one period and apply it to
another, which is the usual bias-adjustment workflow:

```python
corrected = bias_adjust(mod_calibration, obs_calibration, mod_future)
```

The transform is derived from `mod_calibration` against `obs_calibration`, and
`mod_future` is standardised using the **calibration period's** model mean and
standard deviation, so a shift between the two periods survives the adjustment
rather than being absorbed.

The two periods need not be the same length, and the result has `mod_future`'s
shape:

```python
mod_calibration.shape   # (30, 50, 2)   calibration period
mod_future.shape        # (20, 50, 2)   period to adjust
bias_adjust(mod_calibration, obs_calibration, mod_future).shape
# (20, 50, 2)
```

Passing `mod_calibration` itself as `mod_future` gives exactly the same result
as omitting the argument.

The two arrays must agree on every axis except time. Any other disagreement,
including a different number of ensemble members, grid cells, variables or
dimensions, raises `ValueError` before any adjustment is attempted.

## Missing data

Both NaN and masked arrays work, and neither raises an exception.

**NaN** propagates per block. A member or cell whose input contains NaN
produces NaN in the output; its neighbours are unaffected. 

**Masked arrays** are returned as masked arrays with the mask preserved.

There is no skipping: E.g., a fully masked ocean cell still costs the same time as a
unmasked land cell, it just produces nothing. If you are adjusting a large domain that
is mostly missing, consider subsetting to the valid cells first.

`eps` guards against a variable with no variance across time, which would
otherwise divide by zero. It floors standard deviations and eigenvalues alike,
identically in all six functions.

A constant variable therefore stays finite rather than producing NaN, and
comes back at the observed mean. That matters because the variables are
adjusted jointly: without the floor, one constant variable divides by zero and
takes the whole block with it, including variables that were perfectly well
behaved.

## Numerical notes

**Dtype is preserved.** float32 in, float32 out. Internally the covariance and
eigen-decomposition are computed in float64 regardless, then the result is
returned in the input dtype.

**Use one dtype consistently.** Results depend measurably on the dtype of the
input, more so than the usual float32-versus-float64 rounding would suggest.
Adjusting the same data as float32 and as float64 does not give the same
answer to within rounding, so do not mix the two across a workflow or compare
results produced at different precisions.

**Inputs are never modified**, including `mod_future`.

**There is no random element**, so a given input always gives the same output
on a given machine and NumPy build.

## Performance

Cost is linear in the number of `(cell, member)` blocks.

## Worked example

[`examples/example.py`](examples/example.py) applies the method to a small
sample grid included in the repository, in both the gridded and area-mean
forms, and reports the mean, correlation and standard deviation before and
after:

```bash
python examples/example.py
```

It needs only the package and NumPy. See
[`example-data/README.md`](example-data/README.md) for the array layouts it
uses.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ValueError: Unsupported dimensions` | The dispatchers accept 2-D or 3-D (area-mean) and 4-D or 5-D (gridded) model input. Gridded input needs both spatial axes, even if one is length 1. |
| `ValueError: expected 5 with an ensemble axis, or 4 without one` | An explicit form was given the other layout's rank, e.g. area-mean data passed to `bias_adjust_gridded`. Use the dispatcher, or the matching function. |
| `ValueError: Only the leading time axis may differ` | `mod_future` disagrees with `mod_calibration` somewhere other than time. See [Adjusting a different period](#adjusting-a-different-period). |
| Any other `ValueError`, `IndexError`, or `matmul` dimension error | Some array does not match the expected layout. Check every axis of all three arrays against [Array layouts](#array-layouts), rather than assuming the reported axis is the one at fault: the error usually surfaces at whichever operation happens to reach the mismatch first. |
| All-NaN output | The input block contained NaN. A constant variable does *not* cause this; `eps` keeps it finite. See [Missing data](#missing-data). |
| Corrected correlation does not match the observations | Check the variable axis is last and time is first. A transposed array is adjusted without complaint. |
| Output looks plausible but wrong, with no error | Most likely an `obs_calibration` mismatch, which is only partly checked. See the warning under [Array layouts](#array-layouts). |
| Results differ between two runs of the same script | Check the input dtype is the same in both. See [Numerical notes](#numerical-notes). |
| Adjusted spread looks wrong | You may want the other variant. See [The two variants](#the-two-variants). |

## Citing

See [`CITATION.cff`](CITATION.cff) and the citation section of
[`README.md`](README.md).
