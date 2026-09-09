# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""EMBCCA-UNSEEN: fast multivariate bias correction for compound extremes.

Eigen-decomposition-based Multivariate Bias Correction with Covariance
Alignment, for use with the UNSEEN approach to compound event analysis.

There are two entry points, differing only in how the adjusted data is
returned to physical units:

``bias_adjust``
    Transfers the observed mean, standard deviation and correlation structure
    onto the model.

``bias_adjust_unseen``
    Transfers the observed mean and correlation structure but keeps the
    model's own standard deviation. This is what makes the method suitable for
    UNSEEN, which depends on the model ensemble's larger sampled spread.

Example
-------
>>> import numpy as np
>>> from embcca import bias_adjust_unseen
>>> obs = np.random.default_rng(0).normal(size=(30, 2))          # (years, vars)
>>> mod = np.random.default_rng(1).normal(size=(30, 50, 2))      # (years, members, vars)
>>> corrected = bias_adjust_unseen(mod, obs)
>>> corrected.shape
(30, 50, 2)

Either entry point also handles gridded data, where ``mod`` is
``(years, members, lons, lats, vars)`` and ``obs`` is
``(years, lons, lats, vars)``.

Both accept an optional third argument, ``mod_future``. When given, the
transform is calibrated on the first two arguments and then applied to it,
and the result has ``mod_future``'s shape:

>>> future = np.random.default_rng(2).normal(size=(20, 50, 2))
>>> bias_adjust_unseen(mod, obs, future).shape
(20, 50, 2)

This package contains the bias-adjustment method only. The analyses that
produced the manuscript figures live in ``paper/`` and carry their own,
heavier dependencies.
"""

from embcca.adjustment import (
    bias_adjust,
    bias_adjust_area_mean,
    bias_adjust_area_mean_unseen,
    bias_adjust_gridded,
    bias_adjust_gridded_unseen,
    bias_adjust_unseen,
)

__version__ = "1.0.0"

__all__ = [
    "bias_adjust",
    "bias_adjust_unseen",
    "bias_adjust_area_mean",
    "bias_adjust_gridded",
    "bias_adjust_area_mean_unseen",
    "bias_adjust_gridded_unseen",
    "__version__",
]
