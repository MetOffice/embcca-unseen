# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""EMBCCA-UNSEEN: fast multivariate bias correction for compound extremes.

Eigen-decomposition-based Multivariate Bias Correction with Covariance
Alignment, for use with the UNSEEN approach to compound event analysis.

Unlike other multivariate bias-adjustment methods, EMBCCA transfers the
observed mean and correlation structure onto the model while preserving the
model's own variance -- the property that makes it suitable for UNSEEN, which
depends on the model ensemble's larger sampled spread.

Example
-------
>>> import numpy as np
>>> from embcca import correct
>>> obs = np.random.default_rng(0).normal(size=(30, 2))          # (years, vars)
>>> mod = np.random.default_rng(1).normal(size=(30, 50, 2))      # (years, members, vars)
>>> corrected = correct(mod, obs)
>>> corrected.shape
(30, 50, 2)

The same entry point handles gridded data, where ``mod`` is
``(years, members, lons, lats, vars)`` and ``obs`` is
``(years, lons, lats, vars)``.

This package contains the bias-adjustment method only. The analyses that
produced the manuscript figures live in ``paper/`` and carry their own,
heavier dependencies.
"""

from embcca.correction import correct, correct_area_mean, correct_gridded

__version__ = "1.0.0"

__all__ = [
    "correct",
    "correct_area_mean",
    "correct_gridded",
    "__version__",
]
