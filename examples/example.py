# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE.txt in the root of the repository for full licensing details.

"""Applying EMBCCA to a small worked example.

Runs on the 2x2 grid in example-data/, which is stored ready to use: no
reshaping needed. Requires only the embcca package and NumPy, so it works in a
plain `pip install -e .` environment with none of the heavier dependencies the
manuscript analyses need.

    python examples/example.py
"""

from pathlib import Path

import numpy as np

from embcca import bias_adjust_unseen

DATA = Path(__file__).resolve().parent.parent / "example-data"

# The variable axis is ordered (precipitation, temperature).
PRECIP, TEMP = 0, 1

LABEL_WIDTH = 22
COL_WIDTH = 11


def correlation(block):
    """Pearson r between the two variables of a (n_years, n_vars) block."""
    return np.corrcoef(block, rowvar=False)[PRECIP, TEMP]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_table(headers, rows, label_width=LABEL_WIDTH, col_width=COL_WIDTH):
    """Print a labelled numeric table.

    `rows` is a sequence of (label, values) pairs, one value per header.
    """
    head = "".join(f"{h:>{col_width}}" for h in headers)
    print(f"\n{'':{label_width}}{head}")
    for label, values in rows:
        cells = "".join(f"{v:>{col_width}.4f}" for v in values)
        print(f"{label:{label_width}}{cells}")


def print_checks(checks):
    """Print a list of (label, bool) results."""
    for label, passed in checks:
        print(f"  {label:30} {passed}")


def print_inputs(mod, obs):
    print(f"model {mod.shape}  (year, member, lon, lat, variable)")
    print(f"obs   {obs.shape}        (year, lon, lat, variable)")


def print_area_mean_summary(obs_area, mod_area, corrected_area, member=0):
    """Compare observed, raw and corrected for one ensemble member.

    EMBCCA gives every member the observed mean and the observed correlation
    between the variables, while keeping that member's own variance. The rows
    below are that statement, measured.
    """
    def triple(fn):
        return fn(obs_area), fn(mod_area[:, member, :]), fn(corrected_area[:, member, :])

    rows = [("correlation (pr,tas)", triple(correlation))]
    for name, var in (("precip", PRECIP), ("temp", TEMP)):
        rows.append((f"mean {name}", triple(lambda b, v=var: b[:, v].mean())))
    for name, var in (("precip", PRECIP), ("temp", TEMP)):
        rows.append((f"std {name}", triple(lambda b, v=var: b[:, v].std())))

    print_table(("observed", "raw model", "corrected"), rows)


def print_area_mean_checks(obs_area, mod_area, corrected_area):
    member_r = [correlation(corrected_area[:, m, :]) for m in range(mod_area.shape[1])]
    print("\nAcross all members:")
    print_checks([
        ("mean matches observed",
         np.allclose(corrected_area.mean(axis=0), obs_area.mean(axis=0))),
        ("correlation matches observed",
         np.allclose(member_r, correlation(obs_area))),
        ("variance kept from the model",
         np.allclose(corrected_area.std(axis=0), mod_area.std(axis=0))),
    ])


def print_cell_table(obs, mod, corrected, member=0):
    """Per-grid-cell correlation, showing each cell corrected independently."""
    n_lons, n_lats = mod.shape[2], mod.shape[3]
    rows = []
    for ilon in range(n_lons):
        for ilat in range(n_lats):
            rows.append((
                f"cell ({ilon},{ilat})",
                (
                    correlation(obs[:, ilon, ilat, :]),
                    correlation(mod[:, member, ilon, ilat, :]),
                    correlation(corrected[:, member, ilon, ilat, :]),
                ),
            ))
    print_table(("observed", "raw model", "corrected"), rows)


# ---------------------------------------------------------------------------


def main():
    mod = np.load(DATA / "test_grid_1992_2021_jja_model_DePreSys4.npy")
    obs = np.load(DATA / "test_grid_1992_2021_jja_obs_ERA5_Land.npy")
    print_inputs(mod, obs)

    # --- area-mean: average away the two spatial axes first ---
    mod_area = mod.mean(axis=(2, 3))   # (year, member, variable)
    obs_area = obs.mean(axis=(1, 2))   # (year, variable)
    corrected_area = bias_adjust_unseen(mod_area, obs_area)

    print(f"\n--- area-mean ---\n{mod_area.shape} -> {corrected_area.shape}")
    print_area_mean_summary(obs_area, mod_area, corrected_area)
    print_area_mean_checks(obs_area, mod_area, corrected_area)

    # --- gridded: the same call, applied independently at each grid cell ---
    corrected = bias_adjust_unseen(mod, obs)

    print(f"\n--- gridded ---\n{mod.shape} -> {corrected.shape}")
    print_cell_table(obs, mod, corrected)
    print("\nEvery cell now carries its own observed correlation, and each is\n"
          "corrected independently of its neighbours.")


if __name__ == "__main__":
    main()
