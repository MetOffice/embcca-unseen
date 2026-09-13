# (C) Crown Copyright, Met Office. All rights reserved.
# This file is released under the BSD 3-Clause license.
# See LICENCE in the root of the repository for full licensing details.
"""
Regrid ERA5-Land observations onto the DePreSys4 grid.

Loads seasonal ERA5-Land and DePreSys4 datasets,
regrids observations to the model grid,
and writes regridded observation files for
subsequent EMBCCA-UNSEEN analysis.
"""

from __future__ import annotations

import iris
import iris.analysis
import iris.coords as icoords


PROVINCE = "China"
YEARS = [1992, 2023]
SEASON = "jja"

OUT_DIR = ("path/to/data_dir")

SEASON_NAMES = {
    "djf": "winter",
    "mam": "spring",
    "jja": "summer",
    "son": "autumn",
}


def regrid_cube(
    cube: iris.cube.Cube,
    template: iris.cube.Cube,
    regrid_scheme,
) -> iris.cube.Cube:
    """
    Regrid one cube onto another using the specified scheme.
    """

    assert template is not None, (
        f"Template is not valid --> {type(template)}"
    )

    if template.ndim == 3:
        template = template[0]

    cube.coord("longitude").coord_system = (
        template.coord("longitude").coord_system
    )
    cube.coord("latitude").coord_system = (
        template.coord("latitude").coord_system
    )

    for coord in ["latitude", "longitude"]:

        if not template.coord(coord).has_bounds():
            template.coord(coord).guess_bounds()

        if not cube.coord(coord).has_bounds():
            cube.coord(coord).guess_bounds()

        cube.coord(coord).coord_system = (
            template.coord(coord).coord_system
        )

        cube.coord(coord).units = (
            template.coord(coord).units
        )

    return cube.regrid(template, regrid_scheme)


def run() -> None:

    filename_season = SEASON_NAMES.get(SEASON, SEASON)

    tas_obs_fname = (
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_tas_obs_ERA5_Land.nc"
    )

    pr_obs_fname = (
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_pr_obs_ERA5_Land.nc"
    )

    tas_mod_fname = (
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_tas_model_DePreSys4.nc"
    )

    pr_mod_fname = (
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_pr_model_DePreSys4.nc"
    )

    tas_obs_cube = iris.load_cube(tas_obs_fname)
    pr_obs_cube = iris.load_cube(pr_obs_fname)

    tas_mod_cube = iris.load_cube(tas_mod_fname)
    pr_mod_cube = iris.load_cube(pr_mod_fname)

    regrid_scheme = iris.analysis.Linear()

    tas_obs_cube_regrid = regrid_cube(
        tas_obs_cube,
        tas_mod_cube,
        regrid_scheme,
    )

    pr_obs_cube_regrid = regrid_cube(
        pr_obs_cube,
        pr_mod_cube,
        regrid_scheme,
    )

    iris.save(
        tas_obs_cube_regrid,
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_tas_obs_ERA5_Land_regridded.nc",
    )

    iris.save(
        pr_obs_cube_regrid,
        f"{OUT_DIR}/"
        f"{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_"
        f"{filename_season}_pr_obs_ERA5_Land_regridded.nc",
    )

    print("Finished regridding.")


if __name__ == "__main__":
    run()