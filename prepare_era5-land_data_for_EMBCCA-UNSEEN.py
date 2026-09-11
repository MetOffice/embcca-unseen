"""
Extract seasonal temperature or precipitation for a specified country from ERA5-Land
monthly data.

The script:
- loads monthly ERA5-Land files for each requested year and season month
- subsets to the specified country shape
- converts tas to Celsius, pr to monthly totals in mm
- aggregates each year to a seasonal value
- stacks the annual seasonal cubes along a year dimension
- writes a single NetCDF output file
The output file is named according to the country, years, season, and variable name.
"""

from __future__ import annotations

import calendar
import os
import warnings

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import iris
import iris.analysis
import iris.coord_categorisation
import iris.coords as icoords
import iris.coord_systems as icoord_systems
import iris.util
import numpy as np
import shapely.ops

# Keep Iris date precision behavior stable.
iris.FUTURE.date_microseconds = True

# Ignore optional metadata warnings common in ERA5-Land files.
warnings.filterwarnings(
    "ignore",
    message=r"Missing CF-netCDF measure variable 'areacella'.*",
    module=r"iris\.fileformats\.cf",
)

PROVINCE = "China"
VARNAME = "tas"  # "tas" or "pr"
YEARS = [1992, 2023]
SEASON = "jja"  # e.g. "djf", "mam", "jja", "son", or [6, 7, 8]

SHAPEFILE = "/data/users/appldata/Data/Spatial_data/Natural_Earth/v5.0.1/ne_10m_admin_0_countries.shp"
OUT_DIR = "/data/users/cst/Projects/CSSP/CSSP_China/FY2526/Yiwei_paper/data"
ERA5_LAND_PATH = "/data/users/appldata/Data/OBS-ERA5-Land/"

MINIMUM_WEIGHT = 0.5

SEASON_MONTHS = {"djf": [12, 1, 2],
                 "mam": [3, 4, 5],
                 "jja": [6, 7, 8],
                 "son": [9, 10, 11],}

MONTH_LENGTHS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,}


def _parse_season(season: str | list[int] | tuple[int, ...]) -> tuple[list[int], str]:
    '''
    Parse season input into a month list and a label.

    Inputs:
    ------
    season : str | list[int] | tuple[int, ...]
        Season definition. Can be a known key (e.g. "jja") or explicit months.

    Returns:
    -------
    months, label : tuple[list[int], str]
        The month numbers and a lowercase label used in output names.
    '''
    if isinstance(season, str):
        label = season.lower()
        months = SEASON_MONTHS.get(label)
        if months is None:
            raise ValueError(f"Unknown season {season!r}. Use one of {sorted(SEASON_MONTHS)} or a month list.")

        return months, label

    months = [int(month) for month in season]
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError("Season month list must contain values between 1 and 12.")

    label = "m" + "_".join(str(month) for month in months)
    return months, label

def _load_region_shape(country_name: str):
    '''
    Load the shapefile and return the geometry for the given country name.
    Raises ValueError if the country is not found.

    Inputs:
    ------
    country_name : str
        The name of the country to extract from the shapefile.
    
    Returns:
    -------
    shp : shapely.geometry.base.BaseGeometry
        The geometry of the specified country.
    '''
    records = shpreader.Reader(SHAPEFILE).records()
    geometries = [record.geometry for record in records if record.attributes.get("NAME_LONG") == country_name]
    if not geometries:
        raise ValueError(f"Country {country_name!r} not found in shapefile: {SHAPEFILE}")
    shp = shapely.ops.unary_union(geometries)
    return shp

def _add_month_num_dim(cube: iris.cube.Cube) -> iris.cube.Cube:
    '''
    Add a month_number coordinate to the cube if it doesn't already exist.

    Inputs:
    ------
    cube : iris.cube.Cube
        The input cube to which the month_number coordinate will be added.
    
    Returns:
    -------
    cube : iris.cube.Cube
        The cube with the month_number coordinate added.
    '''
    try:
        cube.coord("month_number")
    except Exception:
        iris.coord_categorisation.add_month_number(cube, "time", name="month_number")
    return cube

def _add_year_dim(cube: iris.cube.Cube) -> iris.cube.Cube:
    '''
    Add a year coordinate to the cube if it doesn't already exist.

    Inputs:
    ------
    cube : iris.cube.Cube
        The input cube to which the year coordinate will be added.

    Returns:
    -------
    cube : iris.cube.Cube
        The cube with the year coordinate added.
    '''
    try:
        cube.coord("year")
    except Exception:
        iris.coord_categorisation.add_year(cube, "time", name="year")
    return cube


def _add_season_year_dim(cube: iris.cube.Cube) -> iris.cube.Cube:
    '''Add a season_year coordinate to the cube if it doesn't already exist.'''
    try:
        cube.coord("season_year")
    except Exception:
        iris.coord_categorisation.add_season_year(cube, "time", name="season_year")
    return cube


def _season_month_calendar_year(season_year: int, month: int, season_months: list[int]) -> int:
    '''Map a season year and month to the actual calendar year for the month.'''
    if season_months and season_months[0] > season_months[-1]:
        return season_year - 1 if month >= season_months[0] else season_year
    return season_year

def _subset_by_shape(cube: iris.cube.Cube, region_shape, minimum_weight: float = 0.5) -> iris.cube.Cube:
    '''
    Subset the cube to the region defined by the given shape, with an optional border.

    Inputs:
    ------
    cube : iris.cube.Cube
        The input cube to be subsetted.
    region_shape : shapely.geometry.base.BaseGeometry
        The geometry defining the region to subset to.
    minimum_weight : float, optional
        The minimum weight for the mask (default is 0.5).

    Returns:
    -------
    cube : iris.cube.Cube
        The subsetted cube.
    '''
    if region_shape is None:
        return cube

    cube = _ensure_lat_lon_crs(cube)

    cube = iris.util.mask_cube_from_shape(cube, region_shape, shape_crs=ccrs.PlateCarree(),
                                          minimum_weight=minimum_weight)
    return cube

def _ensure_lat_lon_crs(cube: iris.cube.Cube) -> iris.cube.Cube:
    '''
    Ensure latitude and longitude coordinates have a geographic CRS.

    Inputs:
    ------
    cube : iris.cube.Cube
        Cube whose horizontal coordinates will be checked.

    Returns:
    -------
    cube : iris.cube.Cube
        Cube with latitude/longitude coord_system set when missing.
    '''
    lat_coord = cube.coord("latitude")
    lon_coord = cube.coord("longitude")

    existing_cs = lat_coord.coord_system or lon_coord.coord_system
    if existing_cs is None:
        # Natural Earth polygons are lon/lat in WGS84
        existing_cs = icoord_systems.GeogCS(6378137.0, inverse_flattening=298.257223563)

    if lat_coord.coord_system is None:
        lat_coord.coord_system = existing_cs
    if lon_coord.coord_system is None:
        lon_coord.coord_system = existing_cs

    return cube


def _load_month_cube(varname: str, year: int, month: int, region_shape) -> iris.cube.Cube:
    if varname == "tas":
        varstr = "2m_temperature"
    elif varname == "pr":
        varstr = "total_precipitation"
    else:
        raise ValueError(f"Unsupported varname: {varname!r}. Use 'tas' or 'pr'.")

    month_fname = (f"{ERA5_LAND_PATH}/monthly/{varstr}/era5-land_monthly_{varstr}_{year}{month:02d}.nc")

    if not os.path.exists(month_fname):
        raise FileNotFoundError(f"ERA5-Land file not found: {month_fname}")

    cube = iris.load_cube(month_fname)
    cube = _add_year_dim(cube)
    cube = _add_month_num_dim(cube)
    cube = _add_season_year_dim(cube)
    cube = _subset_by_shape(cube, region_shape, minimum_weight=MINIMUM_WEIGHT)
    cube.attributes = {}

    if varname == "tas":
        cube.data = cube.data - 273.15
        cube.units = "celsius"
        cube.long_name = "mean_season_temperature"
    elif varname == "pr":
        num_days = MONTH_LENGTHS[month]
        cube.data = (cube.data * num_days) * 1000.0
        cube.units = "mm"
        cube.long_name = "total_season_precipitation"

    return cube


def _season_cube(cube: iris.cube.Cube, varname: str,
                 season: str | list[int] | tuple[int, ...]) -> tuple[iris.cube.Cube, str]:
    '''
    Extract the chosen seasonal months from the cube and aggregate by year.

    Inputs:
    ------
    cube : iris.cube.Cube
        The input cube from which to extract seasonal data.
    varname : str
        The name of the variable (e.g., "tas" or "pr") to determine the aggregation method.
    season : str | list[int] | tuple[int, ...]
        Season definition, e.g. "jja" or [6, 7, 8].
    
    Returns:
    -------
    out : tuple[iris.cube.Cube, str]
        A tuple containing the aggregated seasonal cube and a descriptive long name.
    '''
    months, season_label = _parse_season(season)
    seasonal = cube.extract(iris.Constraint(month_number=lambda cell: int(getattr(cell, "point", cell)) in months))

    if seasonal is None:
        available_months = sorted({int(value) for value in cube.coord("month_number").points.tolist()})
        raise ValueError(f"No data found for season {season_label!r} (months={months}) in cube. Available months are {available_months}.")

    season_years = seasonal.coord("season_year").points.astype(int)
    season_months = seasonal.coord("month_number").points.astype(int)

    complete_years = [int(year) for year in np.unique(season_years) if set(season_months[season_years == year]) == set(months)]
    seasonal = seasonal.extract(iris.Constraint(season_year=lambda cell: int(getattr(cell, "point", cell)) in complete_years))

    if seasonal is None:
        raise ValueError(f"No complete {season_label!r} seasons found in cube.")

    if varname == "tas":
        seasonal = seasonal.copy()
        seasonal.data = seasonal.data - 273.15
        out = seasonal.aggregated_by("season_year", iris.analysis.MEAN)
        out.units = "celsius"
        return out, f"mean_{season_label}_temperature"

    if varname == "pr":
        seasonal = seasonal.copy()
        for month in months:
            month_mask = seasonal.coord("month_number").points == month
            seasonal.data[month_mask, ...] = seasonal.data[month_mask, ...] * MONTH_LENGTHS[month]
        out = seasonal.aggregated_by("season_year", iris.analysis.SUM)
        out.units = "mm"
        return out, f"total_{season_label}_precipitation"

    raise ValueError(f"Unsupported varname: {varname!r}. Use 'tas' or 'pr'.")


def _build_output_cube(season_cubelist: iris.cube.CubeList, long_name: str) -> iris.cube.Cube:
    '''
    Build a single output cube from the list of seasonal cubes, stacking season_year, latitude, and longitude.

    Inputs:
    ------
    season_cubelist : iris.cube.CubeList
        The list of seasonal cubes to combine.
    long_name : str
        The long name for the output cube.
    
    Returns:
    -------
    output : iris.cube.Cube
        The combined output cube with the specified long name.
    '''
    if not season_cubelist:
        raise ValueError("No seasonal cubes to combine; check file discovery and season selection.")

    years = [int(c.coord("season_year").points[0]) for c in season_cubelist]
    years = sorted(years)

    sample = season_cubelist[0]
    lats = sample.coord("latitude").points
    lons = sample.coord("longitude").points
    units = sample.units

    data = np.full((len(years), len(lats), len(lons)), np.nan, dtype=np.float32)

    for cube in season_cubelist:
        year = int(cube.coord("season_year").points[0])
        year_index = years.index(year)
        if cube.shape[-2:] != (len(lats), len(lons)):
            raise ValueError(f"Grid mismatch across cubes. Expected {(len(lats), len(lons))}, got {cube.shape[-2:]}")
        data[year_index, :, :] = np.asarray(cube.data, dtype=np.float32)

    output = iris.cube.Cube(data, long_name=long_name, units=units,
                            dim_coords_and_dims=[(icoords.DimCoord(years, long_name="season_year"), 0),
                                                 (icoords.DimCoord(lats, standard_name="latitude", units="degrees"), 1),
                                                 (icoords.DimCoord(lons, standard_name="longitude", units="degrees"), 2)])
    return output


def run() -> None:
    region_shape = _load_region_shape(PROVINCE)
    months, season_label = _parse_season(SEASON)

    os.makedirs(OUT_DIR, exist_ok=True)
    year_dir = os.path.join(OUT_DIR, "individual_years")
    os.makedirs(year_dir, exist_ok=True)

    season_cubelist = iris.cube.CubeList()
    output_long_name = None
    skipped = 0

    for season_year in range(YEARS[0], YEARS[-1] + 1):
        print(f"Processing season year {season_year}...")
        annual_fname = f"{year_dir}/{PROVINCE}_{season_year}_{season_label}_{VARNAME}_obs_ERA5_Land.nc"

        if os.path.exists(annual_fname):
            season_cube_year = iris.load_cube(annual_fname)
            print(f"Loaded existing file for season year {season_year}.")
        else:
            month_cubelist = iris.cube.CubeList()
            incomplete = False
            for month in months:
                print(f"  Processing month {month}...")
                month_year = _season_month_calendar_year(season_year, month, months)
                if VARNAME == "tas":
                    varname_str = "2m_temperature"
                elif VARNAME == "pr":
                    varname_str = "total_precipitation"
                month_fname = f"{ERA5_LAND_PATH}/monthly/{varname_str}/era5-land_monthly_{varname_str}_{month_year}{month:02d}.nc"
                if not os.path.exists(month_fname):
                    print(f"    Missing file {month_fname}; skipping season year {season_year}.")
                    incomplete = True
                    break
                month_cube = _load_month_cube(VARNAME, month_year, month, region_shape)
                month_cubelist.append(month_cube)

            if incomplete or not month_cubelist:
                print(f"  No complete months found for season year {season_year}; skipping.")
                skipped += 1
                continue

            season_months_cube = month_cubelist.merge_cube()
            season_cube_year, output_long_name = _season_cube(season_months_cube, VARNAME, SEASON)

            iris.save(season_cube_year, annual_fname)
            print(f"Saved seasonal cube for season year {season_year} to {annual_fname}.")

        season_cube_year.attributes = {}
        season_cube_year.cell_methods = ()

        season_cubelist.append(season_cube_year)

    if skipped:
        print(f"Skipped {skipped} years with incomplete or missing season data.")

    if not season_cubelist:
        raise ValueError("No seasonal cubes were produced.")

    output_cube = _build_output_cube(season_cubelist, output_long_name or "seasonal_era5_land")

    print(output_cube.coord("season_year").points)
    print(output_cube.data.max())
    print(output_cube.data.min())

    out_fname = f"{OUT_DIR}/{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_{season_label}_{VARNAME}_obs_ERA5_Land.nc"
    iris.save(output_cube, out_fname)
    print(f"Wrote {out_fname}")


if __name__ == "__main__":
    run()