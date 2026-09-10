'''
Extract seasonal temperature or precipitation for specified country from DePreSys4
data that has been downloaded from the CEDA archive. The output is a single netCDF file with the seasonal
data stacked by sub_experiment_id, realisation, leadtime, latitude, and longitude.
The output file is named according to the country, years, season, and variable name.
'''

from glob import glob
import os
import re
import warnings
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import iris
import iris.coord_categorisation
import iris.coords as icoords
import iris.coord_systems as icoord_systems
import iris.util
import numpy as np
import shapely.ops

# Opt in to the future Iris date precision behavior to avoid runtime warning.
iris.FUTURE.date_microseconds = True

# Ignore missing optional cell-area metadata in some CEDA files.
warnings.filterwarnings("ignore", message=r"Missing CF-netCDF measure variable 'areacella'.*",
                        module=r"iris\.fileformats\.cf")


PROVINCE = "China"
VARNAME = "tas"  # "tas" or "pr"
YEARS = [1961, 1963]
# YEARS = [1992, 2023]
SEASON = "jja"  # e.g. "djf", "mam", "jjas", or [6, 7, 8]

SHAPEFILE = "Natural_Earth/v5.0.1/ne_10m_admin_0_countries.shp"
OUT_DIR = "data"
RAW_DATA_DIR = "CMIP6"
INSTITUTION_ID = "MOHC"
SOURCE_ID = "HadGEM3-GC31-MM"

MINIMUM_WEIGHT = 0.5
LAT_RANGE = (15, 55)
LON_RANGE = (70, 140)

SEASON_MONTHS = {"djf": [12, 1, 2],
                 "mam": [3, 4, 5],
                 "jja": [6, 7, 8],
                 "son": [9, 10, 11]}

MONTH_LENGTHS = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


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

def _get_raw_data_filenames(varname: str) -> list[str]:
    '''
    Construct glob patterns for CEDA DePreSys4 raw data files.

    Inputs:
    ------
    varname : str
        The variable name, either "tas" for temperature or "pr" for precipitation.
    
    Returns:
    -------
    patterns : list[str]
        Glob patterns for both hindcast and forecast experiment folders.
    '''
    if varname not in {"tas", "pr"}:
        raise ValueError(f"Unsupported varname: {varname!r}. Use 'tas' or 'pr'.")

    # Example CEDA layouts:
    # /.../DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/sYYYY-r*i1p1f*/Amon/<var>/<grid>/<version>/...
    # /.../DCPP/MOHC/HadGEM3-GC31-MM/dcppB-forecast/sYYYY-r*i1p1f*/Amon/<var>/<grid>/<version>/...
    patterns = [(f"{RAW_DATA_DIR}/DCPP/{INSTITUTION_ID}/{SOURCE_ID}/dcppA-hindcast/"
                 f"s????-r*i1p1f*/Amon/{varname}/**/"
                 f"{varname}_Amon_*_dcppA-hindcast_s????-r*i1p1f*_*.nc"),
                 (f"{RAW_DATA_DIR}/DCPP/{INSTITUTION_ID}/{SOURCE_ID}/dcppB-forecast/"
                  f"s????-r*i1p1f*/Amon/{varname}/**/"
                  f"{varname}_Amon_*_dcppB-forecast_s????-r*i1p1f*_*.nc")]
    return patterns

def _sub_experiment_id_from_filename(filename: str) -> str:
    '''
    Extract the sub_experiment_id (e.g., "s1992") from the filename.
    Raises ValueError if the sub_experiment_id cannot be parsed.

    Inputs:
    ------
    filename : str
        The filename from which to extract the sub_experiment_id.
    
    Returns:
    -------
    str
        The extracted sub_experiment_id.
    '''
    text = os.path.basename(filename)
    match = re.search(r"_s(\d{4})-r\d+i\d+p\d+f\d+_", text)
    if not match:
        # Some CEDA layouts include the token in a parent directory name.
        match = re.search(r"/s(\d{4})-r\d+i\d+p\d+f\d+(?:/|$)", filename)
    if not match:
        raise ValueError(f"Could not parse sub_experiment_id from {filename}")
    sub_experiment_id = match.group(1)
    return sub_experiment_id

def _realisation_from_filename(filename: str) -> int:
    '''
    Extract the realisation number (e.g., 9 from r9i1p1f2) from the filename or path.

    Inputs:
    ------
    filename : str
        The filename from which to extract the realisation number.

    Returns:
    -------
    realisation : int
        The extracted realisation number.
    '''
    text = os.path.basename(filename)
    match = re.search(r"_s\d{4}-r(\d+)i\d+p\d+f\d+_", text)
    if not match:
        match = re.search(r"/s\d{4}-r(\d+)i\d+p\d+f\d+(?:/|$)", filename)
    if not match:
        raise ValueError(f"Could not parse realisation from {filename}")
    return int(match.group(1))

def _remove_unwanted_files_from_filename_list(filenames: list[str],
                                              years_wanted: list[int]) -> list[str]:
    '''
    Filter the list of filenames to only include those that match the desired years.

    Inputs:
    ------
    filenames : list[str]
        The list of filenames to filter.
    years_wanted : list[int]
        The list of years to include in the filtered filenames.

    Returns:
    -------
    filtered : list[str]
        The filtered list of filenames that match the desired years.
    '''
    filtered = []
    for filename in filenames:
        for year in years_wanted:
            if f"s{year}" in filename:
                filtered.append(filename)
                break
    return filtered

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

    Some CEDA files provide lat/lon axes without a declared coordinate system,
    which causes iris.util.mask_cube_from_shape to raise an IrisError.

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

def _load_cubes(varname: str, years: list[int], region_shape) -> iris.cube.CubeList:
    '''
    Load the raw data cubes for the specified variable and years, and subset them to the given region shape.

    Inputs:
    ------
    varname : str
        The name of the variable to load (e.g., "tas" or "pr").
    years : list[int]
        The list of years to load.
    region_shape : shapely.geometry.base.BaseGeometry
        The geometry defining the region to subset to.

    Returns:
    -------
    cubelist : iris.cube.CubeList
        The list of loaded and subsetted cubes.
    '''
    lat_constraint = iris.Constraint(latitude=lambda cell: LAT_RANGE[0] < cell < LAT_RANGE[1])
    lon_constraint = iris.Constraint(longitude=lambda cell: LON_RANGE[0] < cell < LON_RANGE[1])

    years_wanted = list(range(int(years[0]), int(years[-1] + 1)))
    patterns = _get_raw_data_filenames(varname)
    filenames = []
    for pattern in patterns:
        filenames.extend(glob(pattern, recursive=True))
    discovered = sorted(set(filenames))
    filenames = _remove_unwanted_files_from_filename_list(discovered, years_wanted)

    print(f"Discovered {len(discovered)} files for {varname}; {len(filenames)} match years {years[0]}-{years[-1]}")
    if not filenames:
        pattern_lines = "\n".join(patterns)
        raise FileNotFoundError(f"No files found for varname={varname!r} and years={years}. Searched patterns:\n{pattern_lines}")

    cubelist = iris.cube.CubeList()
    for filename in filenames:
        cube = iris.load_cube(filename, lat_constraint & lon_constraint)
        cube = _add_month_num_dim(cube)
        cube = _add_year_dim(cube)
        cube = _subset_by_shape(cube, region_shape, minimum_weight=MINIMUM_WEIGHT)
        cube.attributes = {"sub_experiment_id": _sub_experiment_id_from_filename(filename),
                           "realisation": _realisation_from_filename(filename)}
        cubelist.append(cube)

    return cubelist

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
    seasonal = cube.extract(
        iris.Constraint(month_number=lambda cell: int(getattr(cell, "point", cell)) in months)
    )
    if seasonal is None:
        available_months = sorted({int(value) for value in cube.coord("month_number").points.tolist()})
        raise ValueError(
            f"No data found for season {season_label!r} (months={months}) in cube "
            f"with sub_experiment_id={cube.attributes.get('sub_experiment_id')}. "
            f"Available months are {available_months}."
        )

    # Keep only complete seasonal years (e.g. drop Nov-Dec-only first chunk for MOHC starts).
    season_years = seasonal.coord("year").points.astype(int)
    season_months = seasonal.coord("month_number").points.astype(int)
    complete_years = [int(year) for year in np.unique(season_years) if set(season_months[season_years == year]) == set(months)]
    seasonal = seasonal.extract(iris.Constraint(year=lambda cell: int(getattr(cell, "point", cell)) in complete_years))
    if seasonal is None:
        raise ValueError(f"No complete {season_label!r} seasons found in cube with sub_experiment_id={cube.attributes.get('sub_experiment_id')}.")

    if varname == "tas":
        seasonal.data = seasonal.data - 273.15
        out = seasonal.aggregated_by("year", iris.analysis.MEAN)
        out.units = "celsius"
        return out, f"mean_{season_label}_temperature"

    if varname == "pr":
        for month in months:
            month_mask = seasonal.coord("month_number").points == month
            seasonal.data[month_mask, ...] = seasonal.data[month_mask, ...] * MONTH_LENGTHS[month]

        out = seasonal.aggregated_by("year", iris.analysis.SUM)
        out.units = "mm"
        return out, f"total_{season_label}_precipitation"

    raise ValueError(f"Unsupported varname: {varname!r}. Use 'tas' or 'pr'.")

def _build_output_cube(season_cubelist: iris.cube.CubeList, long_name: str) -> iris.cube.Cube:
    '''
    Build a single output cube from the list of seasonal cubes, stacking sub_experiment_id, realisation, leadtime, latitude, and longitude.

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

    sub_experiment_ids = sorted({int(c.attributes["sub_experiment_id"]) for c in season_cubelist})
    realisations = sorted({int(c.attributes["realisation"]) for c in season_cubelist})
    leadtimes = np.arange(1, 11)

    sample = season_cubelist[0]
    lats = sample.coord("latitude").points
    lons = sample.coord("longitude").points
    units = sample.units

    data = np.full((len(sub_experiment_ids), len(realisations), len(leadtimes), len(lats), len(lons)),
                   np.nan, dtype=np.float32)

    for cube in season_cubelist:
        sub_exp_id = int(cube.attributes["sub_experiment_id"])
        realisation = int(cube.attributes["realisation"])
        i = sub_experiment_ids.index(sub_exp_id)
        j = realisations.index(realisation)

        if cube.shape[-2:] != (len(lats), len(lons)):
            raise ValueError(f"Grid mismatch across loaded cubes. Expected lat/lon shape {(len(lats), len(lons))}, got {cube.shape[-2:]} for sub_experiment_id={sub_exp_id}, realisation={realisation}.")

        years = cube.coord("year").points.astype(int)
        for t, year in enumerate(years):
            leadtime = int(year) - sub_exp_id
            if leadtime in leadtimes:
                k = leadtime - 1
                data[i, j, k, :, :] = cube.data[t, :, :]

    output = iris.cube.Cube(data, long_name=long_name, units=units,
                            dim_coords_and_dims=[(icoords.DimCoord(sub_experiment_ids, long_name="sub_experiment_id"), 0),
                                                 (icoords.DimCoord(realisations, long_name="realisation"), 1),
                                                 (icoords.DimCoord(leadtimes, long_name="leadtime"), 2),
                                                 (icoords.DimCoord(lats, standard_name="latitude", units="degrees"), 3),
                                                 (icoords.DimCoord(lons, standard_name="longitude", units="degrees"), 4),])

    year_points = (np.array(sub_experiment_ids)[:, None, None]
                   + np.array(leadtimes)[None, None, :]
                   + np.zeros((len(sub_experiment_ids), len(realisations), len(leadtimes)), dtype=int))

    output.add_aux_coord(iris.coords.AuxCoord(year_points, long_name="year"), data_dims=(0, 1, 2))

    return output

def run() -> None:
    '''
    Main function to extract seasonal temperature or precipitation from DePreSys4 for the specified country.
    '''
    region_shape = _load_region_shape(PROVINCE)
    model_cubes = _load_cubes(VARNAME, YEARS, region_shape)
    if not model_cubes:
        raise ValueError("No model cubes were loaded after filtering.")

    _, season_label = _parse_season(SEASON)

    season_cubelist = iris.cube.CubeList()
    output_long_name = None
    skipped = 0
    for cube in model_cubes:
        try:
            season_cube, output_long_name = _season_cube(cube, VARNAME, SEASON)
            season_cubelist.append(season_cube)
        except ValueError as exc:
            if "No data found for season" in str(exc) or "No complete" in str(exc):
                skipped += 1
                continue
            raise

    if skipped:
        print(f"Skipped {skipped} cubes with incomplete or missing requested season.")

    output_cube = _build_output_cube(season_cubelist, output_long_name)

    print(output_cube.coord("year").points)
    print(output_cube.data.max())
    print(output_cube.data.min())

    iris.save(output_cube, f"{OUT_DIR}/{PROVINCE}_{YEARS[0]}_{YEARS[-1]}_{season_label}_{VARNAME}_model_DePreSys4.nc")


if __name__ == "__main__":
    run()
