#!/usr/bin/env python3
"""Standalone fidelity testing for one variable using Iris.

This script:
1. Loads model and observation data with Iris cubes.
2. Applies unit normalization (Kelvin/Celsius for temperature fields).
3. Applies mode-specific regional checks/masking for observations.
4. Builds annual observation series (single file or monthly directory input).
5. Aligns model and observations on common years and runs fidelity testing.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import iris
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from fidelity_test_cube import FidelityTestCube


MONTHLY_FILE_RE = re.compile(r"(?P<year>\d{4})(?P<month>\d{2})(?:\D|$)")
INIT_FILE_RE = re.compile(r"^s(?P<year>\d{4})")
CELSIUS_UNITS = {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}
KELVIN_UNITS = {"k", "kelvin"}
LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")

_DEBUG = False


def _set_debug(enabled: bool) -> None:
    """Enable or disable verbose debug printing for this module."""
    global _DEBUG
    _DEBUG = enabled


def _debug(message: str) -> None:
    """Print a debug message if debug output is enabled."""
    if _DEBUG:
        print(f"[DEBUG] {message}")


def _debug_cube(cube: iris.cube.Cube, label: str) -> None:
    """Print a cube's shape together with the coordinate name for each dimension."""
    if not _DEBUG:
        return
    dim_info = []
    for dim in range(cube.ndim):
        coords = cube.coords(dimensions=(dim,))
        names = ", ".join(coord.name() for coord in coords) if coords else "?"
        dim_info.append(f"dim{dim}={cube.shape[dim]} ({names})")
    scalar_coords = [coord.name() for coord in cube.coords() if len(cube.coord_dims(coord)) == 0]
    scalar_info = f", scalar coords=[{', '.join(scalar_coords)}]" if scalar_coords else ""
    print(f"[DEBUG] {label}: shape={cube.shape}, dims=[{', '.join(dim_info)}]{scalar_info}")


def _debug_array(label: str, array: np.ndarray) -> None:
    """Print an array's shape and dtype."""
    if not _DEBUG:
        return
    print(f"[DEBUG] {label}: shape={np.shape(array)}, dtype={np.asarray(array).dtype}")


def _as_float_array(values: np.ndarray) -> np.ndarray:
    """Return float ndarray with masked values converted to NaN."""
    array = np.ma.asarray(values)
    if np.ma.isMaskedArray(array) and np.any(array.mask):
        array = array.filled(np.nan)
    return np.asarray(array, dtype=float)


def _canonical_unit_name(units: str | None) -> str | None:
    """Normalize unit strings so comparisons are robust."""
    if units is None:
        return None
    return str(units).strip().lower().replace(" ", "_")


def _convert_temperature_units(data: np.ndarray, source_units: str | None, target_units: str | None) -> np.ndarray:
    """Convert temperature arrays between Kelvin and Celsius when needed."""
    source = _canonical_unit_name(source_units)
    target = _canonical_unit_name(target_units)

    if source is None or target is None or source == target:
        return data
    if source in KELVIN_UNITS and target in CELSIUS_UNITS:
        return data - 273.15
    if source in CELSIUS_UNITS and target in KELVIN_UNITS:
        return data + 273.15

    raise ValueError(f"Unsupported temperature unit conversion from {source_units!r} to {target_units!r}")


def _find_coord(cube: iris.cube.Cube, candidate_names: tuple[str, ...]) -> iris.coords.Coord:
    """Return the first matching coord from a list of possible names."""
    for name in candidate_names:
        try:
            return cube.coord(name)
        except Exception:
            continue
    names = ", ".join(candidate_names)
    raise KeyError(f"Could not find any of the coordinates: {names}")


def _load_variable_cube(file_path: Path, variable_name: str) -> iris.cube.Cube:
    """Load a single cube that matches the requested variable name."""
    cubes = iris.load(str(file_path))
    for cube in cubes:
        if cube.var_name == variable_name or cube.name() == variable_name:
            return cube
    raise KeyError(f"Variable {variable_name!r} not found in {file_path}")


def _slice_dim(cube: iris.cube.Cube, dim_index: int, index: int) -> iris.cube.Cube:
    """Slice a cube along one dimension by index."""
    slicer = [slice(None)] * cube.ndim
    slicer[dim_index] = index
    return cube[tuple(slicer)]


def _resolve_leadtime_indices(cube: iris.cube.Cube, single_leadtime_index: int | None) -> list[int]:
    """Return leadtime indices to use.

    If a single leadtime index is provided, only that index is used.
    Otherwise, all leadtimes with at least one finite year value are used.
    """
    _debug_cube(cube, "_resolve_leadtime_indices: loaded cube")
    lead_coord = _find_coord(cube, ("leadtime",))
    lead_dim = cube.coord_dims(lead_coord)[0]
    n_lead = cube.shape[lead_dim]

    if single_leadtime_index is not None:
        if single_leadtime_index < 0 or single_leadtime_index >= n_lead:
            raise ValueError(f"leadtime index {single_leadtime_index} is out of range [0, {n_lead - 1}]")
        _debug(f"_resolve_leadtime_indices: using single leadtime index {single_leadtime_index}")
        return [single_leadtime_index]

    year_coord = cube.coord("year")
    year_points = np.asarray(year_coord.points, dtype=float)
    year_dims = cube.coord_dims(year_coord)
    lead_axis_in_year = year_dims.index(lead_dim)

    valid_indices: list[int] = []
    for lead_index in range(n_lead):
        years_at_lead = np.take(year_points, lead_index, axis=lead_axis_in_year)
        if np.any(np.isfinite(years_at_lead)):
            valid_indices.append(lead_index)

    if not valid_indices:
        raise ValueError("No valid leadtimes were found in the model file")

    _debug(f"_resolve_leadtime_indices: {len(valid_indices)} valid leadtimes found: {valid_indices}")
    return valid_indices


def _extract_model_spatial_info(
    model_path: Path,
    variable_name: str,
    leadtime_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Get model lat/lon arrays, extent, and persistent valid-cell mask.

    Cells that are always zero or always NaN across all years and members are
    treated as invalid, which captures shapefile-masked regions in this dataset.
    """
    cube = _load_variable_cube(model_path, variable_name)
    return _extract_model_spatial_info_from_cube(cube, leadtime_indices)


def _extract_model_spatial_info_from_cube(
    cube: iris.cube.Cube,
    leadtime_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Get model lat/lon arrays, extent, and persistent valid-cell mask from a loaded cube."""
    _debug_cube(cube, "_extract_model_spatial_info: raw model cube")
    lead_coord = _find_coord(cube, ("leadtime",))
    lead_dim = cube.coord_dims(lead_coord)[0]
    data = _as_float_array(cube.data)
    data = np.take(data, leadtime_indices, axis=lead_dim)
    _debug_array("_extract_model_spatial_info: data after leadtime selection", data)

    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    lat_dim = cube.coord_dims(lat_coord)[0]
    lon_dim = cube.coord_dims(lon_coord)[0]

    model_lats = np.asarray(lat_coord.points, dtype=float)
    model_lons = np.asarray(lon_coord.points, dtype=float)

    # Collapse over all non-spatial dimensions (years, members, selected leadtimes)
    # to identify persistently invalid cells from model masking.
    other_dims = tuple(i for i in range(data.ndim) if i not in (lat_dim, lon_dim))
    bad = (~np.isfinite(data)) | (data == 0)
    valid_mask = ~np.all(bad, axis=other_dims)
    _debug(
        f"_extract_model_spatial_info: valid_mask shape={valid_mask.shape}, "
        f"valid cells={int(np.sum(valid_mask))}/{valid_mask.size}"
    )

    extent = (
        float(np.nanmin(model_lats)),
        float(np.nanmax(model_lats)),
        float(np.nanmin(model_lons)),
        float(np.nanmax(model_lons)),
    )
    _debug(f"_extract_model_spatial_info: extent (latmin, latmax, lonmin, lonmax)={extent}")
    return model_lats, model_lons, valid_mask, extent


def _extent_from_mask(
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return spatial extent from a 2D validity mask on the model grid."""
    if not np.any(valid_mask):
        raise ValueError("Region mask removed all model grid cells")
    lat_used = np.where(np.any(valid_mask, axis=1))[0]
    lon_used = np.where(np.any(valid_mask, axis=0))[0]
    return (
        float(np.nanmin(model_lats[lat_used])),
        float(np.nanmax(model_lats[lat_used])),
        float(np.nanmin(model_lons[lon_used])),
        float(np.nanmax(model_lons[lon_used])),
    )


def _extent_within(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
    tol: float = 1e-8,
) -> bool:
    """Return True when inner extent is fully contained within outer extent."""
    return (
        inner[0] >= outer[0] - tol
        and inner[1] <= outer[1] + tol
        and inner[2] >= outer[2] - tol
        and inner[3] <= outer[3] + tol
    )


def _cube_valid_extent(cube: iris.cube.Cube) -> tuple[float, float, float, float]:
    """Return extent of spatial cells that contain at least one finite value."""
    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    lat_dim = cube.coord_dims(lat_coord)[0]
    lon_dim = cube.coord_dims(lon_coord)[0]

    lats = np.asarray(lat_coord.points, dtype=float)
    lons = np.asarray(lon_coord.points, dtype=float)
    data = _as_float_array(cube.data)

    other_dims = tuple(i for i in range(data.ndim) if i not in (lat_dim, lon_dim))
    if other_dims:
        valid_mask = np.any(np.isfinite(data), axis=other_dims)
    else:
        valid_mask = np.isfinite(data)

    if valid_mask.shape != (lats.size, lons.size):
        raise ValueError(
            f"Could not construct spatial validity mask from cube with shape {cube.shape}. "
            f"Expected mask shape {(lats.size, lons.size)}, got {valid_mask.shape}."
        )

    return _extent_from_mask(lats, lons, valid_mask)


def _assert_obs_is_prebounded(obs_cube: iris.cube.Cube, model_extent: tuple[float, float, float, float], obs_label: str) -> None:
    """Fail if supplied observation data are not already bounded/masked to model domain."""
    obs_extent = _cube_valid_extent(obs_cube)
    if not _extent_within(obs_extent, model_extent):
        raise ValueError(
            "Supplied observation file is not already bounded/masked to the model domain. "
            f"Obs valid extent={obs_extent}, model extent={model_extent}. "
            "Provide a pre-bounded observation file or use --obs-dir with --region-shapefile."
        )
    _debug(f"_assert_obs_is_prebounded: {obs_label} valid extent={obs_extent} within model extent={model_extent}")


def _build_obs_region_mask(
    cube: iris.cube.Cube,
    region_bbox: tuple[float, float, float, float] | None,
    region_shape,
    required_coverage_extent: tuple[float, float, float, float] | None,
) -> np.ndarray | None:
    """Build a reusable 2D region mask on an observation cube grid.

    Returns None when no regional masking is requested.
    """
    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    obs_lats = np.asarray(lat_coord.points, dtype=float)
    obs_lons = np.asarray(lon_coord.points, dtype=float)

    def half_cell_tol(points: np.ndarray) -> float:
        if points.size < 2:
            return 1e-8
        diffs = np.abs(np.diff(np.sort(points)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 1e-8
        return float(0.5 * np.nanmedian(diffs))

    if required_coverage_extent is not None:
        lat_tol = half_cell_tol(obs_lats)
        lon_tol = half_cell_tol(obs_lons)
        if np.nanmin(obs_lats) > required_coverage_extent[0] + lat_tol or np.nanmax(obs_lats) < required_coverage_extent[1] - lat_tol:
            raise ValueError("Observation latitude coverage does not include required regional extent")
        if np.nanmin(obs_lons) > required_coverage_extent[2] + lon_tol or np.nanmax(obs_lons) < required_coverage_extent[3] - lon_tol:
            raise ValueError("Observation longitude coverage does not include required regional extent")

    if region_bbox is None and region_shape is None:
        return None

    obs_mask = np.ones((obs_lats.size, obs_lons.size), dtype=bool)
    if region_bbox is not None:
        obs_mask &= _mask_from_bbox(obs_lats, obs_lons, region_bbox)
    if region_shape is not None:
        obs_mask &= _mask_from_shape(cube, region_shape)

    if not np.any(obs_mask):
        raise ValueError("Observation region mask removed all cells")

    return obs_mask


def _subset_year_range(
    years: np.ndarray,
    data: np.ndarray,
    start_year: int | None,
    end_year: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Subset a year-indexed array to an inclusive [start_year, end_year] range."""
    years = np.asarray(years, dtype=int)
    keep = np.ones(years.shape, dtype=bool)
    if start_year is not None:
        keep &= years >= int(start_year)
    if end_year is not None:
        keep &= years <= int(end_year)
    return years[keep], np.asarray(data)[keep]


def _parse_init_year_from_filename(model_file: Path) -> int | None:
    """Extract init year from raw model filename like s1960.nc."""
    match = INIT_FILE_RE.search(model_file.stem)
    if match is None:
        return None
    return int(match.group("year"))


def _region_shape_bounds(region_shape) -> tuple[float, float, float, float]:
    """Return (lat_min, lat_max, lon_min, lon_max) bounds from a shape."""

    def unpack_bounds(bounds):
        if bounds is None:
            return None
        if callable(bounds):
            bounds = bounds()
        values = np.asarray(bounds).squeeze()
        if values.size != 4:
            return None
        min_lon, min_lat, max_lon, max_lat = [float(v) for v in values]
        return float(min_lat), float(max_lat), float(min_lon), float(max_lon)

    for candidate in [
        getattr(region_shape, "total_bounds", None),
        getattr(region_shape, "bounds", None),
        getattr(getattr(region_shape, "data", None), "bounds", None),
        getattr(getattr(region_shape, "geometry", None), "bounds", None),
    ]:
        result = unpack_bounds(candidate)
        if result is not None:
            return result

    raise ValueError("Unable to infer bounds from region shapefile geometry")


def _intersect_bboxes(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Return intersection of two bboxes (lat_min, lat_max, lon_min, lon_max)."""
    lat_min = max(bbox_a[0], bbox_b[0])
    lat_max = min(bbox_a[1], bbox_b[1])
    lon_min = max(bbox_a[2], bbox_b[2])
    lon_max = min(bbox_a[3], bbox_b[3])
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("Requested regional bounds do not overlap")
    return (lat_min, lat_max, lon_min, lon_max)


def _subset_cube_to_bbox(
    cube: iris.cube.Cube,
    bbox: tuple[float, float, float, float],
) -> iris.cube.Cube:
    """Subset a cube spatially to an inclusive lat/lon bbox."""
    lat_min, lat_max, lon_min, lon_max = bbox
    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    lat_name = lat_coord.name()
    lon_name = lon_coord.name()

    lat_constraint = iris.Constraint(**{lat_name: lambda cell: lat_min <= cell <= lat_max})
    lon_constraint = iris.Constraint(**{lon_name: lambda cell: lon_min <= cell <= lon_max})
    sub = cube.extract(lat_constraint & lon_constraint)
    if sub is None:
        raise ValueError(f"Spatial subset removed all cells for bbox {bbox}")
    return sub


def _mask_from_bbox(model_lats: np.ndarray, model_lons: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Create a 2D mask from an inclusive lat/lon bounding box."""
    lat_min, lat_max, lon_min, lon_max = bbox
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError("Bounding box must be provided as lat_min lat_max lon_min lon_max")

    lat_keep = (model_lats >= lat_min) & (model_lats <= lat_max)
    lon_keep = (model_lons >= lon_min) & (model_lons <= lon_max)
    return np.outer(lat_keep, lon_keep)


def _load_region_shape(shapefile_path: Path):
    """Load and merge all geometries from a shapefile into one Ascend shape."""
    try:
        from ascend import shape as ashape
    except Exception as exc:
        raise ImportError("Shapefile masking requires the Ascend package") from exc

    shapes = ashape.load_shp(str(shapefile_path))
    if len(shapes) == 0:
        raise ValueError(f"No geometries found in shapefile: {shapefile_path}")

    merged = shapes[0]
    for shp in shapes[1:]:
        merged = merged.union(shp)

    return merged


def _mask_from_shape(cube: iris.cube.Cube, region_shape) -> np.ndarray:
    """Create a 2D boolean mask from an Ascend shape on a cube grid."""
    weights_cube = region_shape.cube_2d_weights(cube, intersection=True)
    weights = _as_float_array(weights_cube.data)
    return weights > 0


def _extract_raw_model_spatial_info(
    sample_model_file: Path,
    variable_name: str,
    region_bbox: tuple[float, float, float, float] | None,
    region_shape,
    spatial_clip_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Get model lat/lon arrays and valid mask for raw DePreSys4 init-year files."""
    cube = _load_variable_cube(sample_model_file, variable_name)
    if spatial_clip_bbox is not None:
        cube = _subset_cube_to_bbox(cube, spatial_clip_bbox)
    _debug_cube(cube, "_extract_raw_model_spatial_info: sample raw model cube")

    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    model_lats = np.asarray(lat_coord.points, dtype=float)
    model_lons = np.asarray(lon_coord.points, dtype=float)

    valid_mask = np.ones((model_lats.size, model_lons.size), dtype=bool)
    if region_bbox is not None:
        valid_mask &= _mask_from_bbox(model_lats, model_lons, region_bbox)
    if region_shape is not None:
        valid_mask &= _mask_from_shape(cube, region_shape)

    extent = _extent_from_mask(model_lats, model_lons, valid_mask)
    _debug(
        f"_extract_raw_model_spatial_info: valid_mask shape={valid_mask.shape}, "
        f"valid cells={int(np.sum(valid_mask))}/{valid_mask.size}, extent={extent}"
    )
    return model_lats, model_lons, valid_mask, extent


def _load_model_series_from_raw_directory(
    model_dir: Path,
    variable_name: str,
    obs_months: tuple[int, ...] | None,
    model_valid_mask: np.ndarray,
    start_year: int | None,
    end_year: int | None,
    spatial_clip_bbox: tuple[float, float, float, float] | None = None,
    pattern: str = "s*.nc",
) -> tuple[np.ndarray, np.ndarray, str | None, int, int]:
    """Load raw init-year model files and aggregate to yearly pooled simulations.

    Returns:
      years_kept: 1D array of target calendar years.
      model_data: 2D array [year, pooled_simulation].
      model_units: canonical units string.
      pooled_rows: number of pooled (init-file, target-year) rows before collapse.
      full_pool_size: number of simulations retained for each kept year.
    """
    files = sorted(model_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model files found in {model_dir} matching {pattern!r}")

    # Filter init files up front using requested target years and the file's lead-year span.
    # This avoids loading many raw init files that cannot contribute to [start_year, end_year].
    if start_year is not None or end_year is not None:
        sample_cube = _load_variable_cube(files[0], variable_name)
        time_coord = _find_coord(sample_cube, ("time",))
        sample_times = time_coord.units.num2date(time_coord.points)
        sample_years = np.array([int(dt.year) for dt in sample_times], dtype=int)

        sample_init_year = _parse_init_year_from_filename(files[0])
        if sample_init_year is not None:
            lead_offsets = sample_years - sample_init_year
            min_offset = int(np.min(lead_offsets))
            max_offset = int(np.max(lead_offsets))

            min_init_needed = None if end_year is None else int(end_year) - min_offset
            max_init_needed = None if start_year is None else int(start_year) - max_offset

            filtered_files = []
            for model_file in files:
                init_year = _parse_init_year_from_filename(model_file)
                if init_year is None:
                    filtered_files.append(model_file)
                    continue
                if max_init_needed is not None and init_year < max_init_needed:
                    continue
                if min_init_needed is not None and init_year > min_init_needed:
                    continue
                filtered_files.append(model_file)

            _debug(
                "_load_model_series_from_raw_directory: "
                f"prefiltered raw model files {len(files)} -> {len(filtered_files)} "
                f"using offsets [{min_offset}, {max_offset}] and target years "
                f"[{start_year}, {end_year}]"
            )
            files = filtered_files

    if not files:
        raise ValueError(
            "No raw model files remain after filtering by requested start/end years"
        )

    month_filter = set(obs_months) if obs_months else None
    pooled_by_year: dict[int, list[float]] = defaultdict(list)
    pooled_rows = 0
    model_units: str | None = None

    for model_file in files:
        cube = _load_variable_cube(model_file, variable_name)
        if spatial_clip_bbox is not None:
            cube = _subset_cube_to_bbox(cube, spatial_clip_bbox)
        source_units = _canonical_unit_name(str(cube.units))
        if model_units is None:
            if source_units in CELSIUS_UNITS or source_units in KELVIN_UNITS:
                model_units = "celsius"
            else:
                model_units = source_units

        time_coord = _find_coord(cube, ("time",))
        real_coord = _find_coord(cube, ("realisation", "realization", "realization_number", "realization"))
        lat_coord = _find_coord(cube, LAT_NAMES)
        lon_coord = _find_coord(cube, LON_NAMES)

        time_dim = cube.coord_dims(time_coord)[0]
        real_dim = cube.coord_dims(real_coord)[0]
        lat_dim = cube.coord_dims(lat_coord)[0]
        lon_dim = cube.coord_dims(lon_coord)[0]

        data = _as_float_array(cube.data)
        data = _convert_temperature_units(data, str(cube.units), model_units)
        data = np.moveaxis(data, (real_dim, time_dim, lat_dim, lon_dim), (0, 1, 2, 3))
        data = np.where(model_valid_mask[None, None, :, :], data, np.nan)
        data = np.nanmean(data, axis=(2, 3))

        datetimes = time_coord.units.num2date(time_coord.points)
        years = np.array([int(dt.year) for dt in datetimes], dtype=int)
        months = np.array([int(dt.month) for dt in datetimes], dtype=int)

        target_years = np.unique(years)
        for year in target_years:
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue

            time_idx = np.where(years == year)[0]
            if month_filter is not None:
                present = set(int(m) for m in months[time_idx])
                if not month_filter.issubset(present):
                    continue
                time_idx = np.array([idx for idx in time_idx if int(months[idx]) in month_filter], dtype=int)

            if time_idx.size == 0:
                continue

            values = np.nanmean(data[:, time_idx], axis=1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue

            pooled_by_year[int(year)].extend(values.tolist())
            pooled_rows += 1

    if not pooled_by_year:
        raise ValueError("No model yearly values were produced from raw model files")

    years_sorted = np.array(sorted(pooled_by_year), dtype=int)
    pool_sizes = np.array([len(pooled_by_year[int(y)]) for y in years_sorted], dtype=int)
    full_pool_size = int(pool_sizes.max())
    keep = pool_sizes == full_pool_size

    dropped = years_sorted[~keep]
    if dropped.size:
        if dropped.size <= 8:
            dropped_text = ", ".join(str(int(y)) for y in dropped)
        else:
            dropped_text = (
                ", ".join(str(int(y)) for y in dropped[:4])
                + ", ..., "
                + ", ".join(str(int(y)) for y in dropped[-4:])
            )
        print(
            f"Dropping {dropped.size} model years without full simulation coverage "
            f"({full_pool_size} simulations/year required): {dropped_text}"
        )

    years_kept = years_sorted[keep]
    model_data = np.vstack([np.asarray(pooled_by_year[int(y)], dtype=float) for y in years_kept])
    _debug_array("_load_model_series_from_raw_directory: years_kept", years_kept)
    _debug_array("_load_model_series_from_raw_directory: model_data", model_data)
    return years_kept, model_data, model_units, pooled_rows, full_pool_size


def _load_model_series(
    model_path: Path,
    variable_name: str,
    leadtime_indices: list[int],
    model_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load model data as (simulated_year_row, member) across selected leadtimes."""
    cube = _load_variable_cube(model_path, variable_name)
    return _load_model_series_from_cube(cube, leadtime_indices, model_valid_mask, source_label=str(model_path))


def _load_model_series_from_cube(
    cube: iris.cube.Cube,
    leadtime_indices: list[int],
    model_valid_mask: np.ndarray,
    source_label: str = "model cube",
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load model data as (simulated_year_row, member) across selected leadtimes from loaded cube."""
    _debug_cube(cube, "_load_model_series: raw model cube")
    source_units = _canonical_unit_name(str(cube.units))
    if source_units in CELSIUS_UNITS or source_units in KELVIN_UNITS:
        model_units = "celsius"
    else:
        model_units = source_units

    lead_coord = _find_coord(cube, ("leadtime",))
    lead_dim = cube.coord_dims(lead_coord)[0]
    data = _as_float_array(cube.data)
    data = _convert_temperature_units(data, str(cube.units), model_units)
    data = np.take(data, leadtime_indices, axis=lead_dim)
    _debug_array("_load_model_series: data after leadtime selection", data)

    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    lat_dim = cube.coord_dims(lat_coord)[0]
    lon_dim = cube.coord_dims(lon_coord)[0]

    if data.shape[lat_dim] != model_valid_mask.shape[0] or data.shape[lon_dim] != model_valid_mask.shape[1]:
        raise ValueError(
            f"Model mask shape {model_valid_mask.shape} does not match model grid {(data.shape[lat_dim], data.shape[lon_dim])}"
        )

    # Broadcast 2D valid mask over non-spatial dimensions.
    expanded_mask = model_valid_mask
    while expanded_mask.ndim < data.ndim:
        expanded_mask = np.expand_dims(expanded_mask, axis=0)
    expanded_mask = np.broadcast_to(expanded_mask, data.shape)
    data = np.where(expanded_mask, data, np.nan)

    data = np.nanmean(data, axis=(lat_dim, lon_dim))
    _debug_array("_load_model_series: data after masking + area-mean (lat/lon collapsed)", data)

    try:
        year_coord = cube.coord("year")
    except Exception as exc:
        raise KeyError(f"Model file {source_label} does not contain a usable year coordinate") from exc

    # Reconstruct the (init_year, leadtime, member) structure explicitly
    # so each leadtime contributes additional simulated years.
    real_coord = _find_coord(cube, ("realisation", "realization"))
    real_dim = cube.coord_dims(real_coord)[0]
    year_dims = cube.coord_dims(year_coord)
    lead_axis_in_year = year_dims.index(lead_dim)
    real_axis_in_year = year_dims.index(real_dim)
    init_dim_candidates = [d for d in year_dims if d not in (lead_dim, real_dim)]
    if len(init_dim_candidates) != 1:
        raise ValueError(f"Unable to identify initialisation dimension from year dims {year_dims}")
    init_dim = init_dim_candidates[0]
    init_axis_in_year = year_dims.index(init_dim)
    _debug(
        f"_load_model_series: year coord dims={year_dims} "
        f"(lead_dim={lead_dim}, real_dim={real_dim}, init_dim={init_dim})"
    )

    year_points = np.asarray(year_coord.points, dtype=float)
    year_points = np.take(year_points, leadtime_indices, axis=lead_axis_in_year)

    # Validate that years do not depend on ensemble member.
    years_first_member = np.take(year_points, 0, axis=real_axis_in_year)
    expanded_first = np.expand_dims(years_first_member, axis=real_axis_in_year)
    if not np.all(year_points == expanded_first):
        raise ValueError("Year coordinate varies by ensemble member; cannot build a unique year x lead mapping")

    remaining_year_axes = [ax for ax in range(year_points.ndim) if ax != real_axis_in_year]
    init_pos = remaining_year_axes.index(init_axis_in_year)
    lead_pos = remaining_year_axes.index(lead_axis_in_year)
    years_2d = np.moveaxis(years_first_member, (init_pos, lead_pos), (0, 1))
    _debug_array("_load_model_series: years_2d (init_year x leadtime)", years_2d)

    remaining_data_dims = [d for d in range(cube.ndim) if d not in (lat_dim, lon_dim)]
    init_pos_data = remaining_data_dims.index(init_dim)
    lead_pos_data = remaining_data_dims.index(lead_dim)
    real_pos_data = remaining_data_dims.index(real_dim)
    data = np.moveaxis(data, (init_pos_data, lead_pos_data, real_pos_data), (0, 1, 2))
    _debug_array("_load_model_series: data moved to (init_year, leadtime, member)", data)

    n_init, n_lead, n_member = data.shape
    model_data = data.reshape(n_init * n_lead, n_member)
    model_years = years_2d.reshape(n_init * n_lead)
    _debug(
        f"_load_model_series: reshaped to (init x lead)={n_init * n_lead} rows, "
        f"n_member={n_member} (n_init={n_init}, n_lead={n_lead})"
    )

    year_valid = np.isfinite(model_years)
    model_years = model_years[year_valid].astype(int)
    model_data = model_data[year_valid]
    _debug_array("_load_model_series: model_years after dropping non-finite years", model_years)
    _debug_array("_load_model_series: model_data after dropping non-finite years", model_data)

    if model_data.ndim != 2:
        raise ValueError(f"Expected model data to be 2D (rows x members), got shape {model_data.shape}")

    return model_years, model_data, model_units


def _collapse_pooled_years_to_unique(
    model_years: np.ndarray,
    model_data: np.ndarray,
    n_samples_per_year: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse leadtime-stacked rows and keep only fully covered calendar years.

    When multiple leadtimes are pooled, the same calendar year can appear many
    times (once per leadtime that reaches it), with very uneven counts across
    the record (edge years appear once, mid-record years many times). Passing
    the raw pooled rows into FidelityTestCube would (a) implicitly reweight the
    paired observation value toward well-covered years when aligned, and (b)
    inflate the number of "years" (nyrs) used in its bootstrap resampling,
    which artificially narrows the reference distribution and can make small,
    genuine model-obs differences look like extreme (0%/100%) failures.

    This groups rows by unique calendar year and pools every (leadtime x
    member) value available for that year (e.g. up to 10 leadtimes x 10
    members = ~100 simulated values for well-covered, mid-record years).
    Years near the edges of the record are reachable by fewer leadtimes and
    so have a smaller real pool (as few as one leadtime's worth of members).

    By default the fixed row width is the largest pool size found across all
    years (typically n_leadtimes x n_members, e.g. 100). Years with fewer
    simulations are dropped so each retained year has complete, like-for-like
    simulation support.
    """
    model_years = np.asarray(model_years, dtype=int)
    model_data = np.asarray(model_data, dtype=float)
    unique_years = np.unique(model_years)
    _debug(
        f"_collapse_pooled_years_to_unique: input model_data shape={model_data.shape}, "
        f"{unique_years.size} unique years ({unique_years.min()} to {unique_years.max()})"
    )

    pools = [model_data[model_years == year].ravel() for year in unique_years]
    pools = [pool[np.isfinite(pool)] for pool in pools]

    if n_samples_per_year is None:
        n_samples_per_year = max((pool.size for pool in pools), default=model_data.shape[1])

    pool_sizes = np.array([pool.size for pool in pools], dtype=int)
    keep = pool_sizes == n_samples_per_year
    dropped = int(np.sum(~keep))
    if dropped:
        dropped_years = unique_years[~keep]
        if dropped_years.size <= 8:
            dropped_years_text = ", ".join(str(int(y)) for y in dropped_years)
        else:
            head = ", ".join(str(int(y)) for y in dropped_years[:4])
            tail = ", ".join(str(int(y)) for y in dropped_years[-4:])
            dropped_years_text = f"{head}, ..., {tail}"
        print(
            f"Dropping {dropped} model years without full simulation coverage "
            f"({n_samples_per_year} simulations/year required): "
            f"{dropped_years_text}"
        )

    if _DEBUG:
        print(
            f"[DEBUG] _collapse_pooled_years_to_unique: per-year pool sizes min={pool_sizes.min()}, "
            f"max={pool_sizes.max()}, required/full size={n_samples_per_year}, "
            f"retained years={int(np.sum(keep))}, dropped years={dropped}"
        )

    kept_years = unique_years[keep]
    kept_pools = [pool for pool, is_kept in zip(pools, keep) if is_kept]

    collapsed = np.empty((kept_years.size, n_samples_per_year), dtype=float)
    for i, pool in enumerate(kept_pools):
        # Every retained year has full pool size by construction.
        collapsed[i, :] = pool

    _debug_array("_collapse_pooled_years_to_unique: collapsed output", collapsed)
    return kept_years, collapsed


def _obs_field_to_masked_mean(
    cube: iris.cube.Cube,
    target_units: str | None,
    obs_region_mask: np.ndarray | None,
) -> float:
    """Apply precomputed regional mask on obs grid and area-average."""
    data = _as_float_array(cube.data)
    data = _convert_temperature_units(data, str(cube.units), target_units)

    if obs_region_mask is None:
        return float(np.nanmean(data))

    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    expected_shape = (len(lat_coord.points), len(lon_coord.points))
    if obs_region_mask.shape != expected_shape:
        raise ValueError(
            f"Observation region mask shape {obs_region_mask.shape} does not match cube grid {expected_shape}"
        )

    expanded_mask = obs_region_mask
    while expanded_mask.ndim < data.ndim:
        expanded_mask = np.expand_dims(expanded_mask, axis=0)
    expanded_mask = np.broadcast_to(expanded_mask, data.shape)
    data = np.where(expanded_mask, data, np.nan)
    return float(np.nanmean(data))


def _load_observation_series(
    obs_path: Path,
    variable_name: str,
    obs_months: tuple[int, ...] | None,
    start_year: int | None,
    end_year: int | None,
    target_units: str | None,
    region_bbox: tuple[float, float, float, float] | None,
    region_shape,
    required_coverage_extent: tuple[float, float, float, float] | None,
    preloaded_cube: iris.cube.Cube | None = None,
    spatial_clip_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one observation file and build yearly series.

    If the file has a time coordinate, selected months are grouped to yearly means.
    If the file has a year coordinate, values are used directly.
    """
    cube = preloaded_cube if preloaded_cube is not None else _load_variable_cube(obs_path, variable_name)
    if spatial_clip_bbox is not None:
        cube = _subset_cube_to_bbox(cube, spatial_clip_bbox)
    _debug_cube(cube, "_load_observation_series: raw obs cube")
    obs_region_mask = _build_obs_region_mask(cube, region_bbox, region_shape, required_coverage_extent)

    try:
        year_coord = cube.coord("year")
        years = np.asarray(year_coord.points, dtype=int).squeeze()
        values = []

        # Use a year-by-year extraction when year is 1D; otherwise reduce full field.
        if years.ndim == 1 and years.size > 1:
            year_dim = cube.coord_dims(year_coord)[0]
            for i in range(years.size):
                year_i = int(years[i])
                if start_year is not None and year_i < start_year:
                    continue
                if end_year is not None and year_i > end_year:
                    continue
                sub_cube = _slice_dim(cube, year_dim, i)
                values.append(
                    _obs_field_to_masked_mean(
                        sub_cube, target_units, obs_region_mask
                    )
                )
            years = np.array(
                [int(y) for y in years if (start_year is None or int(y) >= start_year) and (end_year is None or int(y) <= end_year)],
                dtype=int,
            )
        else:
            year_value = int(np.asarray(years).reshape(-1)[0])
            if (start_year is not None and year_value < start_year) or (end_year is not None and year_value > end_year):
                return np.array([], dtype=int), np.array([], dtype=float)
            values.append(
                _obs_field_to_masked_mean(
                    cube, target_units, obs_region_mask
                )
            )
            years = np.array([year_value], dtype=int)

        return years.astype(int), np.asarray(values, dtype=float)
    except Exception:
        pass

    try:
        time_coord = cube.coord("time")
    except Exception as exc:
        raise KeyError(
            f"Observation file {obs_path} must contain either a year or time coordinate"
        ) from exc

    # If time is scalar but filename embeds YYYYMM, use filename as fallback.
    time_points = np.asarray(time_coord.points)
    if time_points.size == 1:
        match = MONTHLY_FILE_RE.search(obs_path.stem)
        if match:
            year = int(match.group("year"))
            month = int(match.group("month"))
            if obs_months and month not in set(obs_months):
                return np.array([], dtype=int), np.array([], dtype=float)
            if (start_year is not None and year < start_year) or (end_year is not None and year > end_year):
                return np.array([], dtype=int), np.array([], dtype=float)
            value = _obs_field_to_masked_mean(
                cube, target_units, obs_region_mask
            )
            return np.array([year], dtype=int), np.array([value], dtype=float)

    # General time-series path.
    datetimes = time_coord.units.num2date(time_coord.points)
    month_filter = set(obs_months) if obs_months else None
    grouped: dict[int, list[float]] = {}
    months_seen: dict[int, set[int]] = {}

    time_dim = cube.coord_dims(time_coord)[0]
    for i, dt in enumerate(datetimes):
        if start_year is not None and int(dt.year) < start_year:
            continue
        if end_year is not None and int(dt.year) > end_year:
            continue
        if month_filter is not None and int(dt.month) not in month_filter:
            continue
        sub_cube = _slice_dim(cube, time_dim, i)
        value = _obs_field_to_masked_mean(
            sub_cube, target_units, obs_region_mask
        )
        grouped.setdefault(int(dt.year), []).append(value)
        months_seen.setdefault(int(dt.year), set()).add(int(dt.month))

    if month_filter is not None:
        for year in list(grouped):
            missing = sorted(month_filter - months_seen.get(year, set()))
            if missing:
                print(f"Dropping incomplete observation year {year}: missing months {missing}")
                del grouped[year]

    years = np.array(sorted(grouped), dtype=int)
    series = np.array([np.nanmean(grouped[year]) for year in years], dtype=float)
    _debug_array("_load_observation_series: output years", years)
    _debug_array("_load_observation_series: output series", series)
    return years, series


def _load_observation_directory(
    obs_dir: Path,
    variable_name: str,
    obs_months: tuple[int, ...] | None,
    start_year: int | None,
    end_year: int | None,
    target_units: str | None,
    region_bbox: tuple[float, float, float, float] | None,
    region_shape,
    required_coverage_extent: tuple[float, float, float, float] | None,
    spatial_clip_bbox: tuple[float, float, float, float] | None = None,
    pattern: str = "*.nc",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one-file-per-month observations and aggregate to annual means."""
    files = sorted(obs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No observation files found in {obs_dir} matching {pattern!r}")
    _debug(f"_load_observation_directory: found {len(files)} files matching {pattern!r}")

    month_filter = set(obs_months) if obs_months else None
    selected: list[tuple[Path, int, int]] = []
    for obs_file in files:
        match = MONTHLY_FILE_RE.search(obs_file.stem)
        if match is None:
            raise ValueError(
                f"Could not infer year/month from observation filename {obs_file.name}. Expected YYYYMM token."
            )

        year = int(match.group("year"))
        month = int(match.group("month"))
        if month_filter is not None and month not in month_filter:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append((obs_file, year, month))

    if not selected:
        raise ValueError(
            f"No observation files in {obs_dir} matched requested years/months "
            f"(start_year={start_year}, end_year={end_year}, obs_months={obs_months})"
        )

    _debug(f"_load_observation_directory: selected {len(selected)} files after year/month filtering")
    first_cube = _load_variable_cube(selected[0][0], variable_name)
    if spatial_clip_bbox is not None:
        first_cube = _subset_cube_to_bbox(first_cube, spatial_clip_bbox)
    if _DEBUG:
        _debug_cube(first_cube, f"_load_observation_directory: first file cube ({selected[0][0].name})")
    obs_region_mask = _build_obs_region_mask(first_cube, region_bbox, region_shape, required_coverage_extent)
    grouped: dict[int, list[float]] = {}
    months_seen: dict[int, set[int]] = {}

    for i, (obs_file, year, month) in enumerate(selected):
        cube = first_cube if i == 0 else _load_variable_cube(obs_file, variable_name)
        if i > 0 and spatial_clip_bbox is not None:
            cube = _subset_cube_to_bbox(cube, spatial_clip_bbox)
        value = _obs_field_to_masked_mean(
            cube,
            target_units,
            obs_region_mask,
        )
        grouped.setdefault(year, []).append(value)
        months_seen.setdefault(year, set()).add(month)

    if month_filter is not None:
        for year in list(grouped):
            missing = sorted(month_filter - months_seen.get(year, set()))
            if missing:
                print(f"Dropping incomplete observation year {year}: missing months {missing}")
                del grouped[year]

    if not grouped:
        raise ValueError(f"No observation files in {obs_dir} matched the requested months {obs_months}")

    years = np.array(sorted(grouped), dtype=int)
    series = np.array([np.nanmean(grouped[year]) for year in years], dtype=float)
    _debug_array("_load_observation_directory: output years", years)
    _debug_array("_load_observation_directory: output series", series)
    return years, series


def _align_on_years(
    model_years: np.ndarray,
    model_data: np.ndarray,
    obs_years: np.ndarray,
    obs_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align model and observations on overlapping years and finite values.

    Model years may repeat when multiple leadtimes are used. This function keeps
    all matching model rows and maps each to the corresponding observation year.
    """
    obs_years = np.asarray(obs_years, dtype=int)
    obs_data = np.asarray(obs_data, dtype=float)
    model_years = np.asarray(model_years, dtype=int)
    _debug(
        f"_align_on_years: model_data shape={model_data.shape} ({np.unique(model_years).size} unique years), "
        f"obs_data shape={obs_data.shape} ({np.unique(obs_years).size} unique years)"
    )

    if obs_years.size == 0 or model_years.size == 0:
        raise ValueError("Model or observation years are empty")

    unique_obs_years = np.unique(obs_years)
    obs_by_year = {int(year): float(np.nanmean(obs_data[obs_years == year])) for year in unique_obs_years}

    keep_model = np.isin(model_years, unique_obs_years)
    if not np.any(keep_model):
        raise ValueError("Model and observation series do not share any common years")

    model_aligned = model_data[keep_model]
    common_years = model_years[keep_model]
    obs_aligned = np.array([obs_by_year[int(year)] for year in common_years], dtype=float)

    valid = np.isfinite(obs_aligned) & np.all(np.isfinite(model_aligned), axis=1)
    common_years = common_years[valid]
    model_aligned = model_aligned[valid]
    obs_aligned = obs_aligned[valid]

    if common_years.size == 0:
        raise ValueError("All overlapping years were removed because of missing values")

    _debug(
        f"_align_on_years: aligned model_data shape={model_aligned.shape}, obs_data shape={obs_aligned.shape}, "
        f"{np.unique(common_years).size} unique common years"
    )
    return common_years, model_aligned, obs_aligned


def _write_prepared_observations(output_path: Path, years: np.ndarray, values: np.ndarray) -> None:
    """Write the prepared annual observation series to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", "value"])
        for year, value in zip(years, values):
            writer.writerow([int(year), float(value)])


def _write_prepared_model_csv(output_path: Path, years: np.ndarray, values_2d: np.ndarray) -> None:
    """Write prepared model yearly matrix to CSV (year + simulation columns)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_members = values_2d.shape[1]
    header = ["year"] + [f"sim_{i+1}" for i in range(n_members)]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for year, row in zip(years, values_2d):
            writer.writerow([int(year), *[float(v) for v in row]])


def _write_prepared_observations_netcdf(output_path: Path, years: np.ndarray, values: np.ndarray, var_name: str) -> None:
    """Write prepared annual observation series to NetCDF using Iris."""
    year_coord = iris.coords.DimCoord(np.asarray(years, dtype=int), long_name="year", units="1")
    cube = iris.cube.Cube(np.asarray(values, dtype=float), long_name=var_name, dim_coords_and_dims=[(year_coord, 0)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cube, str(output_path))


def _write_prepared_model_netcdf(
    output_path: Path,
    years: np.ndarray,
    values_2d: np.ndarray,
    var_name: str,
) -> None:
    """Write prepared annual model matrix to NetCDF using Iris."""
    year_coord = iris.coords.DimCoord(np.asarray(years, dtype=int), long_name="year", units="1")
    member_coord = iris.coords.DimCoord(np.arange(values_2d.shape[1], dtype=int), long_name="simulation", units="1")
    cube = iris.cube.Cube(
        np.asarray(values_2d, dtype=float),
        long_name=var_name,
        dim_coords_and_dims=[(year_coord, 0), (member_coord, 1)],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cube, str(output_path))


def _overall_distribution_stats(values: np.ndarray) -> dict[str, float]:
    """Return summary statistics from all finite values in an array."""
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise ValueError("No finite model values available to summarize")

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p05": float(np.percentile(flat, 5)),
        "p95": float(np.percentile(flat, 95)),
    }


def _obs_dataset_label(obs_file: Path | None, obs_dir: Path | None) -> str:
    """Return a concise label for the observation dataset source."""
    if obs_file is not None:
        return obs_file.stem
    if obs_dir is not None:
        # Use the final three path segments when possible so the dataset family is retained
        # (e.g. OBS-ERA5/monthly/2m_temperature).
        parts = obs_dir.parts
        if len(parts) >= 3:
            return "/".join(parts[-3:])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return obs_dir.name
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="Run fidelity testing for one variable")
    model_input = parser.add_mutually_exclusive_group(required=False)
    model_input.add_argument("--model-file", type=Path, help="Path to prepared model netCDF file")
    model_input.add_argument("--model-dir", type=Path, help="Directory of raw model init files (e.g. .../DePreSys4/monthly/air_temperature)")
    parser.add_argument("--model-root", type=Path, default=None, help="Optional model root directory (e.g. /data/users/appldata/Data/DePreSys4)")
    parser.add_argument("--model-subdir", default=None, help="Model subdirectory under model root (e.g. monthly/air_temperature)")
    parser.add_argument("--model-pattern", default="s*.nc", help="Glob pattern for raw model files")
    parser.add_argument("--model-var", default="mean_jja_temperature", help="Model variable name")

    obs_input = parser.add_mutually_exclusive_group(required=False)
    obs_input.add_argument("--obs-file", type=Path, help="Single observation netCDF file")
    obs_input.add_argument("--obs-dir", type=Path, help="Directory of monthly observation netCDF files")
    parser.add_argument("--obs-root", type=Path, default=None, help="Optional obs root directory (e.g. /data/users/appldata/Data/OBS-ERA5)")
    parser.add_argument("--obs-subdir", default=None, help="Obs subdirectory under obs root (e.g. monthly/2m_temperature)")

    parser.add_argument("--obs-var", default="t2m", help="Observation variable name")
    parser.add_argument("--start-year", type=int, default=None, help="Inclusive start year for both model and observations")
    parser.add_argument("--end-year", type=int, default=None, help="Inclusive end year for both model and observations")
    parser.add_argument(
        "--region-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"),
        help="Regional bounding box",
    )
    parser.add_argument(
        "--region-shapefile",
        type=Path,
        default=None,
        help="Shapefile path for regional masking",
    )
    parser.add_argument(
        "--leadtime-index",
        type=int,
        default=None,
        help="Optional single leadtime index (prepared model-file mode only)",
    )
    parser.add_argument(
        "--obs-months",
        type=int,
        nargs="*",
        default=[6, 7, 8],
        help="Months to average for observations (empty list disables month filtering)",
    )
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fidelity sampling")
    parser.add_argument("--label", default="temperature", help="Output filename label")
    parser.add_argument("--obs-pattern", default="*.nc", help="Glob pattern for monthly obs files")
    parser.add_argument("--save-prepared-obs", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--save-prepared-model", type=Path, default=None, help="Optional CSV output path for model yearly matrix")
    parser.add_argument("--save-prepared-obs-nc", type=Path, default=None, help="Optional NetCDF output path for prepared observations")
    parser.add_argument("--save-prepared-model-nc", type=Path, default=None, help="Optional NetCDF output path for prepared model")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print cube/array shapes and coordinate names at each processing step",
    )
    return parser


def main() -> None:
    """Execute end-to-end one-variable fidelity workflow."""
    args = build_parser().parse_args()
    _set_debug(args.debug)

    if args.model_dir is None and args.model_file is None:
        if args.model_root is not None and args.model_subdir:
            args.model_dir = args.model_root / args.model_subdir
        else:
            raise ValueError("Provide either --model-file, --model-dir, or (--model-root and --model-subdir)")

    if args.obs_dir is None and args.obs_file is None:
        if args.obs_root is not None and args.obs_subdir:
            args.obs_dir = args.obs_root / args.obs_subdir
        else:
            raise ValueError("Provide either --obs-file, --obs-dir, or (--obs-root and --obs-subdir)")

    obs_dataset = _obs_dataset_label(args.obs_file, args.obs_dir)
    obs_months = tuple(args.obs_months) if args.obs_months else None
    region_bbox = tuple(args.region_bbox) if args.region_bbox is not None else None
    region_shape = _load_region_shape(args.region_shapefile) if args.region_shapefile is not None else None
    obs_required_coverage_extent: tuple[float, float, float, float] | None = None
    obs_region_bbox: tuple[float, float, float, float] | None = None
    obs_region_shape = None
    obs_spatial_clip_bbox: tuple[float, float, float, float] | None = None

    if args.model_file is not None:
        model_mode = "prepared-model-file"
        model_cube = _load_variable_cube(args.model_file, args.model_var)
        leadtime_indices = _resolve_leadtime_indices(model_cube, args.leadtime_index)

        # Derive model grid and mask once, then reuse for both model and observations.
        model_lats, model_lons, model_valid_mask, model_extent = _extract_model_spatial_info_from_cube(
            model_cube,
            leadtime_indices,
        )
        model_extent = _extent_from_mask(model_lats, model_lons, model_valid_mask)
        model_is_pre_masked = not np.all(model_valid_mask)

        # In supplied model-file mode, the model is treated as already bounded/masked.
        # Observation handling depends on source:
        # - supplied obs file: verify already bounded/masked (no remasking).
        # - obs directory: apply supplied region mask, and if model is not masked,
        #   additionally bound obs by model lat/lon extent.
        if args.obs_file is not None:
            obs_region_bbox = None
            obs_region_shape = None
            obs_required_coverage_extent = None
            obs_spatial_clip_bbox = None
        else:
            if args.region_shapefile is None:
                raise ValueError(
                    "When using --model-file with --obs-dir, supply --region-shapefile "
                    "used for observation masking."
                )
            obs_region_bbox = region_bbox
            obs_region_shape = region_shape
            if not model_is_pre_masked:
                obs_region_bbox = model_extent if obs_region_bbox is None else (
                    max(obs_region_bbox[0], model_extent[0]),
                    min(obs_region_bbox[1], model_extent[1]),
                    max(obs_region_bbox[2], model_extent[2]),
                    min(obs_region_bbox[3], model_extent[3]),
                )
                if obs_region_bbox[0] > obs_region_bbox[1] or obs_region_bbox[2] > obs_region_bbox[3]:
                    raise ValueError("Requested observation region does not overlap model extent")
            obs_required_coverage_extent = model_extent
            obs_spatial_clip_bbox = model_extent

        # Recompute model yearly series using the final region mask.
        model_years, model_data, model_units = _load_model_series_from_cube(
            model_cube,
            leadtime_indices,
            model_valid_mask,
            source_label=str(args.model_file),
        )
        pooled_rows = model_data.shape[0]
        model_years, model_data = _collapse_pooled_years_to_unique(model_years, model_data)
        leadtime_summary = f"{len(leadtime_indices)} ({leadtime_indices[0]} to {leadtime_indices[-1]})"
    else:
        model_mode = "raw-model-directory"
        model_files = sorted(args.model_dir.glob(args.model_pattern))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {args.model_dir} matching {args.model_pattern!r}")

        if region_bbox is None and region_shape is None:
            raise ValueError(
                "A regional constraint is required in raw model-directory mode. "
                "Provide either --region-bbox or --region-shapefile."
            )

        raw_clip_bbox = region_bbox
        if region_shape is not None:
            shape_bbox = _region_shape_bounds(region_shape)
            raw_clip_bbox = shape_bbox if raw_clip_bbox is None else _intersect_bboxes(raw_clip_bbox, shape_bbox)

        model_lats, model_lons, model_valid_mask, model_extent = _extract_raw_model_spatial_info(
            model_files[0],
            args.model_var,
            region_bbox,
            region_shape,
            spatial_clip_bbox=raw_clip_bbox,
        )
        model_years, model_data, model_units, pooled_rows, full_pool_size = _load_model_series_from_raw_directory(
            args.model_dir,
            args.model_var,
            obs_months,
            model_valid_mask,
            args.start_year,
            args.end_year,
            spatial_clip_bbox=raw_clip_bbox,
            pattern=args.model_pattern,
        )
        leadtime_summary = f"n/a (raw mode, pooled simulations per year={full_pool_size})"
        obs_region_bbox = region_bbox
        obs_region_shape = region_shape
        # Obs are pre-clipped to model extent in raw mode; an additional strict
        # extent coverage gate can fail spuriously due grid-edge differences.
        obs_required_coverage_extent = None
        obs_spatial_clip_bbox = model_extent

    model_years, model_data = _subset_year_range(model_years, model_data, args.start_year, args.end_year)
    if model_years.size == 0:
        raise ValueError("No model years remain after applying year range filters")

    if args.obs_file is not None:
        obs_preloaded_cube = None
        if args.model_file is not None:
            obs_preloaded_cube = _load_variable_cube(args.obs_file, args.obs_var)
            _assert_obs_is_prebounded(obs_preloaded_cube, model_extent, str(args.obs_file))
        obs_years, obs_data = _load_observation_series(
            args.obs_file,
            args.obs_var,
            obs_months,
            args.start_year,
            args.end_year,
            model_units,
            obs_region_bbox,
            obs_region_shape,
            obs_required_coverage_extent,
            preloaded_cube=obs_preloaded_cube,
            spatial_clip_bbox=obs_spatial_clip_bbox,
        )
    else:
        obs_years, obs_data = _load_observation_directory(
            args.obs_dir,
            args.obs_var,
            obs_months,
            args.start_year,
            args.end_year,
            model_units,
            obs_region_bbox,
            obs_region_shape,
            obs_required_coverage_extent,
            spatial_clip_bbox=obs_spatial_clip_bbox,
            pattern=args.obs_pattern,
        )

    obs_years, obs_data = _subset_year_range(obs_years, obs_data, args.start_year, args.end_year)
    if obs_years.size == 0:
        raise ValueError("No observation years remain after applying year range filters")

    common_years, model_aligned, obs_aligned = _align_on_years(model_years, model_data, obs_years, obs_data)
    model_dist_stats = _overall_distribution_stats(model_aligned)
    obs_dist_stats = _overall_distribution_stats(obs_aligned)

    if args.save_prepared_obs is not None:
        _write_prepared_observations(args.save_prepared_obs, common_years, obs_aligned)
    if args.save_prepared_model is not None:
        _write_prepared_model_csv(args.save_prepared_model, common_years, model_aligned)
    if args.save_prepared_obs_nc is not None:
        _write_prepared_observations_netcdf(args.save_prepared_obs_nc, common_years, obs_aligned, args.obs_var)
    if args.save_prepared_model_nc is not None:
        _write_prepared_model_netcdf(args.save_prepared_model_nc, common_years, model_aligned, args.model_var)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fidelity_dir = outdir / "Fidelity_Testing"
    fidelity_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model source mode: {model_mode}")
    print(f"Model rows: {model_aligned.shape[0]}, model members: {model_aligned.shape[1]}")
    print(f"Leadtimes used: {leadtime_summary}")
    print(f"Observation dataset: {obs_dataset}")
    print(
        f"Leadtime pooling: {pooled_rows} pooled model-year contributions collapsed to "
        f"{model_aligned.shape[0]} unique calendar years with full simulation coverage "
        f"({model_aligned.shape[1]} simulations per year)"
    )
    print(f"Model ensembles used in fidelity testing: {model_aligned.shape[1]}")
    print(
        "Total simulated model years used in fidelity testing: "
        f"{model_aligned.shape[0] * model_aligned.shape[1]} "
        f"({model_aligned.shape[0]} years x {model_aligned.shape[1]} simulations)"
    )
    print(
        "Overall model distribution (all retained year x simulation values): "
        f"mean={model_dist_stats['mean']:.3f}, std={model_dist_stats['std']:.3f}, "
        f"min={model_dist_stats['min']:.3f}, max={model_dist_stats['max']:.3f}, "
        f"p05={model_dist_stats['p05']:.3f}, p95={model_dist_stats['p95']:.3f}"
    )
    print(
        "Overall observation distribution (all retained years): "
        f"mean={obs_dist_stats['mean']:.3f}, std={obs_dist_stats['std']:.3f}, "
        f"min={obs_dist_stats['min']:.3f}, max={obs_dist_stats['max']:.3f}, "
        f"p05={obs_dist_stats['p05']:.3f}, p95={obs_dist_stats['p95']:.3f}"
    )
    print(f"Observations: {obs_aligned.shape[0]} years")
    unique_common_years = np.unique(common_years)
    print(
        f"Common years: {unique_common_years[0]} to {unique_common_years[-1]} "
        f"({unique_common_years.size} unique years)"
    )
    if model_units is not None:
        print(f"Units normalized to: {model_units}")
    print(
        "Model extent: lat {:.3f} to {:.3f}, lon {:.3f} to {:.3f}".format(
            model_extent[0], model_extent[1], model_extent[2], model_extent[3]
        )
    )
    print(f"Model valid cells in mask: {int(np.sum(model_valid_mask))} / {model_valid_mask.size}")

    ftc = FidelityTestCube()
    stats_measures = ftc.timeseries_fid_test(obs_aligned, model_aligned, seed=args.seed)

    output_png = fidelity_dir / f"{args.label}_fidelity.png"
    if args.region_shapefile is not None:
        region_label = args.region_shapefile.stem
    elif args.region_bbox is not None:
        region_label = "region-bbox"
    else:
        region_label = "region-unknown"
    plot_title = f"{region_label} | Obs: {obs_dataset} | Variable: {args.obs_var}"
    with mpl.rc_context(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 13,
        }
    ):
        ftc.plot_fidelity_testing(obs_aligned, model_aligned, stats_measures, 0.1, plot_title, str(output_png))

    # Add run metadata directly to the figure so the context is preserved with the plot.
    metadata_lines = [
        f"Model source: {model_mode}",
        f"Leadtimes used: {leadtime_summary}",
        f"Obs dataset: {obs_dataset}",
        f"Obs variable: {args.obs_var}",
        f"Analysis months: {', '.join(str(month) for month in obs_months) if obs_months else 'all months'}",
        f"Region: {region_label}",
        f"Pooled rows collapsed to unique years: {pooled_rows} -> {model_aligned.shape[0]}",
        f"Model ensembles: {model_aligned.shape[1]}",
        (
            "Total simulated model years: "
            f"{model_aligned.shape[0] * model_aligned.shape[1]} "
            f"({model_aligned.shape[0]} x {model_aligned.shape[1]} simulations)"
        ),
        (
            "Model dist: "
            f"mean={model_dist_stats['mean']:.2f}, std={model_dist_stats['std']:.2f}, "
            f"p05={model_dist_stats['p05']:.2f}, p95={model_dist_stats['p95']:.2f}"
        ),
        (
            "Obs dist: "
            f"mean={obs_dist_stats['mean']:.2f}, std={obs_dist_stats['std']:.2f}, "
            f"p05={obs_dist_stats['p05']:.2f}, p95={obs_dist_stats['p95']:.2f}"
        ),
        (
            f"Common years: {unique_common_years[0]} to {unique_common_years[-1]} "
            f"({unique_common_years.size} unique)"
        ),
        f"Model valid cells: {int(np.sum(model_valid_mask))} / {model_valid_mask.size}",
    ]
    split_index = (len(metadata_lines) + 1) // 2
    left_text = "\n".join(metadata_lines[:split_index])
    right_text = "\n".join(metadata_lines[split_index:])
    fig = plt.gcf()
    # Make the output figure larger and reserve a dedicated metadata panel below
    # the plotting axes so labels and diagnostics never overlap.
    fig.set_size_inches(14, 9)
    fig.subplots_adjust(bottom=0.33)
    metadata_ax = fig.add_axes([0.05, 0.05, 0.90, 0.22])
    metadata_ax.axis("off")
    metadata_ax.text(
        0.0,
        1.0,
        left_text,
        ha="left",
        va="top",
        transform=metadata_ax.transAxes,
        fontsize=9,
        linespacing=1.25,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    metadata_ax.text(
        0.52,
        1.0,
        right_text,
        ha="left",
        va="top",
        transform=metadata_ax.transAxes,
        fontsize=9,
        linespacing=1.25,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    plt.savefig(output_png, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
