#!/usr/bin/env python3
"""Standalone fidelity testing for one variable using Iris.

This script:
1. Loads model and observation data with Iris cubes.
2. Applies unit normalization (Kelvin/Celsius for temperature fields).
3. Transfers the model valid-cell mask to observations.
4. Builds annual observation series (single file or monthly directory input).
5. Aligns model and observations on common years and runs fidelity testing.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import iris
import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from fidelity_test_cube import FidelityTestCube


MONTHLY_FILE_RE = re.compile(r"(?P<year>\d{4})(?P<month>\d{2})(?:\D|$)")
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


def _build_nearest_indices(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Map each target coordinate to the nearest source-grid index."""
    return np.array([int(np.argmin(np.abs(source_coords - value))) for value in target_coords], dtype=int)


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


def _load_model_series(
    model_path: Path,
    variable_name: str,
    leadtime_indices: list[int],
    model_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Load model data as (simulated_year_row, member) across selected leadtimes."""
    cube = _load_variable_cube(model_path, variable_name)
    _debug_cube(cube, "_load_model_series: raw model cube")
    model_units = _canonical_unit_name(str(cube.units))

    lead_coord = _find_coord(cube, ("leadtime",))
    lead_dim = cube.coord_dims(lead_coord)[0]
    data = _as_float_array(cube.data)
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
        raise KeyError(f"Model file {model_path} does not contain a usable year coordinate") from exc

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
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    model_valid_mask: np.ndarray,
    model_extent: tuple[float, float, float, float],
) -> float:
    """Map an observation field onto model grid, apply model mask, and area-average."""
    data = _as_float_array(cube.data)
    data = _convert_temperature_units(data, str(cube.units), target_units)

    lat_coord = _find_coord(cube, LAT_NAMES)
    lon_coord = _find_coord(cube, LON_NAMES)
    lat_dim = cube.coord_dims(lat_coord)[0]
    lon_dim = cube.coord_dims(lon_coord)[0]

    obs_lats = np.asarray(lat_coord.points, dtype=float)
    obs_lons = np.asarray(lon_coord.points, dtype=float)

    # Coverage check before nearest-grid remapping.
    if np.nanmin(obs_lats) > model_extent[0] or np.nanmax(obs_lats) < model_extent[1]:
        raise ValueError("Observation latitude coverage does not include the full model extent")
    if np.nanmin(obs_lons) > model_extent[2] or np.nanmax(obs_lons) < model_extent[3]:
        raise ValueError("Observation longitude coverage does not include the full model extent")

    lat_idx = _build_nearest_indices(obs_lats, model_lats)
    lon_idx = _build_nearest_indices(obs_lons, model_lons)
    data = np.take(data, lat_idx, axis=lat_dim)
    data = np.take(data, lon_idx, axis=lon_dim)

    # Apply same valid-cell mask used for model averaging.
    data = np.where(model_valid_mask, data, np.nan)
    return float(np.nanmean(data))


def _load_observation_series(
    obs_path: Path,
    variable_name: str,
    obs_months: tuple[int, ...] | None,
    target_units: str | None,
    model_extent: tuple[float, float, float, float],
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    model_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one observation file and build yearly series.

    If the file has a time coordinate, selected months are grouped to yearly means.
    If the file has a year coordinate, values are used directly.
    """
    cube = _load_variable_cube(obs_path, variable_name)
    _debug_cube(cube, "_load_observation_series: raw obs cube")

    try:
        year_coord = cube.coord("year")
        years = np.asarray(year_coord.points, dtype=int).squeeze()
        values = []

        # Use a year-by-year extraction when year is 1D; otherwise reduce full field.
        if years.ndim == 1 and years.size > 1:
            year_dim = cube.coord_dims(year_coord)[0]
            for i in range(years.size):
                sub_cube = _slice_dim(cube, year_dim, i)
                values.append(
                    _obs_field_to_masked_mean(
                        sub_cube, target_units, model_lats, model_lons, model_valid_mask, model_extent
                    )
                )
        else:
            year_value = int(np.asarray(years).reshape(-1)[0])
            values.append(
                _obs_field_to_masked_mean(
                    cube, target_units, model_lats, model_lons, model_valid_mask, model_extent
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
            value = _obs_field_to_masked_mean(
                cube, target_units, model_lats, model_lons, model_valid_mask, model_extent
            )
            return np.array([year], dtype=int), np.array([value], dtype=float)

    # General time-series path.
    datetimes = time_coord.units.num2date(time_coord.points)
    month_filter = set(obs_months) if obs_months else None
    grouped: dict[int, list[float]] = {}
    months_seen: dict[int, set[int]] = {}

    time_dim = cube.coord_dims(time_coord)[0]
    for i, dt in enumerate(datetimes):
        if month_filter is not None and int(dt.month) not in month_filter:
            continue
        sub_cube = _slice_dim(cube, time_dim, i)
        value = _obs_field_to_masked_mean(
            sub_cube, target_units, model_lats, model_lons, model_valid_mask, model_extent
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
    target_units: str | None,
    model_extent: tuple[float, float, float, float],
    model_lats: np.ndarray,
    model_lons: np.ndarray,
    model_valid_mask: np.ndarray,
    pattern: str = "*.nc",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one-file-per-month observations and aggregate to annual means."""
    files = sorted(obs_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No observation files found in {obs_dir} matching {pattern!r}")
    _debug(f"_load_observation_directory: found {len(files)} files matching {pattern!r}")
    if _DEBUG:
        _debug_cube(_load_variable_cube(files[0], variable_name), f"_load_observation_directory: first file cube ({files[0].name})")

    month_filter = set(obs_months) if obs_months else None
    grouped: dict[int, list[float]] = {}
    months_seen: dict[int, set[int]] = {}

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

        cube = _load_variable_cube(obs_file, variable_name)
        value = _obs_field_to_masked_mean(
            cube,
            target_units,
            model_lats,
            model_lons,
            model_valid_mask,
            model_extent,
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
    parser.add_argument("--model-file", type=Path, required=True, help="Path to model netCDF file")
    parser.add_argument("--model-var", default="mean_jja_temperature", help="Model variable name")

    obs_input = parser.add_mutually_exclusive_group(required=True)
    obs_input.add_argument("--obs-file", type=Path, help="Single observation netCDF file")
    obs_input.add_argument("--obs-dir", type=Path, help="Directory of monthly observation netCDF files")

    parser.add_argument("--obs-var", default="t2m", help="Observation variable name")
    parser.add_argument(
        "--leadtime-index",
        type=int,
        default=None,
        help="Optional single leadtime index. If omitted, all valid leadtimes are used.",
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
    obs_dataset = _obs_dataset_label(args.obs_file, args.obs_dir)

    model_cube = _load_variable_cube(args.model_file, args.model_var)
    leadtime_indices = _resolve_leadtime_indices(model_cube, args.leadtime_index)

    # Derive model grid and mask once, then reuse for both model and observations.
    model_lats, model_lons, model_valid_mask, model_extent = _extract_model_spatial_info(
        args.model_file,
        args.model_var,
        leadtime_indices,
    )

    model_years, model_data, model_units = _load_model_series(
        args.model_file,
        args.model_var,
        leadtime_indices,
        model_valid_mask,
    )
    pooled_rows = model_data.shape[0]
    model_years, model_data = _collapse_pooled_years_to_unique(model_years, model_data)

    obs_months = tuple(args.obs_months) if args.obs_months else None
    if args.obs_file is not None:
        obs_years, obs_data = _load_observation_series(
            args.obs_file,
            args.obs_var,
            obs_months,
            model_units,
            model_extent,
            model_lats,
            model_lons,
            model_valid_mask,
        )
    else:
        obs_years, obs_data = _load_observation_directory(
            args.obs_dir,
            args.obs_var,
            obs_months,
            model_units,
            model_extent,
            model_lats,
            model_lons,
            model_valid_mask,
            pattern=args.obs_pattern,
        )

    common_years, model_aligned, obs_aligned = _align_on_years(model_years, model_data, obs_years, obs_data)
    model_dist_stats = _overall_distribution_stats(model_aligned)
    obs_dist_stats = _overall_distribution_stats(obs_aligned)

    if args.save_prepared_obs is not None:
        _write_prepared_observations(args.save_prepared_obs, common_years, obs_aligned)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fidelity_dir = outdir / "Fidelity_Testing"
    fidelity_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model rows: {model_aligned.shape[0]}, model members: {model_aligned.shape[1]}")
    print(f"Leadtimes used: {len(leadtime_indices)} ({leadtime_indices[0]} to {leadtime_indices[-1]})")
    print(f"Observation dataset: {obs_dataset}")
    print(
        f"Leadtime pooling: {pooled_rows} pooled (init x leadtime) rows collapsed to "
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
    ftc.plot_fidelity_testing(obs_aligned, model_aligned, stats_measures, 0.1, "", str(output_png))

    # Add run metadata directly to the figure so the context is preserved with the plot.
    metadata_text = "\n".join(
        [
            f"Leadtimes used: {len(leadtime_indices)} ({leadtime_indices[0]} to {leadtime_indices[-1]})",
            f"Obs dataset: {obs_dataset}",
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
    )
    plt.figtext(
        0.02,
        0.98,
        metadata_text,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_png}")


if __name__ == "__main__":
    main()
