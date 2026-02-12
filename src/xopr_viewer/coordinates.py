"""Coordinate conversion utilities for axis mode display.

Provides bidirectional conversion between canonical coordinates
(slow_time, twtt in seconds) and display coordinates that depend
on the selected axis modes. Delegates heavy computation to
``xopr.radar_util``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.constants
import xarray as xr
from xopr.radar_util import add_along_track

# Display dimension name for each axis mode
X_DIM_NAMES: dict[str, str] = {
    "rangeline": "trace",
    "gps_time": "slow_time",
    "along_track": "along_track_km",
}

Y_DIM_NAMES: dict[str, str] = {
    "twtt": "twtt_us",
    "range_bin": "range_bin",
    "range": "range_m",
    "elevation": "elevation_m",
    "surface_flat": "depth_m",
}

# Whether y-axis should be inverted for each mode
Y_INVERT: dict[str, bool] = {
    "twtt": True,
    "range_bin": True,
    "range": True,
    "elevation": False,
    "surface_flat": True,
}


def _along_track_km(ds: xr.Dataset) -> np.ndarray:
    """Return cumulative along-track distance in km.

    Calls ``xopr.add_along_track`` which uses polar stereographic
    projection (EPSG:3031 or EPSG:3413 based on latitude).
    """
    ds_at = add_along_track(ds)
    return ds_at["along_track"].values / 1000.0


def display_to_canonical(
    x: float,
    y: float,
    ds: xr.Dataset,
    x_mode: str,
    y_mode: str,
) -> tuple[Any, float]:
    """Convert display coordinates to canonical (slow_time, twtt in seconds).

    Parameters
    ----------
    x, y : float
        Coordinates in the current display system.
    ds : xr.Dataset
        The echogram dataset (needed for index lookups and inverse transforms).
    x_mode, y_mode : str
        Current axis mode identifiers.

    Returns
    -------
    tuple
        ``(slow_time_value, twtt_seconds)`` in canonical coordinates.
    """
    # --- X conversion ---
    slow_time_vals = ds["slow_time"].values

    if x_mode == "rangeline":
        idx = int(np.clip(np.round(x), 0, len(slow_time_vals) - 1))
        canonical_x = slow_time_vals[idx]
    elif x_mode == "gps_time":
        canonical_x = x  # already slow_time
    elif x_mode == "along_track":
        along_km = _along_track_km(ds)
        # Interpolate back: km → slow_time index
        idx = int(
            np.clip(
                np.round(np.interp(x, along_km, np.arange(len(slow_time_vals)))),
                0,
                len(slow_time_vals) - 1,
            )
        )
        canonical_x = slow_time_vals[idx]
    else:
        raise ValueError(f"Unknown x_mode: {x_mode!r}")

    # --- Y conversion ---
    twtt_vals = ds["twtt"].values

    if y_mode == "twtt":
        canonical_y = y * 1e-6  # µs → s
    elif y_mode == "range_bin":
        idx = int(np.clip(np.round(y), 0, len(twtt_vals) - 1))
        canonical_y = float(twtt_vals[idx])
    elif y_mode == "range":
        canonical_y = y * 2.0 / scipy.constants.c  # range_m → twtt_s
    elif y_mode == "elevation":
        # Inverse of elevation transform requires per-trace vertical distances.
        # Find nearest trace, then invert elevation → twtt.
        trace_idx = _nearest_trace_idx(canonical_x, slow_time_vals)
        canonical_y = _elevation_to_twtt(y, trace_idx, ds)
    elif y_mode == "surface_flat":
        trace_idx = _nearest_trace_idx(canonical_x, slow_time_vals)
        canonical_y = _depth_to_twtt(y, trace_idx, ds)
    else:
        raise ValueError(f"Unknown y_mode: {y_mode!r}")

    return canonical_x, canonical_y


def canonical_to_display(
    slow_time: Any,
    twtt: float,
    ds: xr.Dataset,
    x_mode: str,
    y_mode: str,
) -> tuple[Any, float]:
    """Convert canonical (slow_time, twtt seconds) to display coordinates.

    Parameters
    ----------
    slow_time : datetime64 or float
        Canonical x coordinate.
    twtt : float
        Canonical y coordinate in seconds.
    ds : xr.Dataset
        The echogram dataset.
    x_mode, y_mode : str
        Current axis mode identifiers.

    Returns
    -------
    tuple
        ``(display_x, display_y)`` in the current display system.
    """
    slow_time_vals = ds["slow_time"].values

    # --- X conversion ---
    if x_mode == "rangeline":
        display_x = float(_nearest_trace_idx(slow_time, slow_time_vals))
    elif x_mode == "gps_time":
        display_x = slow_time  # passthrough
    elif x_mode == "along_track":
        along_km = _along_track_km(ds)
        idx = _nearest_trace_idx(slow_time, slow_time_vals)
        display_x = float(along_km[idx])
    else:
        raise ValueError(f"Unknown x_mode: {x_mode!r}")

    # --- Y conversion ---
    if y_mode == "twtt":
        display_y = twtt * 1e6  # s → µs
    elif y_mode == "range_bin":
        twtt_vals = ds["twtt"].values
        display_y = float(np.argmin(np.abs(twtt_vals - twtt)))
    elif y_mode == "range":
        display_y = twtt * scipy.constants.c / 2.0  # twtt_s → range_m
    elif y_mode == "elevation":
        trace_idx = _nearest_trace_idx(slow_time, slow_time_vals)
        display_y = _twtt_to_elevation(twtt, trace_idx, ds)
    elif y_mode == "surface_flat":
        trace_idx = _nearest_trace_idx(slow_time, slow_time_vals)
        display_y = _twtt_to_depth(twtt, trace_idx, ds)
    else:
        raise ValueError(f"Unknown y_mode: {y_mode!r}")

    return display_x, display_y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_trace_idx(slow_time_value: Any, slow_time_vals: np.ndarray) -> int:
    """Find the index of the nearest slow_time value."""
    arr = np.asarray(slow_time_value)
    if np.issubdtype(arr.dtype, np.datetime64):
        target_i = arr.astype("datetime64[ns]").astype(np.int64)
        vals_i = slow_time_vals.astype("datetime64[ns]").astype(np.int64)
        return int(np.argmin(np.abs(vals_i - target_i)))
    target_f = float(arr)
    vals_f = slow_time_vals.astype(np.float64)
    return int(np.argmin(np.abs(vals_f - target_f)))


def _twtt_to_elevation(twtt: float, trace_idx: int, ds: xr.Dataset) -> float:
    """Convert TWTT (seconds) to WGS-84 elevation (meters) for a single trace."""
    c = scipy.constants.c
    n_ice = np.sqrt(3.15)

    aircraft_elev = float(ds["Elevation"].values[trace_idx])
    surface_twtt = float(ds["Surface"].values[trace_idx])

    if twtt <= surface_twtt:
        # Above surface: air velocity
        return aircraft_elev - twtt * c / 2.0
    else:
        # Below surface: ice velocity
        surface_range = surface_twtt * c / 2.0
        ice_range = (twtt - surface_twtt) * c / (2.0 * n_ice)
        return aircraft_elev - surface_range - ice_range


def _elevation_to_twtt(elev: float, trace_idx: int, ds: xr.Dataset) -> float:
    """Convert WGS-84 elevation (meters) to TWTT (seconds) for a single trace."""
    c = scipy.constants.c
    n_ice = np.sqrt(3.15)

    aircraft_elev = float(ds["Elevation"].values[trace_idx])
    surface_twtt = float(ds["Surface"].values[trace_idx])
    surface_elev = aircraft_elev - surface_twtt * c / 2.0

    if elev >= surface_elev:
        # Above surface: air velocity
        return (aircraft_elev - elev) * 2.0 / c
    else:
        # Below surface: ice velocity
        ice_range = surface_elev - elev
        return surface_twtt + ice_range * 2.0 * n_ice / c


def _twtt_to_depth(twtt: float, trace_idx: int, ds: xr.Dataset) -> float:
    """Convert TWTT (seconds) to depth below surface (meters)."""
    c = scipy.constants.c
    n_ice = np.sqrt(3.15)
    surface_twtt = float(ds["Surface"].values[trace_idx])
    return (twtt - surface_twtt) * c / (2.0 * n_ice)


def _depth_to_twtt(depth: float, trace_idx: int, ds: xr.Dataset) -> float:
    """Convert depth below surface (meters) to TWTT (seconds)."""
    c = scipy.constants.c
    n_ice = np.sqrt(3.15)
    surface_twtt = float(ds["Surface"].values[trace_idx])
    return surface_twtt + depth * 2.0 * n_ice / c
