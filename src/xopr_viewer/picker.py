"""
Grounding line picking tool using HoloViews streams.

Inspired by CReSIS imb.picker MATLAB tool.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import scipy.constants
from xopr.radar_util import add_along_track, interpolate_to_vertical_grid

from xopr_viewer.coordinates import (
    Y_INVERT,
    display_to_canonical,
    canonical_to_display,
    X_DIM_NAMES,
    Y_DIM_NAMES,
)

if TYPE_CHECKING:
    import xarray as xr


def _interpolate_uniform(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Interpolate a DataArray onto a uniform grid along *dim*.

    Mirrors the MATLAB imb.picker approach: compute the median step size
    from the existing coordinate, build a uniform grid spanning the same
    range, and linearly interpolate the data onto it.

    Handles both numeric and datetime64 coordinates.
    """
    coords = da.coords[dim].values
    if len(coords) < 3:
        return da

    is_datetime = np.issubdtype(coords.dtype, np.datetime64)

    if is_datetime:
        coords_num = coords.astype("datetime64[ns]").astype(np.int64)
    else:
        coords_num = coords.astype(np.float64)

    diffs = np.diff(coords_num)
    median_step = np.median(diffs)

    if median_step <= 0:
        raise ValueError(
            f"Coordinate '{dim}' is not monotonically increasing "
            f"(median step = {median_step}). "
            "Sort or fix the coordinate before plotting."
        )

    # Check if already uniform (all diffs within 1% of median)
    if np.allclose(diffs, median_step, rtol=0.01):
        return da

    uniform_num = np.arange(
        coords_num[0], coords_num[-1] + median_step / 2, median_step
    )

    if is_datetime:
        uniform_coords = uniform_num.astype("datetime64[ns]")
    else:
        uniform_coords = uniform_num

    return da.interp({dim: uniform_coords}, method="linear")


# Layer colors for visualization
_LAYER_COLORS = {
    "standard:surface": "cyan",
    "standard:bottom": "yellow",
    ":surface": "cyan",
    ":bottom": "yellow",
    "surface": "cyan",
    "bottom": "yellow",
    "internal": "lime",
}
_DEFAULT_LAYER_COLOR = "red"


def _get_title(ds: xr.Dataset) -> str:
    """Get title from dataset attributes, handling merged frame sets."""
    segment = ds.attrs.get("segment_path", "")
    if isinstance(segment, set):
        segment = sorted(segment)[0] if segment else ""

    granule = ds.attrs.get("granule", "")
    if isinstance(granule, set):
        return f"{segment} ({len(granule)} frames)" if segment else "Echogram"
    elif granule:
        return str(granule)
    elif segment:
        return str(segment)
    return "Echogram"


def _create_image(
    ds: xr.Dataset,
    data_var: str = "Data",
    x_mode: str = "gps_time",
    y_mode: str = "twtt",
    log_scale: bool = True,
    clim_percentile: tuple[float, float] = (5, 99),
    cmap: str = "gray",
) -> hv.Image:
    """Create HoloViews Image from echogram data.

    Parameters
    ----------
    ds : xr.Dataset
        Echogram dataset with ``Data``, ``twtt``, and ``slow_time``.
    x_mode : str
        X-axis display mode: ``"rangeline"``, ``"gps_time"``, or
        ``"along_track"``.
    y_mode : str
        Y-axis display mode: ``"twtt"``, ``"range_bin"``, ``"range"``,
        ``"elevation"``, or ``"surface_flat"``.
    """
    if data_var not in ds:
        raise ValueError(f"Data variable '{data_var}' not found")

    _MAX_ELEMENTS = 10_000_000
    n_elements = ds[data_var].size
    if n_elements > _MAX_ELEMENTS:
        warnings.warn(
            f"Array has {n_elements:,} elements ({n_elements / 1e6:.1f}M), "
            f"which exceeds the recommended limit of {_MAX_ELEMENTS / 1e6:.0f}M "
            f"for interactive rendering. Consider slicing or downsampling before plotting.",
            stacklevel=2,
        )

    # --- Y-axis transform (before x, regridding replaces twtt dim) ---
    if y_mode in ("elevation", "surface_flat"):
        vert_coord = "wgs84" if y_mode == "elevation" else "range"
        ds_vert = interpolate_to_vertical_grid(ds, vertical_coordinate=vert_coord)
        da = ds_vert[data_var]
        # Rename dim to our display name
        src_dim = vert_coord  # 'wgs84' or 'range'
        tgt_dim = Y_DIM_NAMES[y_mode]
        if y_mode == "surface_flat":
            # Convert range-from-aircraft to depth-below-surface
            surface_range = ds["Surface"].mean().values * scipy.constants.c / 2.0
            new_coords = da.coords[src_dim].values - surface_range
            da = da.assign_coords({src_dim: new_coords})
        da = da.rename({src_dim: tgt_dim})
    else:
        da = ds[data_var]
        if y_mode == "twtt":
            if "twtt" in da.dims and "twtt_us" not in da.coords:
                da = da.assign_coords(twtt_us=("twtt", da.coords["twtt"].values * 1e6))
                da = da.swap_dims({"twtt": "twtt_us"})
        elif y_mode == "range_bin":
            if "twtt" in da.dims:
                da = da.assign_coords(
                    range_bin=("twtt", np.arange(len(da.coords["twtt"])))
                )
                da = da.swap_dims({"twtt": "range_bin"})
        elif y_mode == "range":
            if "twtt" in da.dims:
                range_m = da.coords["twtt"].values * scipy.constants.c / 2.0
                da = da.assign_coords(range_m=("twtt", range_m))
                da = da.swap_dims({"twtt": "range_m"})

    # --- X-axis transform ---
    if x_mode == "rangeline":
        if "slow_time" in da.dims:
            da = da.assign_coords(
                trace=("slow_time", np.arange(len(da.coords["slow_time"])))
            )
            da = da.swap_dims({"slow_time": "trace"})
    elif x_mode == "gps_time":
        if "slow_time" in da.dims:
            da = _interpolate_uniform(da, "slow_time")
    elif x_mode == "along_track":
        if "slow_time" in da.dims:
            ds_at = add_along_track(ds)
            along_km = ds_at["along_track"].values / 1000.0
            # Match length after possible y-axis regridding
            if len(along_km) == len(da.coords["slow_time"]):
                da = da.assign_coords(along_track_km=("slow_time", along_km))
                da = da.swap_dims({"slow_time": "along_track_km"})
                da = _interpolate_uniform(da, "along_track_km")

    # --- Log scale and clim ---
    if log_scale:
        da = cast("xr.DataArray", 10 * np.log10(np.abs(da) + 1e-10))

    clim = (
        float(np.nanpercentile(da.values, clim_percentile[0])),
        float(np.nanpercentile(da.values, clim_percentile[1])),
    )

    title = _get_title(ds)
    invert_y = Y_INVERT.get(y_mode, True)

    # Explicit kdims ordering: [x_dim, y_dim].  The DataArray's dim order
    # (y, x) comes from the original shape (twtt, slow_time) and does NOT
    # match the HoloViews convention where kdims[0] = x-axis.
    x_dim_name = X_DIM_NAMES.get(x_mode, "slow_time")
    y_dim_name = Y_DIM_NAMES.get(y_mode, "twtt_us")

    return hv.Image(da, kdims=[x_dim_name, y_dim_name], vdims=["power_dB"]).opts(
        cmap=cmap,
        colorbar=True,
        clim=clim,
        invert_yaxis=invert_y,
        title=title,
    )


def _create_layer_curves(
    layers: dict[str, xr.Dataset],
    visible_layers: list[str] | None = None,
    ds: xr.Dataset | None = None,
    x_mode: str = "gps_time",
    y_mode: str = "twtt",
) -> dict[str, hv.Curve]:
    """Create HoloViews Curve elements for each layer.

    When *ds* and non-default axis modes are provided, layer coordinates
    are converted to match the current display system.
    """
    curves = {}
    x_dim = X_DIM_NAMES.get(x_mode, "slow_time")
    y_dim = Y_DIM_NAMES.get(y_mode, "twtt_us")

    for name, layer_ds in layers.items():
        if visible_layers is not None and name not in visible_layers:
            continue

        if "twtt" not in layer_ds:
            continue

        twtt_s = layer_ds.twtt.values

        if "slow_time" in layer_ds.coords:
            slow_times = layer_ds.slow_time.values
        else:
            slow_times = np.arange(len(twtt_s))

        if ds is not None and (x_mode != "gps_time" or y_mode != "twtt"):
            x_vals: list[Any] = []
            y_vals: list[float] = []
            for st, t in zip(slow_times, twtt_s):
                dx, dy = canonical_to_display(st, float(t), ds, x_mode, y_mode)
                x_vals.append(dx)
                y_vals.append(dy)
            x_values = np.array(x_vals)
            y_values = np.array(y_vals)
        else:
            x_values = slow_times
            y_values = twtt_s * 1e6

        color = _LAYER_COLORS.get(name, _DEFAULT_LAYER_COLOR)

        curves[name] = hv.Curve(
            (x_values, y_values),
            kdims=[x_dim],
            vdims=[y_dim],
            label=name,
        ).opts(
            color=color,
            line_width=2,
            alpha=0.8,
        )

    return curves


def compute_layer_slope(
    twtt: xr.DataArray,
    smoothing_window: int = 1,
) -> xr.DataArray:
    """Compute slope of a layer profile after smoothing.

    Smoothing uses ``xr.DataArray.rolling`` with a centered uniform
    (box-car) window followed by ``.mean()``.  Every output point is the
    unweighted mean of the ``smoothing_window`` nearest input points.
    The window is centered so the smoothed value at index *i* averages
    indices ``i - window//2 … i + window//2``.

    Points within ``window // 2`` of either edge have fewer neighbors,
    so ``rolling().mean()`` returns NaN there.  These NaN edges propagate
    through the derivative and appear as gaps at the ends of the slope
    curve.

    The derivative is computed by ``xr.DataArray.differentiate``, which
    uses second-order central finite differences in the interior and
    first-order one-sided differences at the boundaries.  Because it
    operates on coordinate values (not integer indices), the result is in
    physical units: ``d(twtt) / d(coordinate)``.  For datetime coordinates
    like ``slow_time`` the denominator is in nanoseconds (NumPy's internal
    representation), giving slope in seconds per nanosecond.

    Developer notes — common customizations:

    * **Weighted smoothing** — Replace ``rolling().mean()`` with a
      Gaussian kernel (``scipy.ndimage.gaussian_filter1d``) or a
      Savitzky–Golay filter (``scipy.signal.savgol_filter``) to better
      preserve sharp features.  Both require scipy.
    * **Edge handling** — Pass ``min_periods=1`` to ``rolling()`` to get
      biased shorter-window estimates near boundaries instead of NaN gaps.
    * **Units** — For datetime coordinates, multiply by ``1e9`` to convert
      from seconds/nanosecond to dimensionless seconds/second, or convert
      the coordinate to elapsed seconds before calling this function.

    Parameters
    ----------
    twtt : xr.DataArray
        Two-way travel time values in seconds, indexed by a single dimension.
    smoothing_window : int
        Size of the rolling mean window for smoothing (1 = no smoothing).

    Returns
    -------
    xr.DataArray
        Slope as d(twtt)/d(coordinate) along the first dimension.
    """
    dim = twtt.dims[0]
    smoothing_window = max(1, min(smoothing_window, len(twtt)))

    if smoothing_window > 1:
        smoothed = twtt.rolling({dim: smoothing_window}, center=True).mean()
    else:
        smoothed = twtt

    return smoothed.differentiate(dim)


def _create_slope_curves(
    layers: dict[str, xr.Dataset],
    visible_layers: list[str] | None = None,
    smoothing_window: int = 1,
    ds: xr.Dataset | None = None,
    x_mode: str = "gps_time",
) -> dict[str, hv.Curve]:
    """Create HoloViews Curve elements for layer slopes.

    Parameters
    ----------
    layers : dict[str, xr.Dataset]
        Layer datasets, each with 'twtt' variable and 'slow_time' coordinate.
    visible_layers : list[str] | None
        Layer names to include, or None for all.
    smoothing_window : int
        Rolling mean window size for smoothing before gradient.
    ds : xr.Dataset, optional
        Echogram dataset (needed for x-coordinate conversion when
        *x_mode* is not ``"gps_time"``).
    x_mode : str
        X-axis display mode (default ``"gps_time"``).

    Returns
    -------
    dict[str, hv.Curve]
        Mapping of layer name to slope Curve element.
    """
    curves = {}
    x_dim = X_DIM_NAMES.get(x_mode, "slow_time")

    for name, layer_ds in layers.items():
        if visible_layers is not None and name not in visible_layers:
            continue

        if "twtt" not in layer_ds:
            continue

        slope = compute_layer_slope(layer_ds.twtt, smoothing_window)
        slope_values = slope.values

        if "slow_time" in layer_ds.coords:
            slow_times = layer_ds.slow_time.values
        else:
            slow_times = np.arange(len(slope_values))

        # Convert x-coordinates to match the current display mode
        if ds is not None and x_mode != "gps_time":
            x_vals: list[Any] = []
            for st in slow_times:
                dx, _ = canonical_to_display(st, 0.0, ds, x_mode, "twtt")
                x_vals.append(dx)
            x_values = np.array(x_vals)
        else:
            x_values = slow_times

        color = _LAYER_COLORS.get(name, _DEFAULT_LAYER_COLOR)

        curves[name] = hv.Curve(
            (x_values, slope_values),
            kdims=[x_dim],
            vdims=["twtt"],
            label=f"{name} slope",
        ).opts(
            color=color,
            line_width=2,
            alpha=0.8,
        )

    return curves


class GroundingLinePicker(param.Parameterized):
    """Interactive point picker for echogram annotation.

    Click on the image to add points. Points are always stored in
    canonical coordinates ``(slow_time, twtt)`` regardless of the
    current display axis mode, and converted to/from display
    coordinates at the boundary.

    Parameters
    ----------
    image : hv.Image
        The echogram image to annotate.
    ds : xr.Dataset, optional
        The echogram dataset (needed for coordinate conversion when
        axis modes other than the default are used).
    x_mode : str
        X-axis display mode (default: ``"gps_time"``).
    y_mode : str
        Y-axis display mode (default: ``"twtt"``).
    layers : dict, optional
        Dictionary of layer name -> xr.Dataset for snap-to-layer.
    snap_threshold : float
        Snap threshold in microseconds (default: 5.0).
    """

    points = param.List(default=[], doc="List of picked points as dicts")
    snap_enabled = param.Boolean(
        default=False, doc="Whether snapping to layers is enabled"
    )

    def __init__(
        self,
        image: hv.Image,
        ds: xr.Dataset | None = None,
        x_mode: str = "gps_time",
        y_mode: str = "twtt",
        x_dim: str | None = None,
        y_dim: str | None = None,
        layers: dict[str, xr.Dataset] | None = None,
        snap_threshold: float = 5.0,
        **params,
    ) -> None:
        super().__init__(**params)

        self._image = image
        self._ds = ds
        self._x_mode = x_mode
        self._y_mode = y_mode

        # Display dims: use explicit x_dim/y_dim if given, else derive from
        # mode when a dataset is provided, else fall back to image kdims.
        if x_dim is not None:
            self.x_dim = x_dim
        elif ds is not None and x_mode in X_DIM_NAMES:
            self.x_dim = X_DIM_NAMES[x_mode]
        elif hasattr(image, "kdims") and image.kdims:
            self.x_dim = str(image.kdims[0])
        else:
            self.x_dim = "x"

        if y_dim is not None:
            self.y_dim = y_dim
        elif ds is not None and y_mode in Y_DIM_NAMES:
            self.y_dim = Y_DIM_NAMES[y_mode]
        elif hasattr(image, "kdims") and len(image.kdims) > 1:
            self.y_dim = str(image.kdims[1])
        else:
            self.y_dim = "y"

        # Marker styling
        self._marker_opts = params.get(
            "marker_opts",
            {"color": "red", "size": 12, "marker": "cross", "line_width": 2},
        )

        # Layer snapping configuration
        self._layers = layers or {}
        self._visible_layers: list[str] | None = None
        self._snap_threshold = snap_threshold

        # Points data source for dynamic updates
        self._points_pipe = hv.streams.Pipe(data=[])

        # Will be set up when element() is called
        self._tap_stream: hv.streams.Tap | None = None
        self._cached_element: hv.Overlay | None = None
        self._cached_size: tuple[int, int] | None = None

    # --- Axis mode management ---

    def set_axis_modes(self, x_mode: str, y_mode: str, *, _notify: bool = True) -> None:
        """Update current axis modes (called when sidebar dropdowns change).

        Parameters
        ----------
        _notify : bool
            If True (default), send on the points pipe to re-render
            points in the new coordinate system.  Pass False when
            calling from inside a DynamicMap callback to avoid
            recursive overlay resolution.
        """
        self._x_mode = x_mode
        self._y_mode = y_mode
        self.x_dim = X_DIM_NAMES.get(x_mode, self.x_dim)
        self.y_dim = Y_DIM_NAMES.get(y_mode, self.y_dim)
        if _notify:
            # Re-render points in new display coordinates so the DynamicMap
            # updates its cached element and keeps extents compatible.
            self._points_pipe.send(self.points)

    @property
    def _display_x_dim(self) -> str:
        return X_DIM_NAMES.get(self._x_mode, self.x_dim)

    @property
    def _display_y_dim(self) -> str:
        return Y_DIM_NAMES.get(self._y_mode, self.y_dim)

    # --- Layer snapping ---

    @property
    def visible_layers(self) -> list[str] | None:
        """Layer names to snap to, or None for all layers."""
        return self._visible_layers

    @visible_layers.setter
    def visible_layers(self, value: list[str] | None) -> None:
        self._visible_layers = value

    def _snap_to_layer(self, slow_time: Any, twtt: float) -> tuple[Any, float]:
        """Snap canonical coordinates to nearest layer within threshold.

        Operates in canonical space (slow_time, twtt in seconds).
        Threshold comparison is done in microseconds.
        """
        if not self._layers:
            return slow_time, twtt

        layers_to_check = self._layers
        if self._visible_layers is not None:
            layers_to_check = {
                k: v for k, v in self._layers.items() if k in self._visible_layers
            }

        valid_layers = {
            name: layer_ds
            for name, layer_ds in layers_to_check.items()
            if "twtt" in layer_ds
        }

        if not valid_layers:
            return slow_time, twtt

        best_twtt = twtt
        best_dist_us = self._snap_threshold + 1

        for layer_ds in valid_layers.values():
            try:
                nearest = layer_ds.twtt.sel({"slow_time": slow_time}, method="nearest")
                layer_twtt = float(nearest.values)
            except KeyError:
                layer_twtt = float(layer_ds.twtt.values[0])

            dist_us = abs(twtt - layer_twtt) * 1e6
            if dist_us < best_dist_us:
                best_dist_us = dist_us
                best_twtt = layer_twtt

        if best_dist_us <= self._snap_threshold:
            return slow_time, best_twtt
        return slow_time, twtt

    # --- Tap handling ---

    def _on_tap(self, x: float | None, y: float | None) -> None:
        """Handle tap events — convert to canonical coords and store."""
        if x is None or y is None:
            return

        # Convert display → canonical
        if self._ds is not None:
            canonical_x, canonical_y = display_to_canonical(
                x, y, self._ds, self._x_mode, self._y_mode
            )
        else:
            # No dataset: store as-is (legacy / simple image mode)
            canonical_x, canonical_y = x, y

        # Snap in canonical space
        if self.snap_enabled and self._layers:
            canonical_x, canonical_y = self._snap_to_layer(canonical_x, canonical_y)

        point = {
            "id": str(uuid4())[:8],
            "slow_time": canonical_x,
            "twtt": canonical_y,
        }
        self.points = [*self.points, point]
        self._points_pipe.send(self.points)

    def _points_element(self, data: list[dict]) -> hv.Points:
        """Render canonical points in current display coordinates."""
        x_dim = self._display_x_dim
        y_dim = self._display_y_dim

        if not data:
            if self._ds is not None and self._x_mode == "gps_time":
                # Use NaT sentinel so the empty Points element reports
                # datetime64 extents, preventing dtype clashes with
                # datetime64 Image/Curve overlays in holoviews.
                empty = pd.DataFrame(
                    {
                        x_dim: np.array([np.datetime64("NaT")], dtype="datetime64[ns]"),
                        y_dim: np.array([np.nan], dtype=float),
                    }
                )
                return hv.Points(empty, kdims=[x_dim, y_dim]).opts(**self._marker_opts)
            return hv.Points([], kdims=[x_dim, y_dim]).opts(**self._marker_opts)

        if self._ds is not None:
            display_pts = []
            for p in data:
                dx, dy = canonical_to_display(
                    p["slow_time"],
                    p["twtt"],
                    self._ds,
                    self._x_mode,
                    self._y_mode,
                )
                display_pts.append({x_dim: dx, y_dim: dy})
            df = pd.DataFrame(display_pts)
        else:
            # Legacy mode: rename canonical keys to display dims
            df = pd.DataFrame(data)
            rename = {}
            if x_dim not in df.columns:
                rename["slow_time"] = x_dim
            if y_dim not in df.columns:
                rename["twtt"] = y_dim
            if rename:
                df = df.rename(columns=rename)

        return hv.Points(df, kdims=[x_dim, y_dim]).opts(**self._marker_opts)

    # --- Data access ---

    @property
    def df(self) -> pd.DataFrame:
        """Get picked points as a DataFrame (canonical coordinates)."""
        if not self.points:
            return pd.DataFrame(columns=["id", "slow_time", "twtt"])
        return pd.DataFrame(self.points)

    def clear(self) -> None:
        """Clear all picked points."""
        self.points = []
        self._points_pipe.send([])

    def undo(self) -> None:
        """Remove the last added point."""
        if self.points:
            self.points = self.points[:-1]
            self._points_pipe.send(self.points)

    def delete_by_id(self, point_id: str) -> None:
        """Delete a specific point by ID."""
        self.points = [p for p in self.points if p["id"] != point_id]
        self._points_pipe.send(self.points)

    def to_csv(self, path: str) -> None:
        """Export picked points to CSV (canonical coordinates)."""
        self.df.to_csv(path, index=False)

    def from_csv(self, path: str) -> None:
        """Load picked points from CSV (expects canonical coordinates)."""
        df = pd.read_csv(path)
        self.points = df.to_dict("records")
        self._points_pipe.send(self.points)

    # --- HoloViews integration ---

    def element(self, width: int = 700, height: int = 400) -> hv.Overlay:
        """Get the HoloViews element (image + points overlay)."""
        if self._cached_element is not None and self._cached_size == (width, height):
            return self._cached_element

        image_with_opts = self._image.opts(
            width=width, height=height, tools=["tap"], active_tools=["tap"]
        )

        self._tap_stream = hv.streams.Tap(source=image_with_opts)
        self._tap_stream.add_subscriber(self._on_tap)

        points_dmap = hv.DynamicMap(self._points_element, streams=[self._points_pipe])
        overlay = (image_with_opts * points_dmap).opts(
            width=width, height=height, framewise=True
        )

        self._cached_element = overlay
        self._cached_size = (width, height)

        return overlay

    def panel(self, width: int = 700, height: int = 400) -> pn.Column:
        """Get a Panel layout with the picker and control buttons."""
        undo_btn = pn.widgets.Button(name="Undo", button_type="warning", width=80)
        clear_btn = pn.widgets.Button(name="Clear All", button_type="danger", width=80)
        export_btn = pn.widgets.Button(
            name="Export CSV", button_type="primary", width=80
        )

        export_input = pn.widgets.TextInput(
            name="Export path", value="picks.csv", width=200
        )

        @pn.depends(self.param.points)
        def point_count(points):
            return pn.pane.Markdown(f"**Picks: {len(points)}**")

        def on_undo(event):
            self.undo()

        def on_clear(event):
            self.clear()

        def on_export(event):
            self.to_csv(export_input.value)

        undo_btn.on_click(on_undo)
        clear_btn.on_click(on_clear)
        export_btn.on_click(on_export)

        # Layout
        controls = pn.Row(
            undo_btn, clear_btn, pn.Spacer(width=20), export_input, export_btn
        )
        plot = pn.pane.HoloViews(
            self.element(width=width, height=height),
            width=width,
            height=height,
            sizing_mode="fixed",
        )

        return pn.Column(plot, pn.Row(point_count, pn.Spacer(), controls))


def pick_echogram(
    data: xr.DataArray | hv.Image,
    x_dim: str = "trace",
    y_dim: str = "twtt_us",
    **opts,
) -> GroundingLinePicker:
    """
    Convenience function to create a picker from xarray or holoviews data.

    Parameters
    ----------
    data : xr.DataArray or hv.Image
        The echogram data
    x_dim : str
        X dimension name
    y_dim : str
        Y dimension name
    **opts
        Additional options passed to hv.Image.opts()

    Returns
    -------
    GroundingLinePicker
        The picker instance
    """
    if isinstance(data, hv.Image):
        image = data
    else:
        # Assume xarray DataArray
        image = hv.Image(data, kdims=[x_dim, y_dim])

    default_opts = {"cmap": "gray", "colorbar": True, "tools": ["tap", "hover"]}
    default_opts.update(opts)
    image = image.opts(**default_opts)

    return GroundingLinePicker(image, x_dim=x_dim, y_dim=y_dim)
