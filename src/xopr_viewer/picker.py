"""
Grounding line picking tool using HoloViews streams.

Inspired by CReSIS imb.picker MATLAB tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import uuid4

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

if TYPE_CHECKING:
    import xarray as xr


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
    log_scale: bool = True,
    clim_percentile: tuple[float, float] = (5, 99),
    cmap: str = "gray",
) -> hv.Image:
    """Create HoloViews Image from echogram data."""
    if data_var not in ds:
        raise ValueError(f"Data variable '{data_var}' not found")

    da = ds[data_var]

    if log_scale:
        da = cast("xr.DataArray", 10 * np.log10(np.abs(da) + 1e-10))

    if "twtt" in da.dims and "twtt_us" not in da.coords:
        da = da.assign_coords(twtt_us=("twtt", da.coords["twtt"].values * 1e6))
        da = da.swap_dims({"twtt": "twtt_us"})

    clim = (
        float(np.nanpercentile(da.values, clim_percentile[0])),
        float(np.nanpercentile(da.values, clim_percentile[1])),
    )

    title = _get_title(ds)

    return hv.Image(da, kdims=list(da.dims), vdims=["power_dB"]).opts(
        cmap=cmap,
        colorbar=True,
        clim=clim,
        invert_yaxis=True,
        title=title,
    )


def _create_layer_curves(
    layers: dict[str, xr.Dataset],
    visible_layers: list[str] | None = None,
) -> dict[str, hv.Curve]:
    """Create HoloViews Curve elements for each layer."""
    curves = {}

    for name, layer_ds in layers.items():
        if visible_layers is not None and name not in visible_layers:
            continue

        if "twtt" not in layer_ds:
            continue

        twtt_us = layer_ds.twtt.values * 1e6

        if "slow_time" in layer_ds.coords:
            x_values = layer_ds.slow_time.values
            x_dim = "slow_time"
        else:
            x_values = np.arange(len(twtt_us))
            x_dim = "trace"

        color = _LAYER_COLORS.get(name, _DEFAULT_LAYER_COLOR)

        curves[name] = hv.Curve(
            (x_values, twtt_us),
            kdims=[x_dim],
            vdims=["twtt_us"],
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

    Returns
    -------
    dict[str, hv.Curve]
        Mapping of layer name to slope Curve element.
    """
    curves = {}

    for name, layer_ds in layers.items():
        if visible_layers is not None and name not in visible_layers:
            continue

        if "twtt" not in layer_ds:
            continue

        slope = compute_layer_slope(layer_ds.twtt, smoothing_window)
        dim = slope.dims[0]

        color = _LAYER_COLORS.get(name, _DEFAULT_LAYER_COLOR)

        curves[name] = hv.Curve(
            slope,
            kdims=[dim],
            vdims=["twtt"],
            label=f"{name} slope",
        ).opts(
            color=color,
            line_width=2,
            alpha=0.8,
        )

    return curves


class GroundingLinePicker(param.Parameterized):
    """
    Interactive point picker for echogram annotation.

    Click on the image to add points. Points are stored in a list
    and can be exported as a DataFrame or CSV for ML training.

    Inspired by CReSIS imb.picker MATLAB tool.

    Parameters
    ----------
    image : hv.Image
        The echogram image to annotate
    x_dim : str
        Name of the x dimension (default: inferred from image)
    y_dim : str
        Name of the y dimension (default: inferred from image)
    layers : dict, optional
        Dictionary of layer name -> xr.Dataset for snap-to-layer functionality
    snap_threshold : float
        Snap threshold in microseconds (default: 5.0)
    x_coord : str
        Name of the x coordinate in layer datasets (default: "slow_time")

    Attributes
    ----------
    snap_enabled : bool
        Whether snapping to layers is enabled (default: False). Set this to True
        to enable snap-to-layer functionality when clicking.

    Example
    -------

    ```python
    import holoviews as hv
    import numpy as np
    hv.extension("bokeh")

    # Create sample echogram
    data = np.random.rand(100, 200)
    image = hv.Image(data, kdims=["trace", "twtt_us"])

    # Create picker
    picker = GroundingLinePicker(image)
    picker.panel()  # Display in notebook

    # After clicking some points:
    picker.df  # Get the DataFrame
    picker.to_csv("picks.csv")  # Export
    ```
    """

    points = param.List(default=[], doc="List of picked points as dicts")
    snap_enabled = param.Boolean(
        default=False, doc="Whether snapping to layers is enabled"
    )

    def __init__(
        self,
        image: hv.Image,
        x_dim: str | None = None,
        y_dim: str | None = None,
        layers: dict[str, xr.Dataset] | None = None,
        snap_threshold: float = 5.0,
        x_coord: str = "slow_time",
        **params,
    ) -> None:
        """
        Parameters
        ----------
        image : hv.Image
            The echogram image to annotate
        x_dim : str
            Name of the x dimension (default: inferred from image)
        y_dim : str
            Name of the y dimension (default: inferred from image)
        layers : dict, optional
            Dictionary of layer name -> xr.Dataset for snap-to-layer functionality
        snap_threshold : float
            Snap threshold in microseconds (default: 5.0)
        x_coord : str
            Name of the x coordinate in layer datasets (default: "slow_time")
        """
        super().__init__(**params)

        self._image = image
        # Try to get kdims - handle both hv.Image and hvplot outputs
        if hasattr(image, "kdims") and image.kdims:
            self.x_dim = x_dim or str(image.kdims[0])
            self.y_dim = y_dim or str(image.kdims[1])
        else:
            self.x_dim = x_dim or "x"
            self.y_dim = y_dim or "y"

        # Marker styling
        self._marker_opts = params.get(
            "marker_opts",
            {"color": "red", "size": 12, "marker": "cross", "line_width": 2},
        )

        # Layer snapping configuration
        self._layers = layers or {}
        self._visible_layers: list[str] | None = None  # None means all layers
        self._snap_threshold = snap_threshold
        self._x_coord = x_coord

        # Points data source for dynamic updates
        self._points_pipe = hv.streams.Pipe(data=[])

        # Will be set up when element() is called
        self._tap_stream: hv.streams.Tap | None = None
        self._cached_element: hv.Overlay | None = None
        self._cached_size: tuple[int, int] | None = None

    @property
    def visible_layers(self) -> list[str] | None:
        """Layer names to snap to, or None for all layers."""
        return self._visible_layers

    @visible_layers.setter
    def visible_layers(self, value: list[str] | None) -> None:
        self._visible_layers = value

    def _snap_to_layer(self, x: float, y: float) -> tuple[float, float]:
        """Snap coordinates to nearest layer if within threshold."""
        if not self._layers:
            return x, y

        # Determine which layers to check
        layers_to_check = self._layers
        if self._visible_layers is not None:
            layers_to_check = {
                k: v for k, v in self._layers.items() if k in self._visible_layers
            }

        # Filter to layers that have twtt data
        valid_layers = {
            name: layer_ds
            for name, layer_ds in layers_to_check.items()
            if "twtt" in layer_ds
        }

        if not valid_layers:
            return x, y

        # Find nearest layer value at this x
        best_y = y
        best_dist = self._snap_threshold + 1  # Start above threshold

        for layer_ds in valid_layers.values():
            # Use xarray's sel with method='nearest' to handle any coordinate type
            try:
                nearest = layer_ds.twtt.sel({self._x_coord: x}, method="nearest")
                layer_y = float(nearest.values) * 1e6  # Convert to microseconds
            except KeyError:
                # x_coord not in dataset, fall back to first value
                layer_y = float(layer_ds.twtt.values[0]) * 1e6

            dist = abs(y - layer_y)
            if dist < best_dist:
                best_dist = dist
                best_y = layer_y

        # Snap if within threshold
        if best_dist <= self._snap_threshold:
            return x, best_y
        return x, y

    def _on_tap(self, x: float | None, y: float | None) -> None:
        """Handle tap events - add point to list."""
        if x is not None and y is not None:
            # Apply snap if enabled
            if self.snap_enabled and self._layers:
                x, y = self._snap_to_layer(x, y)

            point = {
                "id": str(uuid4())[:8],
                self.x_dim: x,
                self.y_dim: y,
            }
            self.points = [*self.points, point]
            self._points_pipe.send(self.points)

    def _points_element(self, data: list[dict]) -> hv.Points:
        """Create Points element from current data."""
        if not data:
            return hv.Points([], kdims=[self.x_dim, self.y_dim]).opts(
                **self._marker_opts
            )
        df = pd.DataFrame(data)
        return hv.Points(df, kdims=[self.x_dim, self.y_dim]).opts(**self._marker_opts)

    @property
    def df(self) -> pd.DataFrame:
        """Get picked points as a DataFrame."""
        if not self.points:
            return pd.DataFrame(columns=["id", self.x_dim, self.y_dim])
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
        """Export picked points to CSV."""
        self.df.to_csv(path, index=False)

    def from_csv(self, path: str) -> None:
        """Load picked points from CSV."""
        df = pd.read_csv(path)
        self.points = df.to_dict("records")
        self._points_pipe.send(self.points)

    def element(self, width: int = 700, height: int = 400) -> hv.Overlay:
        """Get the HoloViews element (image + points overlay)."""
        # Return cached element if size matches
        if self._cached_element is not None and self._cached_size == (width, height):
            return self._cached_element

        # Apply size opts to image
        image_with_opts = self._image.opts(
            width=width, height=height, tools=["tap"], active_tools=["tap"]
        )

        # Set up tap stream on the image BEFORE creating overlay
        self._tap_stream = hv.streams.Tap(source=image_with_opts)
        self._tap_stream.add_subscriber(self._on_tap)

        # Create points DynamicMap and overlay
        points_dmap = hv.DynamicMap(self._points_element, streams=[self._points_pipe])
        overlay = (image_with_opts * points_dmap).opts(
            width=width, height=height, framewise=True
        )

        # Cache the element
        self._cached_element = overlay
        self._cached_size = (width, height)

        return overlay

    def panel(self, width: int = 700, height: int = 400) -> pn.Column:
        """
        Get a Panel layout with the picker and control buttons.

        Returns a Column with:
        - The interactive plot
        - Undo/Clear buttons
        - Point count display
        """
        # Buttons
        undo_btn = pn.widgets.Button(name="Undo", button_type="warning", width=80)
        clear_btn = pn.widgets.Button(name="Clear All", button_type="danger", width=80)
        export_btn = pn.widgets.Button(
            name="Export CSV", button_type="primary", width=80
        )

        # File input for export path
        export_input = pn.widgets.TextInput(
            name="Export path", value="picks.csv", width=200
        )

        # Point counter
        @pn.depends(self.param.points)
        def point_count(points):
            return pn.pane.Markdown(f"**Picks: {len(points)}**")

        # Button callbacks
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
