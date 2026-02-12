"""Xarray accessor for interactive echogram picking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import panel as pn
import xarray as xr

from xopr_viewer.picker import (
    GroundingLinePicker,
    _create_image,
    _create_layer_curves,
    _create_slope_curves,
)

if TYPE_CHECKING:
    from typing import Any


@xr.register_dataset_accessor("pick")
class PickAccessor:
    """
    Xarray accessor for echogram visualization and picking.

    Examples
    --------
    >>> frame.pick.plot(layers=layers)  # Static plot
    >>> frame.pick.show(layers=layers)  # Interactive picker in browser
    """

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def plot(
        self,
        layers: dict[str, xr.Dataset] | None = None,
        x_mode: str = "gps_time",
        y_mode: str = "twtt",
        width: int = 900,
        height: int = 500,
        **image_opts: Any,
    ) -> hv.Element:
        """Create echogram plot with optional layer overlays."""
        image = _create_image(self._obj, x_mode=x_mode, y_mode=y_mode, **image_opts)
        plot = image.opts(width=width, height=height)

        if layers:
            for curve in _create_layer_curves(
                layers, ds=self._obj, x_mode=x_mode, y_mode=y_mode
            ).values():
                plot = plot * curve

        return plot

    def picker(
        self,
        layers: dict[str, xr.Dataset] | None = None,
        x_mode: str = "gps_time",
        y_mode: str = "twtt",
        **image_opts: Any,
    ) -> GroundingLinePicker:
        """Create a GroundingLinePicker for interactive point selection."""
        image = _create_image(self._obj, x_mode=x_mode, y_mode=y_mode, **image_opts)
        return GroundingLinePicker(
            image, ds=self._obj, x_mode=x_mode, y_mode=y_mode, layers=layers
        )

    def panel(
        self,
        layers: dict[str, xr.Dataset] | None = None,
        width: int = 900,
        height: int = 500,
        slope_height: int = 150,
        **image_opts: Any,
    ) -> pn.Row:
        """Create interactive Panel layout with picker, layer overlays, and slope subplot."""
        ds = self._obj

        # --- Axis mode dropdowns (available modes depend on dataset) ---
        x_options = {"Range-line": "rangeline", "GPS time": "gps_time"}
        if "Latitude" in ds and "Longitude" in ds:
            x_options["Along-track (km)"] = "along_track"

        y_options = {
            "TWTT (\u00b5s)": "twtt",
            "Range-bin": "range_bin",
            "Range (m)": "range",
        }
        if "Surface" in ds and "Elevation" in ds:
            y_options["WGS-84 Elevation (m)"] = "elevation"
        if "Surface" in ds:
            y_options["Surface-flat (m)"] = "surface_flat"

        x_mode_select = pn.widgets.Select(
            name="X axis", options=x_options, value="gps_time", width=180
        )
        y_mode_select = pn.widgets.Select(
            name="Y axis", options=y_options, value="twtt", width=180
        )

        # --- Picker (with dataset for coordinate conversion) ---
        picker = GroundingLinePicker(
            image=_create_image(ds, x_mode="gps_time", y_mode="twtt", **image_opts),
            ds=ds,
            x_mode="gps_time",
            y_mode="twtt",
            layers=layers,
        )

        # --- Echogram overlay builder ---
        # Uses pn.bind instead of DynamicMap streams so that the entire
        # Bokeh plot is rebuilt when axis modes change (Bokeh cannot
        # switch between DatetimeAxis and LinearAxis on the same plot).
        def make_echogram(x_mode, y_mode, visible_layers=None):
            picker.set_axis_modes(x_mode, y_mode, _notify=False)

            image = _create_image(ds, x_mode=x_mode, y_mode=y_mode, **image_opts)
            image = image.opts(
                width=width,
                height=height,
                tools=["tap"],
                active_tools=["tap"],
            )

            tap = hv.streams.Tap(source=image)
            tap.add_subscriber(picker._on_tap)

            points_dmap = hv.DynamicMap(
                picker._points_element, streams=[picker._points_pipe]
            )

            overlay = image * points_dmap
            if layers and visible_layers:
                for curve in _create_layer_curves(
                    layers, visible_layers, ds=ds, x_mode=x_mode, y_mode=y_mode
                ).values():
                    overlay = overlay * curve

            return overlay.opts(width=width, height=height, framewise=True)

        # --- Layer overlays and slope ---
        if layers:
            layer_names = list(layers.keys())
            layer_checkboxes = pn.widgets.CheckBoxGroup(
                name="Layers", options=layer_names, value=layer_names
            )
            snap_checkbox = pn.widgets.Checkbox(name="Snap to layer", value=False)

            slope_checkboxes = pn.widgets.CheckBoxGroup(
                name="Slope Layers", options=layer_names, value=[]
            )
            smoothing_slider = pn.widgets.IntSlider(
                name="Smoothing Window",
                start=1,
                end=51,
                step=2,
                value=1,
                width=180,
            )

            def update_snap(event):
                picker.snap_enabled = snap_checkbox.value
                picker.visible_layers = (
                    layer_checkboxes.value if layer_checkboxes.value else None
                )

            snap_checkbox.param.watch(update_snap, "value")
            layer_checkboxes.param.watch(update_snap, "value")

            _sample_layer = next(iter(layers.values()))
            _empty_x = _sample_layer.slow_time.values[:1]

            def slope_overlay(visible_slopes, smoothing_window):
                if not visible_slopes:
                    return hv.Curve(
                        (_empty_x, [float("nan")]),
                        kdims=["slow_time"],
                        vdims=["twtt"],
                    ).opts(
                        width=width,
                        height=slope_height,
                        title="Layer Slope",
                        ylabel="Slope (\u00b5s/trace)",
                    )
                curves = _create_slope_curves(layers, visible_slopes, smoothing_window)
                return hv.Overlay(list(curves.values())).opts(
                    width=width,
                    height=slope_height,
                    title="Layer Slope",
                    ylabel="Slope (\u00b5s/trace)",
                )

            echogram_pane = pn.pane.HoloViews(
                pn.bind(make_echogram, x_mode_select, y_mode_select, layer_checkboxes),
                width=width,
                height=height,
                sizing_mode="fixed",
            )
            slope_pane = pn.pane.HoloViews(
                pn.bind(slope_overlay, slope_checkboxes, smoothing_slider),
                width=width,
                height=slope_height,
                sizing_mode="fixed",
            )
            layer_sidebar = [
                pn.pane.Markdown("### Layers"),
                layer_checkboxes,
                snap_checkbox,
                pn.layout.Divider(),
                pn.pane.Markdown("### Slope"),
                slope_checkboxes,
                smoothing_slider,
            ]
        else:
            echogram_pane = pn.pane.HoloViews(
                pn.bind(make_echogram, x_mode_select, y_mode_select),
                width=width,
                height=height,
                sizing_mode="fixed",
            )
            slope_pane = None
            layer_sidebar = []

        # --- Sidebar ---
        sidebar = pn.Column(
            pn.pane.Markdown("### Display"),
            x_mode_select,
            y_mode_select,
            pn.layout.Divider(),
            *layer_sidebar,
            width=200,
        )

        # --- Controls ---
        @pn.depends(picker.param.points)
        def point_count(points):
            return pn.pane.Markdown(f"**Picks: {len(points)}**")

        undo_btn = pn.widgets.Button(name="Undo", button_type="warning", width=80)
        clear_btn = pn.widgets.Button(name="Clear", button_type="danger", width=80)
        export_input = pn.widgets.TextInput(
            value=f"{ds.attrs.get('granule', 'picks')}.csv", width=200
        )
        export_btn = pn.widgets.Button(name="Export", button_type="primary", width=80)

        undo_btn.on_click(lambda e: picker.undo())
        clear_btn.on_click(lambda e: picker.clear())
        export_btn.on_click(lambda e: picker.to_csv(export_input.value))

        controls = pn.Row(
            point_count, pn.Spacer(), undo_btn, clear_btn, export_input, export_btn
        )

        main_children = [echogram_pane]
        if slope_pane is not None:
            main_children.append(slope_pane)
        main_children.append(controls)

        main = pn.Column(*main_children, sizing_mode="fixed", width=width)

        return pn.Row(sidebar, main)

    def show(self, layers: dict[str, xr.Dataset] | None = None, **opts: Any) -> None:
        """Open picker in browser tab."""
        self.panel(layers=layers, **opts).show()
