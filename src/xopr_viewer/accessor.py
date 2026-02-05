"""Xarray accessor for interactive echogram picking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv
import panel as pn
import xarray as xr

if TYPE_CHECKING:
    from typing import Any

from xopr_viewer.picker import (
    GroundingLinePicker,
    _create_image,
    _create_layer_curves,
    _create_slope_curves,
)


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
        width: int = 900,
        height: int = 500,
        **image_opts: Any,
    ) -> hv.Element:
        """Create echogram plot with optional layer overlays."""
        image = _create_image(self._obj, **image_opts)
        plot = image.opts(width=width, height=height)

        if layers:
            for curve in _create_layer_curves(layers).values():
                plot = plot * curve

        return plot

    def picker(
        self,
        layers: dict[str, xr.Dataset] | None = None,
        **image_opts: Any,
    ) -> GroundingLinePicker:
        """Create a GroundingLinePicker for interactive point selection."""
        image = _create_image(self._obj, **image_opts)
        return GroundingLinePicker(
            image, x_dim="slow_time", y_dim="twtt_us", layers=layers
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
        picker = self.picker(layers=layers, **image_opts)
        base_plot = picker.element(width=width, height=height)

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

            # Update snap settings when checkboxes change
            def update_snap(event):
                picker.snap_enabled = snap_checkbox.value
                picker.visible_layers = (
                    layer_checkboxes.value if layer_checkboxes.value else None
                )

            snap_checkbox.param.watch(update_snap, "value")
            layer_checkboxes.param.watch(update_snap, "value")

            @pn.depends(layer_checkboxes.param.value)
            def layer_overlay(value):
                curves = _create_layer_curves(layers, value)
                return hv.Overlay(list(curves.values()))

            # Get a sample slow_time value to type the empty curve axis
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
                base_plot * hv.DynamicMap(layer_overlay),
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
            sidebar = pn.Column(
                pn.pane.Markdown("### Layers"),
                layer_checkboxes,
                snap_checkbox,
                pn.layout.Divider(),
                pn.pane.Markdown("### Slope"),
                slope_checkboxes,
                smoothing_slider,
                width=200,
            )
        else:
            echogram_pane = pn.pane.HoloViews(
                base_plot, width=width, height=height, sizing_mode="fixed"
            )
            slope_pane = None
            sidebar = pn.Column(width=200)

        @pn.depends(picker.param.points)
        def point_count(points):
            return pn.pane.Markdown(f"**Picks: {len(points)}**")

        undo_btn = pn.widgets.Button(name="Undo", button_type="warning", width=80)
        clear_btn = pn.widgets.Button(name="Clear", button_type="danger", width=80)
        export_input = pn.widgets.TextInput(
            value=f"{self._obj.attrs.get('granule', 'picks')}.csv", width=200
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
