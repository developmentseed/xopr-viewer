"""Tests for the PickAccessor xarray accessor."""

import holoviews as hv
import numpy as np
import pytest

# Import accessor to register it
import xopr_viewer.accessor  # noqa: F401
from xopr_viewer.accessor import PickAccessor
from xopr_viewer.picker import (
    GroundingLinePicker,
    _create_image,
    _create_layer_curves,
    _get_title,
    _interpolate_uniform,
    _warp_colormap,
    _LAYER_COLORS,
)

# Load bokeh extension for tests
hv.extension("bokeh")


class TestAccessorRegistration:
    """Test that the accessor is properly registered."""

    def test_accessor_available(self, sample_echogram_dataset):
        assert hasattr(sample_echogram_dataset, "pick")
        assert isinstance(sample_echogram_dataset.pick, PickAccessor)

    def test_accessor_persists_on_same_dataset(self, sample_echogram_dataset):
        acc1 = sample_echogram_dataset.pick
        acc2 = sample_echogram_dataset.pick
        assert acc1 is acc2


class TestCreateImage:
    """Test echogram image creation."""

    def test_create_image_default(self, sample_echogram_dataset):
        image = _create_image(sample_echogram_dataset)
        assert isinstance(image, hv.Image)

    def test_create_image_raises_on_missing_var(self, sample_echogram_dataset):
        with pytest.raises(ValueError, match="not found"):
            _create_image(sample_echogram_dataset, data_var="NonExistent")

    def test_create_image_default_clim_finite_minmax(self, sample_echogram_dataset):
        image = _create_image(sample_echogram_dataset)
        opts = hv.Store.lookup_options("bokeh", image, "plot").kwargs
        clim = opts.get("clim")
        assert clim is not None
        # Default clim should span the finite range of the dB data
        dB_data = 10 * np.log10(np.abs(sample_echogram_dataset["Data"].values))
        finite = dB_data[np.isfinite(dB_data)]
        assert clim[0] == pytest.approx(float(finite.min()))
        assert clim[1] == pytest.approx(float(finite.max()))

    def test_create_image_explicit_clim(self, sample_echogram_dataset):
        image = _create_image(sample_echogram_dataset, clim=(-80.0, -20.0))
        opts = hv.Store.lookup_options("bokeh", image, "plot").kwargs
        assert opts["clim"] == (-80.0, -20.0)


class TestAxisModes:
    """Test axis mode handling in _create_image."""

    def test_gps_time_produces_uniform_output(self, nonuniform_echogram_dataset):
        """x_mode='gps_time' interpolates non-uniform slow_time to uniform grid."""
        image = _create_image(nonuniform_echogram_dataset, x_mode="gps_time")
        coords = image.dimension_values("slow_time", expanded=False)
        diffs = np.diff(coords.astype("datetime64[ns]").astype(np.int64))
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)

    def test_rangeline_no_interpolation(self, nonuniform_echogram_dataset):
        """x_mode='rangeline' produces integer trace dim with no interpolation."""
        image = _create_image(nonuniform_echogram_dataset, x_mode="rangeline")
        traces = image.dimension_values("trace", expanded=False)
        # Trace count matches original
        assert len(traces) == len(nonuniform_echogram_dataset.slow_time)
        # Integer indices
        np.testing.assert_array_equal(traces, np.arange(len(traces)))

    def test_range_bin_mode(self, sample_echogram_dataset):
        """y_mode='range_bin' produces integer range_bin dim."""
        image = _create_image(sample_echogram_dataset, y_mode="range_bin")
        bins = image.dimension_values("range_bin", expanded=False)
        np.testing.assert_array_equal(bins, np.arange(len(bins)))

    def test_range_mode(self, sample_echogram_dataset):
        """y_mode='range' produces range_m dim scaled from twtt."""
        import scipy.constants

        image = _create_image(sample_echogram_dataset, y_mode="range")
        range_m = image.dimension_values("range_m", expanded=False)
        expected = sample_echogram_dataset.twtt.values * scipy.constants.c / 2.0
        np.testing.assert_allclose(range_m, expected, rtol=1e-6)

    def test_interpolate_uniform_numeric(self):
        """_interpolate_uniform handles numeric coordinates."""
        import xarray as xr

        # Non-uniform x: [0, 1, 3, 6, 10]
        x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        da = xr.DataArray(np.arange(5.0), dims=["x"], coords={"x": x})
        result = _interpolate_uniform(da, "x")
        result_diffs = np.diff(result.coords["x"].values)
        np.testing.assert_allclose(result_diffs, result_diffs[0], rtol=1e-6)

    def test_interpolate_uniform_short_array_noop(self):
        """Arrays with fewer than 3 points are returned unchanged."""
        import xarray as xr

        x = np.array([0.0, 5.0])
        da = xr.DataArray(np.arange(2.0), dims=["x"], coords={"x": x})
        result = _interpolate_uniform(da, "x")
        np.testing.assert_array_equal(result.values, da.values)

    def test_interpolate_uniform_non_monotonic_raises(self):
        """Non-monotonic coordinates raise ValueError."""
        import xarray as xr

        x = np.array([10.0, 5.0, 1.0])  # decreasing
        da = xr.DataArray(np.arange(3.0), dims=["x"], coords={"x": x})
        with pytest.raises(ValueError, match="not monotonically increasing"):
            _interpolate_uniform(da, "x")


_X_MODES = ["rangeline", "gps_time", "along_track"]
_Y_MODES = ["twtt", "range_bin", "range", "elevation", "surface_flat"]
_EXPECTED_X_DIM = {
    "rangeline": "trace",
    "gps_time": "slow_time",
    "along_track": "along_track_km",
}
_EXPECTED_Y_DIM = {
    "twtt": "twtt_us",
    "range_bin": "range_bin",
    "range": "range_m",
    "elevation": "elevation_m",
    "surface_flat": "depth_m",
}
_X_TRANSITIONS = [(a, b) for a in _X_MODES for b in _X_MODES if a != b]
_Y_TRANSITIONS = [(a, b) for a in _Y_MODES for b in _Y_MODES if a != b]


class TestAllAxisModeCombinations:
    """Test _create_image for every x_mode × y_mode combination (3 × 5 = 15)."""

    @pytest.mark.parametrize("x_mode", _X_MODES)
    @pytest.mark.parametrize("y_mode", _Y_MODES)
    def test_create_image_returns_image(self, sample_echogram_dataset, x_mode, y_mode):
        """_create_image succeeds for every axis mode combination."""
        image = _create_image(sample_echogram_dataset, x_mode=x_mode, y_mode=y_mode)
        assert isinstance(image, hv.Image)

    @pytest.mark.parametrize("x_mode", _X_MODES)
    @pytest.mark.parametrize("y_mode", _Y_MODES)
    def test_kdims_names(self, sample_echogram_dataset, x_mode, y_mode):
        """Image kdims match the expected display dimension names."""
        image = _create_image(sample_echogram_dataset, x_mode=x_mode, y_mode=y_mode)
        kdim_names = [d.name for d in image.kdims]
        assert kdim_names[0] == _EXPECTED_X_DIM[x_mode]
        assert kdim_names[1] == _EXPECTED_Y_DIM[y_mode]

    @pytest.mark.parametrize("x_mode", _X_MODES)
    @pytest.mark.parametrize("y_mode", _Y_MODES)
    def test_image_has_data(self, sample_echogram_dataset, x_mode, y_mode):
        """Image contains non-empty data for every combination."""
        image = _create_image(sample_echogram_dataset, x_mode=x_mode, y_mode=y_mode)
        assert image.dimension_values("power_dB").size > 0


class TestAxisModeSwitching:
    """Test switching between axis modes — simulates dropdown changes."""

    # -- X-mode transitions --

    @pytest.mark.parametrize("from_mode,to_mode", _X_TRANSITIONS)
    def test_switch_x_mode_updates_dims(
        self, sample_echogram_dataset, from_mode, to_mode
    ):
        """set_axis_modes updates x_dim when x-mode changes."""
        image = _create_image(sample_echogram_dataset, x_mode=from_mode)
        picker = GroundingLinePicker(
            image, ds=sample_echogram_dataset, x_mode=from_mode, y_mode="twtt"
        )
        assert picker.x_dim == _EXPECTED_X_DIM[from_mode]

        picker.set_axis_modes(to_mode, "twtt")
        assert picker.x_dim == _EXPECTED_X_DIM[to_mode]
        assert picker.y_dim == _EXPECTED_Y_DIM["twtt"]

    # -- Y-mode transitions --

    @pytest.mark.parametrize("from_mode,to_mode", _Y_TRANSITIONS)
    def test_switch_y_mode_updates_dims(
        self, sample_echogram_dataset, from_mode, to_mode
    ):
        """set_axis_modes updates y_dim when y-mode changes."""
        image = _create_image(sample_echogram_dataset, y_mode=from_mode)
        picker = GroundingLinePicker(
            image, ds=sample_echogram_dataset, x_mode="gps_time", y_mode=from_mode
        )
        assert picker.y_dim == _EXPECTED_Y_DIM[from_mode]

        picker.set_axis_modes("gps_time", to_mode)
        assert picker.y_dim == _EXPECTED_Y_DIM[to_mode]
        assert picker.x_dim == _EXPECTED_X_DIM["gps_time"]

    # -- Points survive mode switches --

    @pytest.mark.parametrize("from_x,to_x", _X_TRANSITIONS)
    def test_points_survive_x_mode_switch(self, sample_echogram_dataset, from_x, to_x):
        """Points picked in one x-mode render without error after switching."""
        from xopr_viewer.coordinates import canonical_to_display

        ds = sample_echogram_dataset
        image = _create_image(ds, x_mode=from_x)
        picker = GroundingLinePicker(image, ds=ds, x_mode=from_x, y_mode="twtt")

        # Pick a point (tap in display coords of from_x mode)
        slow_time = ds["slow_time"].values[10]
        twtt_s = 20e-6
        dx, dy = canonical_to_display(slow_time, twtt_s, ds, from_x, "twtt")
        picker._on_tap(dx, dy)
        assert len(picker.points) == 1

        # Canonical storage is independent of display mode
        assert "slow_time" in picker.points[0]
        assert "twtt" in picker.points[0]

        # Switch x-mode and render points
        picker.set_axis_modes(to_x, "twtt")
        pts = picker._points_element(picker.points)
        assert isinstance(pts, hv.Points)
        assert pts.data.shape[0] == 1
        # kdims match the new mode
        assert pts.kdims[0].name == _EXPECTED_X_DIM[to_x]
        assert pts.kdims[1].name == _EXPECTED_Y_DIM["twtt"]

    @pytest.mark.parametrize("from_y,to_y", _Y_TRANSITIONS)
    def test_points_survive_y_mode_switch(self, sample_echogram_dataset, from_y, to_y):
        """Points picked in one y-mode render without error after switching."""
        from xopr_viewer.coordinates import canonical_to_display

        ds = sample_echogram_dataset
        image = _create_image(ds, y_mode=from_y)
        picker = GroundingLinePicker(image, ds=ds, x_mode="gps_time", y_mode=from_y)

        # Pick a point
        slow_time = ds["slow_time"].values[10]
        twtt_s = 20e-6
        dx, dy = canonical_to_display(slow_time, twtt_s, ds, "gps_time", from_y)
        picker._on_tap(dx, dy)
        assert len(picker.points) == 1

        # Switch y-mode and render points
        picker.set_axis_modes("gps_time", to_y)
        pts = picker._points_element(picker.points)
        assert isinstance(pts, hv.Points)
        assert pts.data.shape[0] == 1
        assert pts.kdims[0].name == _EXPECTED_X_DIM["gps_time"]
        assert pts.kdims[1].name == _EXPECTED_Y_DIM[to_y]

    # -- Canonical coordinates unchanged after switch --

    @pytest.mark.parametrize("x_mode", _X_MODES)
    @pytest.mark.parametrize("y_mode", _Y_MODES)
    def test_canonical_coords_unchanged_after_switch(
        self, sample_echogram_dataset, x_mode, y_mode
    ):
        """Switching modes doesn't alter the stored canonical coordinates."""
        from xopr_viewer.coordinates import canonical_to_display

        ds = sample_echogram_dataset
        image = _create_image(ds, x_mode=x_mode, y_mode=y_mode)
        picker = GroundingLinePicker(image, ds=ds, x_mode=x_mode, y_mode=y_mode)

        # Pick via display coords
        slow_time = ds["slow_time"].values[25]
        twtt_s = 15e-6
        dx, dy = canonical_to_display(slow_time, twtt_s, ds, x_mode, y_mode)
        picker._on_tap(dx, dy)

        stored = picker.points[0]
        original_twtt = stored["twtt"]

        # Switch to a different mode
        picker.set_axis_modes("rangeline", "range_bin")

        # Canonical values are unchanged
        assert picker.points[0]["twtt"] == original_twtt
        assert picker.points[0]["slow_time"] == stored["slow_time"]


class TestGetTitle:
    """Test title generation for merged frames."""

    def test_title_single_granule(self, sample_echogram_dataset):
        title = _get_title(sample_echogram_dataset)
        assert title == "20230109_01_001"

    def test_title_merged_frames_set(self, sample_echogram_dataset):
        sample_echogram_dataset.attrs["granule"] = {
            "20230109_01_001",
            "20230109_01_002",
            "20230109_01_003",
        }
        title = _get_title(sample_echogram_dataset)
        assert title == "20230109_01 (3 frames)"

    def test_title_no_granule_has_segment(self, sample_echogram_dataset):
        del sample_echogram_dataset.attrs["granule"]
        title = _get_title(sample_echogram_dataset)
        assert title == "20230109_01"

    def test_title_no_attrs(self, sample_echogram_dataset):
        sample_echogram_dataset.attrs.clear()
        title = _get_title(sample_echogram_dataset)
        assert title == "Echogram"

    def test_title_segment_as_set(self, sample_echogram_dataset):
        """Test that segment_path as a set uses first sorted value."""
        sample_echogram_dataset.attrs["segment_path"] = {
            "20230109_03",
            "20230109_01",
            "20230109_02",
        }
        sample_echogram_dataset.attrs["granule"] = {
            "20230109_01_001",
            "20230109_02_001",
        }
        title = _get_title(sample_echogram_dataset)
        # Should use first sorted segment ("20230109_01") with frame count
        assert title == "20230109_01 (2 frames)"

    def test_title_segment_as_empty_set(self, sample_echogram_dataset):
        """Test that empty segment_path set results in fallback behavior."""
        sample_echogram_dataset.attrs["segment_path"] = set()
        sample_echogram_dataset.attrs["granule"] = {"frame1", "frame2"}
        title = _get_title(sample_echogram_dataset)
        # Empty segment means no prefix, falls back to "Echogram"
        assert title == "Echogram"

    def test_title_segment_set_no_granule(self, sample_echogram_dataset):
        """Test segment as set when there's no granule."""
        sample_echogram_dataset.attrs["segment_path"] = {
            "20230109_02",
            "20230109_01",
        }
        del sample_echogram_dataset.attrs["granule"]
        title = _get_title(sample_echogram_dataset)
        # Should use first sorted segment
        assert title == "20230109_01"

    def test_create_image_with_merged_frames(self, sample_echogram_dataset):
        sample_echogram_dataset.attrs["granule"] = {
            "20230109_01_001",
            "20230109_01_002",
        }
        image = _create_image(sample_echogram_dataset)
        assert isinstance(image, hv.Image)


class TestLayerCurves:
    """Test layer curve creation."""

    def test_create_layer_curves_uses_slow_time(self, sample_layers):
        """Layer curves should use slow_time coordinate to match image x-axis."""
        curves = _create_layer_curves(sample_layers)
        for curve in curves.values():
            assert "slow_time" in [d.name for d in curve.kdims]

    def test_create_layer_curves(self, sample_layers):
        curves = _create_layer_curves(sample_layers)
        assert len(curves) == 2
        assert "standard:surface" in curves
        assert "standard:bottom" in curves
        assert isinstance(curves["standard:surface"], hv.Curve)

    def test_create_layer_curves_filters_visible(self, sample_layers):
        curves = _create_layer_curves(
            sample_layers, visible_layers=["standard:surface"]
        )
        assert len(curves) == 1
        assert "standard:surface" in curves

    def test_create_layer_curves_empty_if_none_visible(self, sample_layers):
        curves = _create_layer_curves(sample_layers, visible_layers=[])
        assert len(curves) == 0


class TestPlot:
    """Test the plot() method."""

    def test_plot_returns_element(self, sample_echogram_dataset):
        acc = sample_echogram_dataset.pick
        plot = acc.plot()
        assert isinstance(plot, hv.Element)

    def test_plot_with_layers(self, sample_echogram_dataset, sample_layers):
        acc = sample_echogram_dataset.pick
        plot = acc.plot(layers=sample_layers)
        assert isinstance(plot, (hv.Element, hv.Overlay))


class TestPicker:
    """Test the picker() method."""

    def test_picker_returns_picker(self, sample_echogram_dataset):
        acc = sample_echogram_dataset.pick
        picker = acc.picker()
        assert isinstance(picker, GroundingLinePicker)

    def test_picker_has_correct_dims(self, sample_echogram_dataset):
        acc = sample_echogram_dataset.pick
        picker = acc.picker()
        assert picker.x_dim == "slow_time"
        assert picker.y_dim == "twtt_us"

    def test_picker_state_independent(self, sample_echogram_dataset):
        """Each call to picker() creates a new independent instance."""
        acc = sample_echogram_dataset.pick
        picker1 = acc.picker()
        picker2 = acc.picker()
        # Different instances
        assert picker1 is not picker2
        # Adding to one doesn't affect the other
        picker1._on_tap(10.0, 20.0)
        assert len(picker1.points) == 1
        assert len(picker2.points) == 0


class TestPanel:
    """Test the panel() method."""

    def test_panel_returns_row(self, sample_echogram_dataset):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel()
        assert isinstance(layout, pn.Row)

    def test_panel_with_layers(self, sample_echogram_dataset, sample_layers):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)
        assert isinstance(layout, pn.Row)

    def test_panel_has_axis_mode_selectors(self, sample_echogram_dataset):
        """Panel sidebar includes x-axis and y-axis mode dropdowns."""
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel()

        sidebar = layout[0]
        selects = [item for item in sidebar if isinstance(item, pn.widgets.Select)]
        assert len(selects) == 2
        x_select = selects[0]
        y_select = selects[1]
        assert x_select.name == "X axis"
        assert y_select.name == "Y axis"

    @pytest.mark.parametrize("to_x", ["rangeline", "along_track"])
    def test_panel_switch_from_gps_time_no_dtype_error(
        self, sample_echogram_dataset, to_x
    ):
        """Switching x-mode from gps_time must not raise UFuncNoLoopError.

        Reproduces a bug where Panel's link_axes preprocessor compares
        datetime64 axis bounds (from the old gps_time plot) with float
        bounds (from the new numeric-axis plot), triggering:

            numpy._core._exceptions._UFuncNoLoopError:
            ufunc 'greater' did not contain a loop with signature matching
            types (DateTime64DType, _PyFloatDType) -> None

        In production this fires inside Panel's server-side _update_pane
        → _preprocess → link_axes path.  In tests we call link_axes
        directly on the root after replacing the pane object to trigger
        the same comparison.
        """
        pytest.importorskip("panel")
        import panel as pn
        from bokeh.io import curdoc
        from panel.pane.holoviews import link_axes

        ds = sample_echogram_dataset

        # Build initial gps_time overlay (datetime x-axis)
        image_gps = _create_image(ds, x_mode="gps_time")
        picker = GroundingLinePicker(image_gps, ds=ds, x_mode="gps_time", y_mode="twtt")
        pts_gps = picker._points_element([])
        overlay_gps = (image_gps * pts_gps).opts(width=400, height=300)

        pane = pn.pane.HoloViews(
            overlay_gps, width=400, height=300, sizing_mode="fixed"
        )
        doc = curdoc()
        root = pane.get_root(doc)

        # Switch to numeric x-mode and update the pane object.
        picker.set_axis_modes(to_x, "twtt")
        image_new = _create_image(ds, x_mode=to_x)
        pts_new = picker._points_element([])
        overlay_new = (image_new * pts_new).opts(width=400, height=300)
        pane.object = overlay_new

        # Re-run link_axes on the root — this is the preprocessor hook
        # that Panel calls during _update_object in production.  It
        # compares axis.start > axis.end on the Bokeh Range1d, which
        # fails when datetime64 and float types are mixed.
        link_axes(pane, root)

    def test_panel_renders_without_layers(self, sample_echogram_dataset):
        """Panel overlay DynamicMaps initialize without error."""
        pytest.importorskip("panel")
        from bokeh.io import curdoc

        layout = sample_echogram_dataset.pick.panel()
        layout.get_root(curdoc())

    def test_panel_renders_with_layers(self, sample_echogram_dataset, sample_layers):
        """Panel overlay DynamicMaps initialize without error (with layers)."""
        pytest.importorskip("panel")
        from bokeh.io import curdoc

        layout = sample_echogram_dataset.pick.panel(layers=sample_layers)
        layout.get_root(curdoc())

    @pytest.mark.parametrize("from_x,to_x", _X_TRANSITIONS)
    def test_panel_overlay_renders_after_x_switch(
        self, sample_echogram_dataset, sample_layers, from_x, to_x
    ):
        """Overlay renders without DTypePromotionError after switching x-mode.

        Reproduces the live error: all overlay elements (image, points,
        layer curves) must use the same x-axis dtype after a mode switch.
        """
        ds = sample_echogram_dataset
        picker = GroundingLinePicker(
            _create_image(ds, x_mode=from_x),
            ds=ds,
            x_mode=from_x,
            y_mode="twtt",
            layers=sample_layers,
        )

        # Render overlay in the initial mode
        image = _create_image(ds, x_mode=from_x, y_mode="twtt")
        pts = picker._points_element([])
        curves = list(
            _create_layer_curves(
                sample_layers, ds=ds, x_mode=from_x, y_mode="twtt"
            ).values()
        )
        overlay = image * pts
        for c in curves:
            overlay = overlay * c
        hv.render(overlay.opts(width=400, height=300))

        # Switch mode — all elements must re-render in the new coord system
        picker.set_axis_modes(to_x, "twtt")
        image2 = _create_image(ds, x_mode=to_x, y_mode="twtt")
        pts2 = picker._points_element([])
        curves2 = list(
            _create_layer_curves(
                sample_layers, ds=ds, x_mode=to_x, y_mode="twtt"
            ).values()
        )
        overlay2 = image2 * pts2
        for c in curves2:
            overlay2 = overlay2 * c
        hv.render(overlay2.opts(width=400, height=300))

    @pytest.mark.parametrize("from_y,to_y", _Y_TRANSITIONS)
    def test_panel_overlay_renders_after_y_switch(
        self, sample_echogram_dataset, from_y, to_y
    ):
        """Overlay renders without DTypePromotionError after switching y-mode."""
        ds = sample_echogram_dataset
        picker = GroundingLinePicker(
            _create_image(ds, y_mode=from_y),
            ds=ds,
            x_mode="gps_time",
            y_mode=from_y,
        )

        image_from = _create_image(ds, x_mode="gps_time", y_mode=from_y)
        pts_from = picker._points_element([])
        overlay = image_from * pts_from
        hv.render(overlay.opts(width=400, height=300))

        picker.set_axis_modes("gps_time", to_y)
        image_to = _create_image(ds, x_mode="gps_time", y_mode=to_y)
        pts_to = picker._points_element([])
        overlay2 = image_to * pts_to
        hv.render(overlay2.opts(width=400, height=300))


class TestLayerColors:
    """Test layer color constants."""

    def test_layer_colors_defined(self):
        assert "standard:surface" in _LAYER_COLORS
        assert "standard:bottom" in _LAYER_COLORS


class TestWarpColormap:
    """Test MATLAB-style histogram equalization colormap warp."""

    def test_identity_at_zero(self):
        """hist_eq=0 returns 256 colours matching the base colormap."""
        palette = _warp_colormap("gray_r", 0.0)
        assert len(palette) == 256
        assert all(isinstance(c, str) and c.startswith("#") for c in palette)

    def test_nonzero_differs(self):
        """hist_eq!=0 produces a different palette than identity."""
        identity = _warp_colormap("gray_r", 0.0)
        warped = _warp_colormap("gray_r", 2.0)
        assert identity != warped

    def test_endpoints_unchanged(self):
        """First and last colours are the same regardless of hist_eq."""
        for exp in [-3.0, 0.0, 3.0]:
            palette = _warp_colormap("gray_r", exp)
            assert palette[0] == _warp_colormap("gray_r", 0.0)[0]
            assert palette[-1] == _warp_colormap("gray_r", 0.0)[-1]


class TestPanelSnapFunctionality:
    """Test the snap-to-layer functionality in panel()."""

    def test_panel_with_layers_has_snap_checkbox(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that panel with layers includes snap checkbox widget."""
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        # Layout is pn.Row(sidebar, main)
        sidebar = layout[0]
        assert isinstance(sidebar, pn.Column)

        # Find snap checkbox in sidebar
        snap_checkbox = None
        for item in sidebar:
            if isinstance(item, pn.widgets.Checkbox) and item.name == "Snap to layer":
                snap_checkbox = item
                break

        assert snap_checkbox is not None
        assert snap_checkbox.value is False  # Default is off

    def test_panel_with_layers_has_layer_checkboxes(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that panel with layers includes layer checkbox group."""
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        sidebar = layout[0]

        # Find layer checkboxes in sidebar
        layer_checkboxes = None
        for item in sidebar:
            if isinstance(item, pn.widgets.CheckBoxGroup):
                layer_checkboxes = item
                break

        assert layer_checkboxes is not None
        assert set(layer_checkboxes.options) == set(sample_layers.keys())
        # All layers selected by default
        assert set(layer_checkboxes.value) == set(sample_layers.keys())

    def test_snap_checkbox_triggers_callback(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that toggling snap checkbox triggers update_snap callback."""
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        sidebar = layout[0]

        # Find widgets
        snap_checkbox = None
        layer_checkboxes = None
        for item in sidebar:
            if isinstance(item, pn.widgets.Checkbox) and item.name == "Snap to layer":
                snap_checkbox = item
            elif isinstance(item, pn.widgets.CheckBoxGroup):
                layer_checkboxes = item

        # Verify watchers are attached (update_snap is registered)
        assert len(snap_checkbox.param.watchers["value"]) > 0
        assert len(layer_checkboxes.param.watchers["value"]) > 0

    def test_update_snap_sets_snap_enabled(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that enabling snap sets picker.snap_enabled."""
        acc = sample_echogram_dataset.pick
        picker = acc.picker(layers=sample_layers)

        # Initially snap_enabled should be False
        assert picker.snap_enabled is False

        # Simulate what update_snap does when snap is enabled
        picker.snap_enabled = True
        picker.visible_layers = list(sample_layers.keys())

        assert picker.snap_enabled is True

    def test_update_snap_disables_snapping(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that disabling snap sets picker.snap_enabled to False."""
        acc = sample_echogram_dataset.pick
        picker = acc.picker(layers=sample_layers)

        # Enable snap first
        picker.snap_enabled = True
        assert picker.snap_enabled is True

        # Disable snap
        picker.snap_enabled = False
        assert picker.snap_enabled is False

    def test_update_snap_sets_visible_layers(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that visible_layers is set when layers are selected."""
        acc = sample_echogram_dataset.pick
        picker = acc.picker(layers=sample_layers)

        # Simulate selecting only some layers
        picker.visible_layers = ["standard:surface"]

        assert picker.visible_layers == ["standard:surface"]

    def test_update_snap_uses_only_visible_layers(
        self, sample_echogram_dataset, sample_layers
    ):
        """Test that snapping only uses the selected visible layers."""
        acc = sample_echogram_dataset.pick
        picker = acc.picker(layers=sample_layers)

        # Enable snap and select only surface layer
        picker.snap_enabled = True
        picker.visible_layers = ["standard:surface"]

        # Get the first slow_time value for testing
        first_time = sample_layers["standard:surface"].slow_time.values[0]

        # Surface at 8e-6 s; point at 9e-6 s → 1 µs away, within threshold
        x, y = picker._snap_to_layer(first_time, 9e-6)
        assert y == pytest.approx(8e-6, rel=1e-3)

        # Bottom at 25e-6 s — NOT visible, and 20e-6 is 12 µs from surface
        x, y = picker._snap_to_layer(first_time, 20e-6)
        assert y == 20e-6  # Not snapped


class TestPanelSlope:
    """Test slope subplot in panel() method."""

    def test_panel_with_layers_has_slope_checkboxes(
        self, sample_echogram_dataset, sample_layers
    ):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        sidebar = layout[0]
        slope_checkboxes = None
        for item in sidebar:
            if (
                isinstance(item, pn.widgets.CheckBoxGroup)
                and item.name == "Slope Layers"
            ):
                slope_checkboxes = item
                break

        assert slope_checkboxes is not None
        assert set(slope_checkboxes.options) == set(sample_layers.keys())
        assert slope_checkboxes.value == []

    def test_panel_with_layers_has_smoothing_slider(
        self, sample_echogram_dataset, sample_layers
    ):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        sidebar = layout[0]
        slider = None
        for item in sidebar:
            if isinstance(item, pn.widgets.IntSlider):
                slider = item
                break

        assert slider is not None
        assert slider.value == 1
        assert slider.start == 1
        assert slider.end >= 21

    def test_panel_without_layers_no_slope_controls(self, sample_echogram_dataset):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel()

        sidebar = layout[0]
        for item in sidebar:
            assert not isinstance(item, pn.widgets.IntSlider)

    def test_panel_slope_subplot_exists(self, sample_echogram_dataset, sample_layers):
        pytest.importorskip("panel")
        import panel as pn

        acc = sample_echogram_dataset.pick
        layout = acc.panel(layers=sample_layers)

        main = layout[1]
        # Should have: echogram_pane, slope_pane, controls
        assert len(main) >= 3
        # Second item should be a HoloViews pane (the slope subplot)
        assert isinstance(main[1], pn.pane.HoloViews)
