"""Tests for the GroundingLinePicker class."""

import tempfile
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from xopr_viewer.picker import (
    GroundingLinePicker,
    compute_layer_slope,
    _create_slope_curves,
    pick_echogram,
)

# Load bokeh extension for tests that use .opts()
hv.extension("bokeh")


@pytest.fixture
def picker(sample_image):
    """Create a picker instance."""
    return GroundingLinePicker(sample_image)


class TestGroundingLinePickerInit:
    """Test picker initialization."""

    def test_init_with_image(self, sample_image):
        picker = GroundingLinePicker(sample_image)
        assert picker.x_dim == "trace"
        assert picker.y_dim == "twtt_us"
        assert picker.points == []

    def test_init_with_custom_dims(self, sample_image):
        picker = GroundingLinePicker(sample_image, x_dim="distance", y_dim="depth")
        assert picker.x_dim == "distance"
        assert picker.y_dim == "depth"

    def test_init_infers_dims_from_image(self):
        image = hv.Image(np.random.rand(10, 10), kdims=["along_track", "depth"])
        picker = GroundingLinePicker(image)
        assert picker.x_dim == "along_track"
        assert picker.y_dim == "depth"


class TestPointOperations:
    """Test adding, removing, and clearing points."""

    def test_on_tap_adds_point(self, picker):
        picker._on_tap(10.0, 20.0)
        assert len(picker.points) == 1
        assert picker.points[0]["trace"] == 10.0
        assert picker.points[0]["twtt_us"] == 20.0
        assert "id" in picker.points[0]

    def test_on_tap_ignores_none(self, picker):
        picker._on_tap(None, None)
        assert len(picker.points) == 0

        picker._on_tap(10.0, None)
        assert len(picker.points) == 0

        picker._on_tap(None, 20.0)
        assert len(picker.points) == 0

    def test_multiple_taps(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)
        picker._on_tap(5.0, 6.0)
        assert len(picker.points) == 3

    def test_undo_removes_last_point(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)
        picker.undo()
        assert len(picker.points) == 1
        assert picker.points[0]["trace"] == 1.0

    def test_undo_on_empty_is_safe(self, picker):
        picker.undo()  # Should not raise
        assert len(picker.points) == 0

    def test_clear_removes_all_points(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)
        picker.clear()
        assert len(picker.points) == 0

    def test_delete_by_id(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)
        point_id = picker.points[0]["id"]
        picker.delete_by_id(point_id)
        assert len(picker.points) == 1
        assert picker.points[0]["trace"] == 3.0

    def test_delete_by_id_nonexistent(self, picker):
        picker._on_tap(1.0, 2.0)
        picker.delete_by_id("nonexistent")
        assert len(picker.points) == 1  # Nothing deleted


class TestDataFrame:
    """Test DataFrame property."""

    def test_df_empty(self, picker):
        df = picker.df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "id" in df.columns
        assert "trace" in df.columns
        assert "twtt_us" in df.columns

    def test_df_with_points(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)
        df = picker.df
        assert len(df) == 2
        assert df["trace"].tolist() == [1.0, 3.0]
        assert df["twtt_us"].tolist() == [2.0, 4.0]

    def test_df_uses_correct_dim_names(self):
        image = hv.Image(np.random.rand(10, 10), kdims=["distance", "depth"])
        picker = GroundingLinePicker(image)
        picker._on_tap(5.0, 10.0)
        df = picker.df
        assert "distance" in df.columns
        assert "depth" in df.columns


class TestCSVExportImport:
    """Test CSV save/load functionality."""

    def test_to_csv(self, picker):
        picker._on_tap(1.0, 2.0)
        picker._on_tap(3.0, 4.0)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        picker.to_csv(path)
        df = pd.read_csv(path)
        assert len(df) == 2
        assert df["trace"].tolist() == [1.0, 3.0]
        Path(path).unlink()

    def test_from_csv(self, picker):
        # Create a CSV file
        df = pd.DataFrame(
            {
                "id": ["abc", "def"],
                "trace": [10.0, 20.0],
                "twtt_us": [30.0, 40.0],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        df.to_csv(path, index=False)

        picker.from_csv(path)
        assert len(picker.points) == 2
        assert picker.points[0]["trace"] == 10.0
        assert picker.points[1]["twtt_us"] == 40.0
        Path(path).unlink()

    def test_roundtrip(self, picker):
        picker._on_tap(1.5, 2.5)
        picker._on_tap(3.5, 4.5)

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name

        picker.to_csv(path)
        picker.clear()
        assert len(picker.points) == 0

        picker.from_csv(path)
        assert len(picker.points) == 2
        Path(path).unlink()


class TestHoloViewsElements:
    """Test HoloViews element generation."""

    def test_element_returns_overlay(self, picker):
        elem = picker.element()
        # Image * DynamicMap returns an Overlay or DynamicMap
        assert isinstance(elem, (hv.Overlay, hv.DynamicMap))

    def test_points_element_empty(self, picker):
        points = picker._points_element([])
        assert isinstance(points, hv.Points)
        assert len(points) == 0

    def test_points_element_with_data(self, picker):
        data = [{"id": "a", "trace": 1.0, "twtt_us": 2.0}]
        points = picker._points_element(data)
        assert isinstance(points, hv.Points)
        assert len(points) == 1


class TestPickEchogramConvenience:
    """Test the pick_echogram convenience function."""

    def test_with_holoviews_image(self):
        image = hv.Image(np.random.rand(10, 10), kdims=["trace", "twtt_us"])
        picker = pick_echogram(image)
        assert isinstance(picker, GroundingLinePicker)

    def test_with_xarray_dataarray(self):
        pytest.importorskip("xarray")
        import xarray as xr

        data = np.random.rand(20, 30)
        da = xr.DataArray(
            data,
            dims=["twtt_us", "trace"],
            coords={"twtt_us": np.arange(20), "trace": np.arange(30)},
        )
        picker = pick_echogram(da, x_dim="trace", y_dim="twtt_us")
        assert isinstance(picker, GroundingLinePicker)
        assert picker.x_dim == "trace"
        assert picker.y_dim == "twtt_us"

    def test_passes_opts_to_image(self):
        image = hv.Image(np.random.rand(10, 10), kdims=["trace", "twtt_us"])
        picker = pick_echogram(image, cmap="gray")
        # Opts are applied - we can verify the picker was created
        assert isinstance(picker, GroundingLinePicker)


class TestPanelIntegration:
    """Test Panel widget integration."""

    def test_panel_returns_column(self, picker):
        pytest.importorskip("panel")
        import panel as pn

        layout = picker.panel()
        assert isinstance(layout, pn.Column)

    def test_panel_custom_size(self, picker):
        pytest.importorskip("panel")
        layout = picker.panel(width=500, height=300)
        # Should not raise, layout created with custom size
        assert layout is not None


class TestSnapToLayer:
    """Test snap-to-layer functionality in GroundingLinePicker."""

    @pytest.fixture
    def numeric_layers(self):
        """Create layer data with numeric x coordinates for testing."""
        import xarray as xr

        n_traces = 100
        x_values = np.arange(n_traces, dtype=float)

        surface_ds = xr.Dataset(
            {"twtt": (["trace"], np.ones(n_traces) * 8e-6)},  # 8 microseconds
            coords={"trace": x_values},
        )

        bottom_ds = xr.Dataset(
            {"twtt": (["trace"], np.linspace(25e-6, 30e-6, n_traces))},  # 25-30 us
            coords={"trace": x_values},
        )

        return {
            "surface": surface_ds,
            "bottom": bottom_ds,
        }

    def test_snaps_to_nearest_layer_within_threshold(
        self, sample_image, numeric_layers
    ):
        """Test that points within threshold snap to nearest layer."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=5.0, x_coord="trace"
        )
        picker.snap_enabled = True

        # Surface layer is at 8 microseconds (8e-6 * 1e6 = 8)
        # Test a point close to surface
        x, y = picker._snap_to_layer(0, 9.0)  # 1 us away from surface (8us)
        assert y == pytest.approx(8.0, rel=1e-3)

    def test_does_not_snap_outside_threshold(self, sample_image, numeric_layers):
        """Test that points outside threshold are not snapped."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=5.0, x_coord="trace"
        )
        picker.snap_enabled = True

        # Surface is at 8us, bottom starts at 25us
        # 16us is far from both layers
        x, y = picker._snap_to_layer(0, 16.0)
        assert y == 16.0  # Should not snap

    def test_snaps_to_closer_layer(self, sample_image, numeric_layers):
        """Test that it snaps to the closest layer when multiple are in range."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=10.0, x_coord="trace"
        )
        picker.snap_enabled = True

        # Surface at 8us, bottom at ~25us at trace 0
        # Test point at 10us - closer to surface (2us away) than bottom (15us away)
        x, y = picker._snap_to_layer(0, 10.0)
        assert y == pytest.approx(8.0, rel=1e-3)

    def test_empty_layers_returns_original(self, sample_image):
        """Test that empty layers dict returns original coordinates."""
        picker = GroundingLinePicker(sample_image, layers={})
        picker.snap_enabled = True

        x, y = picker._snap_to_layer(10.0, 20.0)
        assert x == 10.0
        assert y == 20.0

    def test_x_coordinate_unchanged(self, sample_image, numeric_layers):
        """Test that x coordinate is never modified."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=5.0, x_coord="trace"
        )
        picker.snap_enabled = True

        x, y = picker._snap_to_layer(42.0, 9.0)
        assert x == 42.0

    def test_custom_threshold(self, sample_image, numeric_layers):
        """Test that custom threshold is respected."""
        # With small threshold, should not snap
        picker_small = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=0.5, x_coord="trace"
        )
        x, y = picker_small._snap_to_layer(0, 9.0)  # 1us away from surface
        assert y == 9.0  # Should not snap (outside 0.5us threshold)

        # With larger threshold, should snap
        picker_large = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=2.0, x_coord="trace"
        )
        x, y = picker_large._snap_to_layer(0, 9.0)
        assert y == pytest.approx(8.0, rel=1e-3)  # Should snap

    def test_works_with_datetime_coordinates(self, sample_image, sample_layers):
        """Test that snapping works with datetime64 coordinates."""
        # sample_layers uses datetime64 for slow_time coordinate
        picker = GroundingLinePicker(
            sample_image, layers=sample_layers, snap_threshold=5.0, x_coord="slow_time"
        )
        picker.snap_enabled = True

        # Get a valid datetime value from the layer to use as x
        first_time = sample_layers["standard:surface"].slow_time.values[0]

        # Surface layer is at 8 microseconds
        x, y = picker._snap_to_layer(first_time, 9.0)  # 1 us away from surface
        assert y == pytest.approx(8.0, rel=1e-3)

    def test_snap_enabled_controls_snapping(self, sample_image, numeric_layers):
        """Test that snap_enabled controls whether snapping occurs."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=5.0, x_coord="trace"
        )

        # Snap disabled by default
        assert picker.snap_enabled is False
        picker._on_tap(0, 9.0)
        assert picker.points[0]["twtt_us"] == 9.0  # Not snapped

        picker.clear()

        # Enable snap
        picker.snap_enabled = True
        picker._on_tap(0, 9.0)
        assert picker.points[0]["twtt_us"] == pytest.approx(8.0, rel=1e-3)  # Snapped

    def test_visible_layers_filters_snap_targets(self, sample_image, numeric_layers):
        """Test that visible_layers controls which layers are used for snapping."""
        picker = GroundingLinePicker(
            sample_image, layers=numeric_layers, snap_threshold=5.0, x_coord="trace"
        )
        picker.snap_enabled = True

        # Only snap to surface layer
        picker.visible_layers = ["surface"]

        # Surface is at 8us - should snap
        x, y = picker._snap_to_layer(0, 9.0)
        assert y == pytest.approx(8.0, rel=1e-3)

        # Bottom is at 25us - should NOT snap since it's not visible
        # (and 20us is outside threshold from surface at 8us)
        x, y = picker._snap_to_layer(0, 20.0)
        assert y == 20.0  # Not snapped


class TestComputeLayerSlope:
    """Test compute_layer_slope function."""

    @staticmethod
    def _make_da(values):
        """Helper to create an xarray DataArray from numpy values."""
        import xarray as xr

        return xr.DataArray(
            values, dims=["trace"], coords={"trace": np.arange(len(values))}
        )

    def test_flat_layer_zero_slope(self):
        twtt = self._make_da(np.ones(100) * 8e-6)
        slope = compute_layer_slope(twtt, smoothing_window=1)
        np.testing.assert_allclose(slope.values, 0.0, atol=1e-10)

    def test_linear_layer_constant_slope(self):
        twtt = self._make_da(np.linspace(25e-6, 30e-6, 100))
        slope = compute_layer_slope(twtt, smoothing_window=1)
        expected = 5e-6 / 99  # seconds per trace
        np.testing.assert_allclose(slope.values[1:-1], expected, rtol=1e-3)

    def test_smoothing_reduces_noise(self):
        np.random.seed(42)
        twtt = self._make_da(np.ones(100) * 8e-6 + np.random.normal(0, 0.1e-6, 100))
        slope_raw = compute_layer_slope(twtt, smoothing_window=1)
        slope_smooth = compute_layer_slope(twtt, smoothing_window=11)
        assert np.nanstd(slope_smooth.values) < np.nanstd(slope_raw.values)

    def test_smoothing_window_1_is_no_smoothing(self):
        twtt = self._make_da(np.linspace(0, 10e-6, 50))
        slope = compute_layer_slope(twtt, smoothing_window=1)
        # differentiate gives d(twtt)/d(trace), same as np.gradient
        expected = np.gradient(np.linspace(0, 10e-6, 50))
        np.testing.assert_allclose(slope.values, expected)

    def test_output_length_matches_input(self):
        for n in [2, 10, 100]:
            twtt = self._make_da(np.ones(n) * 8e-6)
            slope = compute_layer_slope(twtt, smoothing_window=1)
            assert len(slope) == n

    def test_large_smoothing_window_clamped(self):
        twtt = self._make_da(np.ones(5) * 8e-6)
        slope = compute_layer_slope(twtt, smoothing_window=99)
        assert len(slope) == 5

    def test_returns_dataarray(self):
        import xarray as xr

        twtt = self._make_da(np.linspace(0, 10e-6, 50))
        slope = compute_layer_slope(twtt, smoothing_window=1)
        assert isinstance(slope, xr.DataArray)


class TestCreateSlopeCurves:
    """Test _create_slope_curves function."""

    def test_creates_curves_for_all_layers(self, sample_layers):
        curves = _create_slope_curves(sample_layers)
        assert len(curves) == 2
        assert "standard:surface" in curves
        assert "standard:bottom" in curves

    def test_filters_by_visible_layers(self, sample_layers):
        curves = _create_slope_curves(
            sample_layers, visible_layers=["standard:surface"]
        )
        assert len(curves) == 1
        assert "standard:surface" in curves

    def test_empty_visible_layers_returns_empty(self, sample_layers):
        curves = _create_slope_curves(sample_layers, visible_layers=[])
        assert len(curves) == 0

    def test_curves_use_slow_time_kdim(self, sample_layers):
        curves = _create_slope_curves(sample_layers)
        for curve in curves.values():
            assert "slow_time" in [d.name for d in curve.kdims]

    def test_curves_use_twtt_vdim(self, sample_layers):
        curves = _create_slope_curves(sample_layers)
        for curve in curves.values():
            assert "twtt" in [d.name for d in curve.vdims]

    def test_smoothing_window_applied(self, sample_layers):
        curves_1 = _create_slope_curves(sample_layers, smoothing_window=1)
        curves_11 = _create_slope_curves(sample_layers, smoothing_window=11)
        assert curves_1.keys() == curves_11.keys()

    def test_returns_holoviews_curves(self, sample_layers):
        curves = _create_slope_curves(sample_layers)
        for curve in curves.values():
            assert isinstance(curve, hv.Curve)
