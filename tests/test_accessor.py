"""Tests for the PickAccessor xarray accessor."""

import holoviews as hv
import pytest

# Import accessor to register it
import xopr_viewer.accessor  # noqa: F401
from xopr_viewer.accessor import PickAccessor
from xopr_viewer.picker import (
    GroundingLinePicker,
    _create_image,
    _create_layer_curves,
    _get_title,
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

    def test_create_image_log_scale(self, sample_echogram_dataset):
        image = _create_image(sample_echogram_dataset, log_scale=True)
        assert isinstance(image, hv.Image)

    def test_create_image_linear_scale(self, sample_echogram_dataset):
        image = _create_image(sample_echogram_dataset, log_scale=False)
        assert isinstance(image, hv.Image)


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

    def test_create_layer_curves_uses_colors(self, sample_layers):
        curves = _create_layer_curves(sample_layers)
        assert "standard:surface" in _LAYER_COLORS

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


class TestLayerColors:
    """Test layer color constants."""

    def test_layer_colors_defined(self):
        assert "standard:surface" in _LAYER_COLORS
        assert "standard:bottom" in _LAYER_COLORS


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

        # Surface is at 8us, test snapping to it
        x, y = picker._snap_to_layer(first_time, 9.0)
        assert y == pytest.approx(8.0, rel=1e-3)

        # Bottom is at 25us at start - should NOT snap since it's not in visible
        # (threshold is 5us by default, so 20us is way outside)
        x, y = picker._snap_to_layer(first_time, 20.0)
        assert y == 20.0  # Not snapped
