"""Tests for coordinate conversion utilities."""

import numpy as np
import pytest
import scipy.constants
import xarray as xr

from xopr_viewer.coordinates import (
    X_DIM_NAMES,
    Y_DIM_NAMES,
    Y_INVERT,
    canonical_to_display,
    display_to_canonical,
    _nearest_trace_idx,
    _twtt_to_elevation,
    _elevation_to_twtt,
    _twtt_to_depth,
    _depth_to_twtt,
)


@pytest.fixture
def simple_ds():
    """Minimal dataset for coordinate conversion tests."""
    n_traces = 50
    n_samples = 30

    base_time = np.datetime64("2023-01-09T12:00:00")
    slow_time = np.array([base_time + np.timedelta64(i, "s") for i in range(n_traces)])
    twtt = np.linspace(0, 50e-6, n_samples)

    return xr.Dataset(
        {
            "Data": (["twtt", "slow_time"], np.random.rand(n_samples, n_traces)),
            "Latitude": (["slow_time"], np.linspace(-75.0, -74.9, n_traces)),
            "Longitude": (["slow_time"], np.linspace(102.0, 102.5, n_traces)),
            "Elevation": (["slow_time"], np.ones(n_traces) * 3000.0),
            "Surface": (["slow_time"], np.ones(n_traces) * 8e-6),
        },
        coords={"slow_time": slow_time, "twtt": twtt},
    )


class TestDimNameConstants:
    """Test dimension name and inversion constants."""

    def test_x_dim_names_complete(self):
        assert set(X_DIM_NAMES) == {"rangeline", "gps_time", "along_track"}

    def test_y_dim_names_complete(self):
        assert set(Y_DIM_NAMES) == {
            "twtt",
            "range_bin",
            "range",
            "elevation",
            "surface_flat",
        }

    def test_y_invert_elevation_false(self):
        assert Y_INVERT["elevation"] is False

    def test_y_invert_others_true(self):
        for mode in ("twtt", "range_bin", "range", "surface_flat"):
            assert Y_INVERT[mode] is True


class TestNearestTraceIdx:
    """Test _nearest_trace_idx helper."""

    def test_exact_match(self, simple_ds):
        vals = simple_ds["slow_time"].values
        idx = _nearest_trace_idx(vals[10], vals)
        assert idx == 10

    def test_nearest_datetime(self, simple_ds):
        vals = simple_ds["slow_time"].values
        # Halfway between index 5 and 6
        target = vals[5] + np.timedelta64(500, "ms")
        idx = _nearest_trace_idx(target, vals)
        assert idx in (5, 6)

    def test_numeric_values(self):
        vals = np.arange(10, dtype=float)
        idx = _nearest_trace_idx(3.7, vals)
        assert idx == 4

    def test_boundary_clamp(self):
        vals = np.arange(10, dtype=float)
        assert _nearest_trace_idx(-5.0, vals) == 0
        assert _nearest_trace_idx(100.0, vals) == 9


class TestTwttDisplayConversion:
    """Test twtt mode (µs ↔ seconds)."""

    def test_display_to_canonical_twtt(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[0]
        _, twtt_s = display_to_canonical(slow_time, 25.0, simple_ds, "gps_time", "twtt")
        assert twtt_s == pytest.approx(25e-6)

    def test_canonical_to_display_twtt(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[0]
        _, twtt_us = canonical_to_display(
            slow_time, 25e-6, simple_ds, "gps_time", "twtt"
        )
        assert twtt_us == pytest.approx(25.0)


class TestRangeBinConversion:
    """Test range_bin mode (index ↔ twtt seconds)."""

    def test_display_to_canonical_range_bin(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[0]
        _, twtt_s = display_to_canonical(
            slow_time, 0, simple_ds, "gps_time", "range_bin"
        )
        assert twtt_s == pytest.approx(simple_ds["twtt"].values[0])

    def test_canonical_to_display_range_bin(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[0]
        twtt_s = simple_ds["twtt"].values[10]
        _, bin_idx = canonical_to_display(
            slow_time, twtt_s, simple_ds, "gps_time", "range_bin"
        )
        assert bin_idx == pytest.approx(10.0)


class TestRangeConversion:
    """Test range mode (meters ↔ twtt seconds)."""

    def test_display_to_canonical_range(self, simple_ds):
        c = scipy.constants.c
        slow_time = simple_ds["slow_time"].values[0]
        range_m = 100.0
        _, twtt_s = display_to_canonical(
            slow_time, range_m, simple_ds, "gps_time", "range"
        )
        assert twtt_s == pytest.approx(range_m * 2.0 / c)

    def test_canonical_to_display_range(self, simple_ds):
        c = scipy.constants.c
        slow_time = simple_ds["slow_time"].values[0]
        twtt_s = 1e-6
        _, range_m = canonical_to_display(
            slow_time, twtt_s, simple_ds, "gps_time", "range"
        )
        assert range_m == pytest.approx(twtt_s * c / 2.0)


class TestRangelineConversion:
    """Test rangeline mode (trace index ↔ slow_time)."""

    def test_display_to_canonical_rangeline(self, simple_ds):
        canonical_x, _ = display_to_canonical(10, 25.0, simple_ds, "rangeline", "twtt")
        assert canonical_x == simple_ds["slow_time"].values[10]

    def test_canonical_to_display_rangeline(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[10]
        display_x, _ = canonical_to_display(
            slow_time, 25e-6, simple_ds, "rangeline", "twtt"
        )
        assert display_x == pytest.approx(10.0)


class TestGpsTimeConversion:
    """Test gps_time mode (passthrough)."""

    def test_gps_time_passthrough(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[5]
        canonical_x, _ = display_to_canonical(
            slow_time, 25.0, simple_ds, "gps_time", "twtt"
        )
        assert canonical_x == slow_time

    def test_canonical_to_display_gps_time(self, simple_ds):
        slow_time = simple_ds["slow_time"].values[5]
        display_x, _ = canonical_to_display(
            slow_time, 25e-6, simple_ds, "gps_time", "twtt"
        )
        assert display_x == slow_time


class TestElevationConversion:
    """Test elevation mode roundtrip."""

    def test_twtt_to_elevation_above_surface(self, simple_ds):
        """Above-surface TWTT uses air velocity."""
        c = scipy.constants.c
        aircraft_elev = 3000.0
        twtt = 4e-6  # above surface (8e-6)
        elev = _twtt_to_elevation(twtt, 0, simple_ds)
        assert elev == pytest.approx(aircraft_elev - twtt * c / 2.0)

    def test_elevation_roundtrip(self, simple_ds):
        """elevation → twtt → elevation recovers original."""
        twtt_original = 20e-6
        elev = _twtt_to_elevation(twtt_original, 0, simple_ds)
        twtt_recovered = _elevation_to_twtt(elev, 0, simple_ds)
        assert twtt_recovered == pytest.approx(twtt_original, rel=1e-6)


class TestDepthConversion:
    """Test surface-flat (depth) mode roundtrip."""

    def test_depth_roundtrip(self, simple_ds):
        """depth → twtt → depth recovers original."""
        twtt_original = 20e-6
        depth = _twtt_to_depth(twtt_original, 0, simple_ds)
        twtt_recovered = _depth_to_twtt(depth, 0, simple_ds)
        assert twtt_recovered == pytest.approx(twtt_original, rel=1e-6)

    def test_surface_depth_is_zero(self, simple_ds):
        """At surface TWTT, depth should be ~0."""
        surface_twtt = float(simple_ds["Surface"].values[0])
        depth = _twtt_to_depth(surface_twtt, 0, simple_ds)
        assert depth == pytest.approx(0.0, abs=1e-6)


class TestDisplayCanonicalRoundtrip:
    """Test that display→canonical→display roundtrips for various modes."""

    @pytest.mark.parametrize(
        "x_mode,y_mode",
        [
            ("gps_time", "twtt"),
            ("rangeline", "twtt"),
            ("rangeline", "range_bin"),
            ("gps_time", "range"),
            ("gps_time", "elevation"),
            ("gps_time", "surface_flat"),
        ],
    )
    def test_roundtrip(self, simple_ds, x_mode, y_mode):
        """Converting display→canonical→display recovers original values."""
        # Pick a display point
        slow_time = simple_ds["slow_time"].values[10]
        twtt_s = 20e-6

        # canonical → display
        dx, dy = canonical_to_display(slow_time, twtt_s, simple_ds, x_mode, y_mode)

        # display → canonical
        cx, cy = display_to_canonical(dx, dy, simple_ds, x_mode, y_mode)

        # canonical → display again
        dx2, dy2 = canonical_to_display(cx, cy, simple_ds, x_mode, y_mode)

        # Display values should match (within floating point tolerance)
        if isinstance(dx, (int, float)):
            assert dx2 == pytest.approx(dx, rel=1e-3)
        assert dy2 == pytest.approx(dy, rel=1e-3)


class TestInvalidModes:
    """Test error handling for unknown modes."""

    def test_unknown_x_mode(self, simple_ds):
        with pytest.raises(ValueError, match="Unknown x_mode"):
            display_to_canonical(0, 0, simple_ds, "invalid", "twtt")

    def test_unknown_y_mode(self, simple_ds):
        with pytest.raises(ValueError, match="Unknown y_mode"):
            display_to_canonical(0, 0, simple_ds, "gps_time", "invalid")

    def test_unknown_x_mode_canonical(self, simple_ds):
        st = simple_ds["slow_time"].values[0]
        with pytest.raises(ValueError, match="Unknown x_mode"):
            canonical_to_display(st, 1e-6, simple_ds, "invalid", "twtt")

    def test_unknown_y_mode_canonical(self, simple_ds):
        st = simple_ds["slow_time"].values[0]
        with pytest.raises(ValueError, match="Unknown y_mode"):
            canonical_to_display(st, 1e-6, simple_ds, "gps_time", "invalid")
