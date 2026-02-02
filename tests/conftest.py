"""Shared pytest fixtures for xopr_viewer tests."""

import holoviews as hv
import numpy as np
import pytest
import xarray as xr

# Load bokeh extension for tests
hv.extension("bokeh")


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    data = np.random.rand(50, 100)
    return hv.Image(data, kdims=["trace", "twtt_us"])


@pytest.fixture
def sample_echogram_dataset():
    """Create a sample xarray Dataset mimicking OPR echogram data."""
    np.random.seed(42)
    n_traces = 100
    n_samples = 50

    # Create time coordinates
    base_time = np.datetime64("2023-01-09T12:00:00")
    slow_time = np.array([base_time + np.timedelta64(i, "s") for i in range(n_traces)])
    twtt = np.linspace(0, 50e-6, n_samples)  # 0-50 microseconds

    # Create data
    data = np.random.rand(n_samples, n_traces) * 1e-6

    # Create Dataset
    ds = xr.Dataset(
        {
            "Data": (["twtt", "slow_time"], data),
            "Latitude": (["slow_time"], np.linspace(-75.0, -74.9, n_traces)),
            "Longitude": (["slow_time"], np.linspace(102.0, 102.5, n_traces)),
            "Elevation": (["slow_time"], np.linspace(1500, 1520, n_traces)),
            "Surface": (["slow_time"], np.ones(n_traces) * 8e-6),
        },
        coords={
            "slow_time": slow_time,
            "twtt": twtt,
        },
        attrs={
            "collection": "2022_Antarctica_BaslerMKB",
            "segment_path": "20230109_01",
            "granule": "20230109_01_001",
            "frame": 1,
        },
    )
    return ds


@pytest.fixture
def sample_layers():
    """Create sample layer data mimicking OPR layers."""
    n_traces = 100

    # Create slow_time coordinate matching sample_echogram_dataset
    base_time = np.datetime64("2023-01-09T12:00:00")
    slow_time = np.array([base_time + np.timedelta64(i, "s") for i in range(n_traces)])

    surface_ds = xr.Dataset(
        {
            "twtt": (["slow_time"], np.ones(n_traces) * 8e-6),
        },
        coords={"slow_time": slow_time},
    )

    bottom_ds = xr.Dataset(
        {
            "twtt": (["slow_time"], np.linspace(25e-6, 30e-6, n_traces)),
        },
        coords={"slow_time": slow_time},
    )

    return {
        "standard:surface": surface_ds,
        "standard:bottom": bottom_ds,
    }
