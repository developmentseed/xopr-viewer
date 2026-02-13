# Contributing

## Development Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/developmentseed/xopr-viewer.git
    cd xopr-viewer
    ```

2. Install all dependencies (including dev and docs groups):

    ```bash
    uv sync --all-groups
    ```

3. Install pre-commit hooks:

    ```bash
    uv tool install prek
    prek install
    ```

4. Run the test suite:

    ```bash
    uv run --group dev pytest tests/ -v
    ```

5. Generate a coverage report:

    ```bash
    uv run --group dev pytest --cov=xopr_viewer --cov-report=html
    ```

6. Serve the docs locally:

    ```bash
    uv run --group docs mkdocs serve
    ```

### Code Standards

All code must conform to PEP 8. Line length limit is 100 characters, though 90 is preferred. The pre-commit hooks run [ruff](https://docs.astral.sh/ruff/) for linting and formatting, [codespell](https://github.com/codespell-project/codespell) for spelling, [nbstripout](https://github.com/kynan/nbstripout) for notebook cleaning, and [mypy](https://mypy-lang.org/) for type checking.

To run all hooks manually:

```bash
prek run --all-files
```

---

## Architecture Overview

xopr-viewer has three source modules, each with a distinct responsibility:

```
src/xopr_viewer/
    __init__.py         # Public API: GroundingLinePicker, PickAccessor, compute_layer_slope
    picker.py           # Echogram rendering, point picker, slope computation
    accessor.py         # Xarray accessor and Panel UI
    coordinates.py      # Bidirectional coordinate conversion
```

### Data Flow

```
User taps on echogram
    |
    v
Tap stream fires with display coordinates (e.g. µs, trace index)
    |
    v
display_to_canonical() converts to (slow_time, twtt_seconds)   [coordinates.py]
    |
    v
_snap_to_layer() optionally snaps twtt to nearest layer         [picker.py]
    |
    v
Point stored in picker.points as {id, slow_time, twtt}          [picker.py]
    |
    v
_points_element() renders points back via canonical_to_display() [picker.py + coordinates.py]
```

The key design principle is that **points are always stored in canonical coordinates** (slow_time as datetime64, twtt in seconds). Display coordinates are only used at the boundaries: when receiving tap events and when rendering points. This means points survive axis mode switches without any conversion of the stored data.

---

## Module Details

### `coordinates.py` -- Coordinate Conversion

This module provides the bidirectional mapping between **canonical coordinates** and **display coordinates**. Canonical coordinates are always `(slow_time, twtt_seconds)`. Display coordinates depend on the active axis modes.

#### Axis Modes

There are 3 x-modes and 5 y-modes, giving 15 valid combinations:

| X-mode | Display dim | Description |
|--------|------------|-------------|
| `rangeline` | `trace` | Integer trace index (0-based) |
| `gps_time` | `slow_time` | datetime64 timestamps (passthrough) |
| `along_track` | `along_track_km` | Cumulative along-track distance in km |

| Y-mode | Display dim | Description |
|--------|------------|-------------|
| `twtt` | `twtt_us` | Two-way travel time in microseconds |
| `range_bin` | `range_bin` | Integer sample index |
| `range` | `range_m` | One-way range in meters (`twtt * c / 2`) |
| `elevation` | `elevation_m` | WGS-84 elevation with ice velocity correction |
| `surface_flat` | `depth_m` | Depth below ice surface |

The `X_DIM_NAMES`, `Y_DIM_NAMES`, and `Y_INVERT` dicts at the top of the module define the display dimension names and axis orientation for each mode.

#### Key Functions

- **`display_to_canonical(x, y, ds, x_mode, y_mode)`** -- Called when the user taps on the echogram. Converts the clicked display coordinates back to canonical for storage.

- **`canonical_to_display(slow_time, twtt, ds, x_mode, y_mode)`** -- Called when rendering stored points. Converts canonical coordinates to the current display system.

Both functions require the dataset `ds` because some conversions need per-trace values (e.g. `Elevation`, `Surface` for the elevation and surface-flat modes). The `elevation` and `surface_flat` y-modes use a two-layer velocity model (air above surface, ice below) matching the MATLAB OPR tool.

#### Helpers

- `_along_track_km(ds)` -- Calls `xopr.radar_util.add_along_track()` to compute cumulative distance using polar stereographic projection.
- `_nearest_trace_idx(slow_time_value, slow_time_vals)` -- Finds the closest trace index by argmin, handling both datetime64 and numeric types.
- `_twtt_to_elevation` / `_elevation_to_twtt` -- Per-trace vertical conversion using aircraft elevation, surface TWTT, speed of light, and ice refractive index (n=sqrt(3.15)).
- `_twtt_to_depth` / `_depth_to_twtt` -- Simpler surface-relative depth conversion.

---

### `picker.py` -- Echogram Rendering and Point Picker

This is the largest module. It contains the echogram image creation pipeline, the interactive point picker, layer curve rendering, and slope computation.

#### Image Creation: `_create_image()`

Builds an `hv.Image` from an xarray Dataset through this pipeline:

1. **Size warning** -- Warns if the array exceeds 10M elements (Bokeh rendering limit).
2. **Y-axis transform** -- For `elevation` and `surface_flat` modes, calls `xopr.radar_util.interpolate_to_vertical_grid()` to regrid onto a uniform vertical coordinate. For `twtt`, `range_bin`, and `range`, assigns new coordinates via `swap_dims`.
3. **X-axis transform** -- For `gps_time` and `along_track`, calls `_interpolate_uniform()` to resample non-uniform spacing onto a uniform grid. For `rangeline`, assigns float trace indices.
4. **dB conversion** -- Always applies `10 * log10(|data|)` (matching MATLAB OPR, which never shows linear power). No floor is applied; zero values produce `-inf` which is excluded from auto-clim by the `np.isfinite` filter.
5. **Auto clim** -- If no explicit `clim` is provided, uses the finite min/max of the dB data.
6. **Colormap** -- Default is `"gray_r"` (inverted grayscale, matching MATLAB OPR's `1-gray(256)`). If `hist_eq != 0`, applies `_warp_colormap()` to produce a power-law-warped palette (see below).
7. **Returns** `hv.Image` with explicit `kdims=[x_dim, y_dim]` and `vdims=["power_dB"]`.

Image parameters (`clim`, `hist_eq`, `cmap`) are passed via `**image_opts` from the accessor methods:

```python
frame.pick.panel(layers=layers, clim=(-80, -20), hist_eq=3.0, cmap="viridis")
```

!!! note "Why `_interpolate_uniform()` exists"
    HoloViews `Image` requires uniformly-spaced coordinates for correct rendering via Bokeh's image glyph. CReSIS radar data often has non-uniform slow_time spacing. The interpolation step mirrors the MATLAB OPR approach: compute the median step size, build a uniform grid, and linearly interpolate.

!!! warning "Index-based coordinates must be float"
    The `rangeline` and `range_bin` modes assign integer-like indices as coordinates. These must use `np.arange(N, dtype=float)`, not plain `np.arange(N)` (which returns int64). Bokeh's image glyph intermittently fails to render with integer coordinates.

#### Histogram Equalization: `_warp_colormap()`

Replicates the MATLAB OPR `imagewin` power-law colormap warp. The formula is:

```
warped_index = linspace(0, 1, 256) ** (10 ** (hist_eq / 10))
```

At `hist_eq=0` the power is 1 (identity). Positive values brighten the image; negative values darken it. The function samples a matplotlib colormap at the warped positions and returns a list of 256 hex color strings for Bokeh.

#### Layer Curves: `_create_layer_curves()` and `_create_slope_curves()`

Both follow the same pattern:

1. Iterate over the `layers` dict (filtered by `visible_layers` if provided).
2. For each layer, extract TWTT values and the corresponding slow_time coordinates.
3. Convert coordinates to the current display system using `canonical_to_display()`.
4. Return a dict of `hv.Curve` elements with colors from `_LAYER_COLORS`.

`_create_slope_curves()` additionally calls `compute_layer_slope()` before plotting, and only converts x-coordinates (slope values are always in units of twtt/trace).

#### Slope Computation: `compute_layer_slope()`

Computes the first derivative of a layer's TWTT values along slow_time:

1. Apply a centered rolling mean with the given `smoothing_window` (must be odd).
2. Differentiate via `xr.DataArray.differentiate()`.
3. Convert to microseconds per trace for display.

#### `GroundingLinePicker`

The main interactive class. Extends `param.Parameterized` so that Panel can observe changes to `points` and `snap_enabled`.

**Internal state:**

| Attribute | Type | Purpose |
|-----------|------|---------|
| `_image` | `hv.Image` | Current echogram image |
| `_ds` | `xr.Dataset` | Dataset for coordinate conversion |
| `_x_mode`, `_y_mode` | `str` | Current axis modes |
| `x_dim`, `y_dim` | `str` | Display dimension names (derived from modes) |
| `_layers` | `dict` | Layer datasets for snapping |
| `_visible_layers` | `list` | Subset of layers enabled for snapping |
| `_snap_threshold` | `float` | Snap distance in microseconds |
| `_points_pipe` | `hv.streams.Pipe` | Feeds points to the DynamicMap |
| `_tap_stream` | `hv.streams.Tap` | Receives click events |
| `_cached_element` | `hv.Overlay` | Cached Image * Points overlay |

**Key methods:**

- `_on_tap(x, y)` -- Entry point for click events. Converts display to canonical, optionally snaps, generates a UUID, appends to `self.points`, and pushes to the Pipe stream.
- `_snap_to_layer(slow_time, twtt)` -- Finds the nearest layer point within `_snap_threshold`. Only considers `_visible_layers`. Operates entirely in canonical coordinates.
- `_points_element(data)` -- Called by the DynamicMap whenever the Pipe updates. Converts each canonical point to display coordinates and returns an `hv.Points` element.
- `set_axis_modes(x_mode, y_mode)` -- Updates the display dimension names and clears the cached element. Called by `accessor.py` when dropdowns change.
- `element()` / `panel()` -- Build standalone HoloViews/Panel layouts (used outside the accessor).

---

### `accessor.py` -- Xarray Accessor and Panel UI

Registers the `.pick` accessor on `xr.Dataset` via `@xr.register_dataset_accessor("pick")`. Provides three entry points:

- **`plot()`** -- Returns a static `hv.Element` (or `hv.Overlay` with layers). No interactivity.
- **`picker()`** -- Returns a `GroundingLinePicker` instance with the dataset attached for coordinate conversion.
- **`panel()`** -- Returns a full interactive `pn.Row` layout. This is the main entry point for the application.

#### `panel()` Architecture

The `panel()` method builds a complete UI with a sidebar and main area. The structure is:

```
pn.Row
    sidebar: pn.Column
        Display section:
            X axis selector (Select)
            Y axis selector (Select)
        Layers section (if layers provided):
            Layer checkboxes (CheckBoxGroup)
            Snap checkbox (Checkbox)
        Slope section (if layers provided):
            Slope layer checkboxes (CheckBoxGroup)
            Smoothing slider (IntSlider)
    main: pn.Column
        echogram_pane (pn.pane.HoloViews)
        slope_pane (pn.pane.HoloViews, if layers)
        controls: pn.Row
            Point count display
            Undo / Clear / Export buttons
```

Image display parameters (`clim`, `hist_eq`, `cmap`) are not exposed as sidebar widgets. They are "set and forget" parameters passed once via `**image_opts` to `panel()`:

```python
frame.pick.panel(layers=layers, clim=(-80, -20), hist_eq=3.0, cmap="viridis")
```

#### Reactive Updates via `pn.bind`

The echogram and slope panes use `pn.bind()` to connect widget values to builder functions. When any bound widget changes, the corresponding function is called and the pane is replaced with the new output.

- **`make_echogram(x_mode, y_mode, visible_layers=None)`** -- Rebuilds the entire echogram overlay. Image parameters (`clim`, `hist_eq`, `cmap`) are captured from `**image_opts` via closure. This full rebuild is necessary because Bokeh cannot switch between `DatetimeAxis` and `LinearAxis` on the same plot.

- **`slope_overlay(x_mode, visible_slopes, smoothing_window)`** -- Rebuilds the slope curves. Follows the echogram's x-mode so both subplots use the same x-axis type.

#### View Limit Preservation

When the user switches axis modes, the current zoom/pan state should be preserved where possible. The `_view` dict tracks:

- `stream` -- A `hv.streams.RangeXY` attached to the current overlay, capturing the visible x/y ranges.
- `x_mode`, `y_mode` -- The modes that were active when the stream was created.

On mode switch, `make_echogram` converts the old view limits to the new coordinate system:

1. **X-limits** are converted through a fractional index. `_x_to_frac_index()` maps any display x-value to a position in `[0, len(slow_time))`. `_frac_index_to_x()` maps back to the new display system. This mirrors OPR's `interp1(image_xaxis, image_gps_time, cur_axis)` pattern.

2. **Y-limits** are only preserved when the y-mode is unchanged. On y-mode switch, the limits reset to the full range (matching OPR's `yaxisPM_callback` which passes `-inf/inf`).

#### `linked_axes=False`

All `pn.pane.HoloViews` instances are created with `linked_axes=False`. This prevents Panel's `link_axes` preprocessor from comparing axis bounds across pane updates. Without this, switching from `gps_time` (DatetimeAxis) to a numeric mode causes a `UFuncNoLoopError` because Panel tries to compare datetime64 and float values on the Bokeh `Range1d`.

---

## Testing

Tests are in `tests/` and use pytest with fixtures defined in `conftest.py`.

### Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_image` | Simple 50x100 `hv.Image` |
| `sample_echogram_dataset` | 100 traces x 50 samples with datetime slow_time, elevation, surface, lat/lon |
| `sample_layers` | Two layers: surface at 8 us, bottom at 25-30 us |
| `nonuniform_echogram_dataset` | 100 traces with jittered slow_time spacing |

### Test Structure

| File | What it tests |
|------|---------------|
| `test_picker.py` | `GroundingLinePicker`, point operations, CSV roundtrip, snap-to-layer, slope computation |
| `test_accessor.py` | `_create_image`, all 15 axis mode combinations, mode switching, Panel UI, layer curves, clim, histogram eq |
| `test_coordinates.py` | All coordinate conversions, roundtrips, edge cases, invalid modes |

### Key Test Patterns

**Parametrized mode combinations** -- `TestAllAxisModeCombinations` tests all 3x5=15 x/y mode pairs. `TestAxisModeSwitching` parametrizes all mode transitions to verify points survive switches.

**Panel rendering tests** -- Tests call `layout.get_root(curdoc())` to trigger Bokeh model creation without a running server. The `link_axes` regression test directly calls Panel's `link_axes()` preprocessor after replacing the pane object.

**Canonical coordinate invariance** -- `test_canonical_coords_unchanged_after_switch` verifies that switching modes doesn't alter stored point data.

---

## Adding a New Axis Mode

To add a new axis mode (e.g. a new y-mode called `"ice_thickness"`):

1. **`coordinates.py`** -- Add entries to `Y_DIM_NAMES` and `Y_INVERT`. Add conversion cases in both `display_to_canonical()` and `canonical_to_display()`.

2. **`picker.py`** -- Add the coordinate transform in `_create_image()` (the y-axis transform section, after the existing `elif` chain). Assign new coordinates via `swap_dims`.

3. **`accessor.py`** -- Add the option to `y_options` in `panel()`.

4. **Tests** -- The parametrized tests will automatically cover the new mode once it's added to the `_Y_MODES` list in `test_accessor.py`.

---

## Adding a New Sidebar Widget

To add a new widget to the `panel()` sidebar:

1. Create the widget in the widget section of `panel()` (after the existing selectors).

2. If it affects the echogram, add it as a parameter to `make_echogram()` and include it in both `pn.bind()` calls (with-layers and without-layers branches). For sliders, bind via `.param.value_throttled` to only fire on mouse release, avoiding expensive redraws during drag.

3. If it affects the slope subplot, add it to `slope_overlay()` and its `pn.bind()` call.

4. Add the widget to the `sidebar` Column.

---

## External Dependencies

| Library | Role |
|---------|------|
| [HoloViews](https://holoviews.org/) | Image, Curve, Points, Overlay, DynamicMap, Tap/Pipe streams |
| [Panel](https://panel.holoviz.org/) | Layout (Row, Column), widgets, reactive binding, HoloViews pane |
| [Bokeh](https://docs.bokeh.org/) | Underlying rendering engine (via HoloViews backend) |
| [xarray](https://docs.xarray.dev/) | Dataset/DataArray, accessor pattern, interpolation, differentiation |
| [xopr](https://github.com/openradar/xopr) | `add_along_track()`, `interpolate_to_vertical_grid()`, Open Polar Radar data loading |
| [matplotlib](https://matplotlib.org/) | Colormap sampling for `_warp_colormap()` |
| [scipy](https://docs.scipy.org/) | `scipy.constants.c` (speed of light) |
