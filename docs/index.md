# xopr-viewer

Interactive tools for viewing and labeling ground lines in radar echogram data, built on [HoloViews](https://holoviews.org/) and [Panel](https://panel.holoviz.org/).

## Features

- **Interactive point picker** for annotating echograms with click-to-add points
- **Layer overlays** to visualize existing surface/bottom picks alongside the echogram
- **Snap-to-layer** functionality for precise annotations aligned to detected layers
- **Xarray accessor** for seamless integration with [xopr](https://github.com/openradar/xopr) radar datasets

## Installation

```bash
pip install "xopr-viewer[xopr] @ git+https://github.com/developmentseed/xopr-viewer.git"
```

or

```bash
uv add "xopr-viewer[xopr] @ git+https://github.com/developmentseed/xopr-viewer.git"
```

Omit `[xopr]` if you only need the viewer without xopr integration.

## Quick Start

### Using the xarray accessor

```python
import holoviews as hv
import xopr

hv.extension("bokeh")

# Load radar frame using xopr
frame = xopr.open_frame("path/to/radar/data")

# Create interactive picker using the xarray accessor
layers = xopr.get_layers(frame)
frame.pick.show(layers=layers)
```

### Using the picker directly

```python
import holoviews as hv
import numpy as np
from xopr_viewer import GroundingLinePicker

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

## Contributing

1. Clone the repository: `git clone https://github.com/developmentseed/xopr-viewer.git`
2. Install development dependencies: `uv sync --all-groups`
3. Run the test suite: `uv run --group dev pytest`
4. Generate an HTML coverage report:

```bash
uv run --group dev pytest --cov=xopr_viewer --cov-report=html
```

The report will be generated in the `htmlcov/` directory. Open `htmlcov/index.html` in a browser to view it.

### Code standards - using prek

All code must conform to the PEP8 standard. Regarding line length, lines up to 100 characters are allowed, although please try to keep under 90 wherever possible.

`xopr-viewer` uses a set of git hooks managed by [`prek`](https://github.com/j178/prek), a fast, Rust-based pre-commit hook manager that is fully compatible with `.pre-commit-config.yaml` files. `prek` can be installed locally by running:

```bash
uv tool install prek
```

or:

```bash
pip install prek
```

The hooks can be installed locally by running:

```bash
prek install
```

This would run the checks every time a commit is created locally. The checks will by default only run on the files modified by a commit, but the checks can be triggered for all the files by running:

```bash
prek run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

## License

`xopr-viewer` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
