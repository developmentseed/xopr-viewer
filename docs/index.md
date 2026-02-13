# xopr-viewer

Interactive tools for viewing and labeling ground lines in radar echogram data, built on [HoloViews](https://holoviews.org/) and [Panel](https://panel.holoviz.org/).

## Features

- **Interactive point picker** for annotating echograms with click-to-add points
- **Layer overlays** to visualize existing surface/bottom picks alongside the echogram
- **Snap-to-layer** functionality for precise annotations aligned to detected layers
- **Xarray accessor** for seamless integration with [xopr](https://github.com/openradar/xopr) radar datasets

## Installation

```bash
pip install "xopr-viewer @ git+https://github.com/developmentseed/xopr-viewer.git"
```

or

```bash
uv add "xopr-viewer @ git+https://github.com/developmentseed/xopr-viewer.git"
```

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

See the [Contributing Guide](contributing.md) for development setup, architecture overview, and how the components work.

## License

`xopr-viewer` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
