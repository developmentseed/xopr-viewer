# xopr-viewer

Interactive tools for viewing and labeling ground lines in radar echogram data, built on [HoloViews](https://holoviews.org/) and [Panel](https://panel.holoviz.org/).

- **Interactive point picker** for annotating echograms with click-to-add points.
- **Layer overlays** to visualize existing surface/bottom picks alongside the echogram.
- **Snap-to-layer** functionality for precise annotations aligned to detected layers.
- **Xarray accessor** for seamless integration with [xopr](https://github.com/openradar/xopr) radar datasets.

## Installation

```sh
pip install "xopr-viewer[xopr] @ git+https://github.com/developmentseed/xopr-viewer.git"
```

or

```sh
uv add "xopr-viewer[xopr] @ git+https://github.com/developmentseed/xopr-viewer.git"
```

Omit `[xopr]` if you only need the viewer without [xopr](https://github.com/openradar/xopr) integration.

## Quick Example

```python
import holoviews as hv
import xopr
from xopr_viewer import GroundingLinePicker

hv.extension("bokeh")

# Load radar frame using xopr
frame = xopr.open_frame("path/to/radar/data")

# Create interactive picker using the xarray accessor
frame.pick.show(layers=frame.xopr.layers())

# Or create a picker directly
image = hv.Image(frame.Data, kdims=["slow_time", "twtt_us"])
picker = GroundingLinePicker(image)
picker.panel()

# Export picks to CSV
picker.to_csv("picks.csv")
```

## License

`xopr-viewer` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
