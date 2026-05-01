# Plotting Overview

`pythermalcomfort.plots.matplotlib` provides class-based plotting utilities for threshold-region and summary visualizations.

The API is intentionally compact:
- `ThresholdPlot`: build threshold-region charts by configuring model, axes, fixed parameters, and regions.
- `SummaryPlot`: build compact horizontal/vertical summaries from DataFrame outputs.
- `ThresholdsConfig`: shared region configuration dataclass.

Both APIs return Matplotlib handles so users can customize visuals with standard Matplotlib code.

## Quick Start

```python
import matplotlib.pyplot as plt

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import SummaryPlot, ThresholdPlot

threshold = (
    ThresholdPlot(pmv_ppd_iso)
    .set_x_axis("tdb", 18.0, 34.0, resolution=0.2)
    .set_y_axis("rh", 20.0, 100.0, resolution=0.5)
    .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
    .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    .plot(title="PMV Threshold Regions")
)

# Full matplotlib customization remains available
threshold.ax.set_xlabel("Air temperature [degC]")
threshold.ax.set_ylabel("Relative humidity [%]")

df = ...
summary = (
    SummaryPlot(df)
    .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    .plot(title="Measured PMV Summary")
)

plt.show()
```

## Design Notes

- Keep configuration explicit and ordered: set axes and params first, then regions, then render.
- Keep implementation simple and maintainable.
- Keep output open for customization by returning artist handles (`ax`, fills, lines, legend, processed data).
