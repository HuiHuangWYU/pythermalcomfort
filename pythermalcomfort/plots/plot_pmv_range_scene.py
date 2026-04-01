"""Simple PMV threshold-fill example using RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.range_scene import RangeScene


def main() -> None:
    plt.close("all")

    scene = RangeScene(model_func=pmv_ppd_iso).x(tdb=(18.0, 34.0)).y(
        rh=(20.0, 100.0)
    ).parameters(
        vr=0.10,
        met=1.2,
        clo=0.5,
        wme=0.0,
    )

    result = scene.plot(
        output="pmv",
        levels=[-0.5, 0.5],
        x_step=0.2,
        y_step=0.5,
    )

    result.ax.set_title("PMV Threshold-Filled Regions")
    result.ax.set_xlabel("Air temperature [degC]")
    result.ax.set_ylabel("Relative humidity [%]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
