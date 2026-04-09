"""Simple UTCI threshold-fill example using RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt

from pythermalcomfort.models import utci
from pythermalcomfort.plots.range_scene import RangeScene


def main() -> None:
    scene = (
        RangeScene(model_func=utci)
        .x(tdb=(-10.0, 40.0))
        .y(rh=(20.0, 100.0))
        .parameters(
            v=1.0,
            units="SI",
            limit_inputs=True,
            round_output=False,
        )
    )

    result = scene.plot(
        output="utci",
        levels=[26.0, 32.0],
        colors=["#4c78a8", "#f2d7b6", "#e15759"],
        x_step=1.0,
        y_step=2.0,
        line_kws={"linewidth": 2.0, "linestyle": "--"},
        fill_kws={"alpha": 0.65},
    )

    result.ax.set_title("UTCI Threshold-Filled Regions")
    result.ax.set_xlabel("Air temperature [degC]")
    result.ax.set_ylabel("Relative humidity [%]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
