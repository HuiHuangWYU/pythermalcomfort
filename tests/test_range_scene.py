from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import pytest

from pythermalcomfort.plots.range_scene import PlotResult
from pythermalcomfort.plots.range_scene import RangeScene
from pythermalcomfort.plots.range_scene import _extract_output_value

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


def attribute_model(tdb, rh):
    return SimpleNamespace(pmv=(tdb - 25.0) / 10.0, ppd=10.0)


def mapping_model(tdb, rh):
    return {"pmv": (tdb - 25.0) / 10.0, "ppd": 10.0}


def scalar_model(tdb, rh):
    return float(tdb)


def required_input_model(tdb, rh, met):
    return SimpleNamespace(pmv=tdb + rh + met)


def test_parameters_rejects_invalid_parameter_name() -> None:
    scene = RangeScene(attribute_model)

    with pytest.raises(ValueError, match="invalid parameter"):
        scene.parameters(bad_name=1)


@pytest.mark.parametrize(
    ("builder_name", "parameter_name", "axis_range"),
    [
        ("x", "tdb", (20.0, 30.0)),
        ("y", "rh", (20.0, 80.0)),
    ],
)
def test_x_and_y_reject_parameter_conflicts_with_parameters(
    builder_name: str,
    parameter_name: str,
    axis_range: tuple[float, float],
) -> None:
    scene = RangeScene(attribute_model).parameters(**{parameter_name: 25.0})
    builder = getattr(scene, builder_name)

    with pytest.raises(ValueError, match="already contains axis parameter"):
        builder(**{parameter_name: axis_range})

    scene = RangeScene(attribute_model)
    builder = getattr(scene, builder_name)
    builder(**{parameter_name: axis_range})

    with pytest.raises(ValueError, match="cannot set axis parameter"):
        scene.parameters(**{parameter_name: 25.0})


def test_plot_raises_for_missing_required_model_inputs() -> None:
    scene = RangeScene(required_input_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))

    with pytest.raises(ValueError, match="Missing required parameter"):
        scene.plot(
            output="pmv",
            levels=[-0.5, 0.5],
            x_step=1.0,
            y_step=10.0,
        )


def test_plot_uses_provided_subplot_axis() -> None:
    fig, ax = plt.subplots()

    result = (
        RangeScene(attribute_model)
        .x(tdb=(20.0, 30.0))
        .y(rh=(20.0, 80.0))
        .plot(
            output="pmv",
            levels=[-0.5, 0.5],
            x_step=1.0,
            y_step=10.0,
            ax=ax,
        )
    )

    assert isinstance(result, PlotResult)
    assert result.ax is ax
    assert len(result.fills) > 0
    plt.close(fig)


def test_extract_output_value_errors_are_clear_for_mapping_and_scalar_results() -> None:
    with pytest.raises(ValueError, match="Available outputs: pmv, ppd"):
        _extract_output_value({"pmv": 0.1, "ppd": 5.0}, "utci")

    with pytest.raises(
        ValueError,
        match="scalar model result.*output='value'",
    ):
        _extract_output_value(25.0, "utci")


def test_multiple_plot_calls_return_independent_plot_results() -> None:
    scene = RangeScene(attribute_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))

    result1 = scene.plot(
        output="pmv",
        levels=[-0.5, 0.5],
        x_step=1.0,
        y_step=10.0,
    )
    result2 = scene.plot(
        output="pmv",
        levels=[-0.5, 0.5],
        x_step=1.0,
        y_step=10.0,
    )

    assert result1 is not result2
    assert result1.ax is not result2.ax
    assert result1.lines is not result2.lines
    assert result1.fills is not result2.fills
    assert result1.fills[0] is not result2.fills[0]


def test_plot_with_show_lines_false_returns_empty_lines_and_nonempty_fills() -> None:
    result = (
        RangeScene(attribute_model)
        .x(tdb=(20.0, 30.0))
        .y(rh=(20.0, 80.0))
        .plot(
            output="pmv",
            levels=[-0.5, 0.5],
            x_step=1.0,
            y_step=10.0,
            show_lines=False,
        )
    )

    assert result.lines == []
    assert len(result.fills) > 0
