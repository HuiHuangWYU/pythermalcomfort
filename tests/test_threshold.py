from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from pythermalcomfort.plots.threshold import (
    Threshold,
    ThresholdPlotResult,
    ThresholdsConfig,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PMV_THRESHOLDS = ThresholdsConfig(thresholds=[-0.5, 0.5])


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


def vectorized_attribute_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(
        pmv=(tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0,
        ppd=np.full_like(tdb_arr, 10.0, dtype=float),
    )


def vectorized_mapping_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return {
        "pmv": (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0,
        "ppd": np.full_like(tdb_arr, 10.0, dtype=float),
    }


def required_input_model(tdb, rh, met):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    met_arr = np.asarray(met, dtype=float)
    return SimpleNamespace(pmv=tdb_arr + rh_arr + met_arr)


def linked_input_model(tdb, tr, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    tr_arr = np.asarray(tr, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=(tdb_arr + tr_arr + rh_arr) / 10.0)


def mismatched_payload_model(tdb, rh):
    _ = np.asarray(tdb, dtype=float)
    _ = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=np.array([1.0, 2.0, 3.0], dtype=float))


def outer_margin_invalid_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    pmv = (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0
    valid = (tdb_arr >= 22.0) & (tdb_arr <= 28.0) & (rh_arr >= 30.0) & (rh_arr <= 70.0)
    return SimpleNamespace(pmv=np.where(valid, pmv, np.nan))


def all_invalid_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    _ = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=np.full_like(tdb_arr, np.nan, dtype=float))


def internal_hole_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    pmv = (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0
    hole = ((tdb_arr - 25.0) ** 2) / 4.0 + ((rh_arr - 50.0) ** 2) / 225.0 <= 1.0
    return SimpleNamespace(pmv=np.where(hole, np.nan, pmv))


def assert_axis_limits(
    result: ThresholdPlotResult,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    assert result.ax.get_xlim() == pytest.approx(xlim)
    assert result.ax.get_ylim() == pytest.approx(ylim)


@pytest.fixture
def base_threshold() -> Threshold:
    return Threshold(vectorized_attribute_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))


def test_parameters_rejects_invalid_parameter_name() -> None:
    threshold = Threshold(vectorized_attribute_model)

    with pytest.raises(ValueError, match="invalid parameter"):
        threshold.parameters(bad_name=1)


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
    threshold = Threshold(vectorized_attribute_model).parameters(
        **{parameter_name: 25.0}
    )
    builder = getattr(threshold, builder_name)

    with pytest.raises(ValueError, match="already contains axis parameter"):
        builder(**{parameter_name: axis_range})

    threshold = Threshold(vectorized_attribute_model)
    builder = getattr(threshold, builder_name)
    builder(**{parameter_name: axis_range})

    with pytest.raises(ValueError, match="cannot set axis parameter"):
        threshold.parameters(**{parameter_name: 25.0})


def test_plot_rejects_empty_output_name(base_threshold: Threshold) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        base_threshold.plot(
            output="   ",
            thresholds=PMV_THRESHOLDS,
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_requires_thresholds_config(base_threshold: Threshold) -> None:
    with pytest.raises(TypeError, match="thresholds"):
        base_threshold.plot(output="pmv", x_resolution=1.0, y_resolution=10.0)


def test_plot_rejects_non_thresholds_config(base_threshold: Threshold) -> None:
    with pytest.raises(TypeError, match="ThresholdsConfig"):
        base_threshold.plot(
            output="pmv",
            thresholds=[-0.5, 0.5],
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_rejects_invalid_threshold_config_values(
    base_threshold: Threshold,
) -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        base_threshold.plot(
            output="pmv",
            thresholds=ThresholdsConfig(thresholds=[]),
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_raises_for_missing_required_model_inputs() -> None:
    threshold = Threshold(required_input_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))

    with pytest.raises(ValueError, match="Missing required parameter"):
        threshold.plot(
            output="pmv",
            thresholds=PMV_THRESHOLDS,
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_uses_provided_subplot_axis(base_threshold: Threshold) -> None:
    fig, ax = plt.subplots()

    result = base_threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        ax=ax,
    )

    assert isinstance(result, ThresholdPlotResult)
    assert result.ax is ax
    assert len(result.fills) > 0


def test_plot_invalid_outer_margins_preserve_requested_limits() -> None:
    threshold = (
        Threshold(outer_margin_invalid_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))
    )

    result = threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        legend=False,
        show_lines=False,
    )

    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))


def test_plot_invalid_regions_add_out_of_model_limits_legend_entry() -> None:
    threshold = (
        Threshold(outer_margin_invalid_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))
    )

    result = threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        show_lines=False,
    )

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" in legend_labels


def test_plot_all_valid_regions_do_not_add_invalid_legend_entry(
    base_threshold: Threshold,
) -> None:
    result = base_threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        show_lines=False,
    )

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" not in legend_labels


def test_plot_internal_invalid_holes_preserve_requested_limits() -> None:
    threshold = Threshold(internal_hole_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))

    result = threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        legend=False,
        show_lines=False,
    )

    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))


def test_plot_all_invalid_surface_renders_and_preserves_requested_limits() -> None:
    threshold = Threshold(all_invalid_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))

    result = threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
    )

    assert isinstance(result, ThresholdPlotResult)
    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))
    assert result.legend is not None


def test_plot_supports_default_link_tr_from_tdb() -> None:
    result = (
        Threshold(linked_input_model)
        .x(tdb=(20.0, 30.0))
        .y(rh=(20.0, 80.0))
        .plot(
            output="pmv",
            thresholds=ThresholdsConfig(thresholds=[8.0]),
            x_resolution=1.0,
            y_resolution=10.0,
        )
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_supports_default_link_tdb_from_tr() -> None:
    result = (
        Threshold(linked_input_model)
        .x(tr=(20.0, 30.0))
        .y(rh=(20.0, 80.0))
        .plot(
            output="pmv",
            thresholds=ThresholdsConfig(thresholds=[8.0]),
            x_resolution=1.0,
            y_resolution=10.0,
        )
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_smoke_returns_editable_lines_present_fills_title_and_legend(
    base_threshold: Threshold,
) -> None:
    result = base_threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        title="Demo Threshold Plot",
        legend=True,
        show_lines=True,
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0
    assert result.ax.get_title() == "Demo Threshold Plot"
    assert result.legend is not None
    assert result.lines
    assert all(isinstance(line, Line2D) for line in result.lines)

    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert legend_labels == ["PMV < -0.5", "-0.5 <= PMV <= 0.5", "PMV > 0.5"]

    result.lines[0].set_linewidth(2.5)
    assert result.lines[0].get_linewidth() == 2.5


def test_plot_with_show_lines_false_returns_empty_lines_and_nonempty_fills(
    base_threshold: Threshold,
) -> None:
    result = base_threshold.plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        x_resolution=1.0,
        y_resolution=10.0,
        show_lines=False,
    )

    assert result.lines == []
    assert len(result.fills) > 0


def test_plot_uses_custom_labels_in_legend(base_threshold: Threshold) -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = base_threshold.plot(
        output="pmv",
        thresholds=ThresholdsConfig(
            thresholds=[-0.5, 0.5],
            labels=custom_labels,
        ),
        x_resolution=1.0,
        y_resolution=10.0,
        legend=True,
    )

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert legend_labels == custom_labels


def test_plot_rejects_wrong_color_count(base_threshold: Threshold) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        base_threshold.plot(
            output="pmv",
            thresholds=ThresholdsConfig(
                thresholds=[-0.5, 0.5],
                colors=["#4c78a8", "#e15759"],
            ),
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_rejects_wrong_label_count(base_threshold: Threshold) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        base_threshold.plot(
            output="pmv",
            thresholds=ThresholdsConfig(
                thresholds=[-0.5, 0.5],
                labels=["Cold", "Hot"],
            ),
            x_resolution=1.0,
            y_resolution=10.0,
        )


def test_plot_rejects_contour_payload_size_mismatch() -> None:
    threshold = (
        Threshold(mismatched_payload_model).x(tdb=(20.0, 30.0)).y(rh=(20.0, 80.0))
    )

    with pytest.raises(
        ValueError, match="Model output shape does not match the contour grid"
    ):
        threshold.plot(
            output="pmv",
            thresholds=PMV_THRESHOLDS,
            x_resolution=1.0,
            y_resolution=10.0,
        )


@pytest.mark.parametrize(
    "fill_kws",
    [
        {"color": "red"},
        {"facecolor": "red"},
    ],
)
def test_plot_rejects_fill_kws_color_override(
    base_threshold: Threshold,
    fill_kws: dict[str, str],
) -> None:
    with pytest.raises(
        ValueError, match="fill_kws cannot include 'color' or 'facecolor'"
    ):
        base_threshold.plot(
            output="pmv",
            thresholds=PMV_THRESHOLDS,
            x_resolution=1.0,
            y_resolution=10.0,
            fill_kws=fill_kws,
        )


def test_plot_extracts_output_from_mapping_result() -> None:
    result = (
        Threshold(vectorized_mapping_model)
        .x(tdb=(20.0, 30.0))
        .y(rh=(20.0, 80.0))
        .plot(
            output="pmv",
            thresholds=PMV_THRESHOLDS,
            x_resolution=1.0,
            y_resolution=10.0,
        )
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0
