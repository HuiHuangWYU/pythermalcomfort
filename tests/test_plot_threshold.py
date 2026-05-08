from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

from pythermalcomfort.plots.matplotlib import ThresholdsConfig
from pythermalcomfort.plots.matplotlib.threshold import (
    OUT_OF_MODEL_LIMITS_COLOR,
    ThresholdPlot,
    ThresholdPlotResult,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D


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


def _new_plot() -> ThresholdPlot:
    return (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def assert_axis_limits(
    result: ThresholdPlotResult,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    assert result.ax.get_xlim() == pytest.approx(xlim)
    assert result.ax.get_ylim() == pytest.approx(ylim)


def test_set_params_rejects_invalid_parameter_name() -> None:
    plot = ThresholdPlot(vectorized_attribute_model)

    with pytest.raises(ValueError, match="were not found"):
        plot.set_params(bad_name=1)


def test_set_axis_rejects_parameter_conflicts_with_set_params() -> None:
    plot = ThresholdPlot(vectorized_attribute_model).set_params(tdb=25.0)

    with pytest.raises(ValueError, match="already contains axis parameter"):
        plot.set_x_axis("tdb", 20.0, 30.0, resolution=1.0)

    plot = ThresholdPlot(vectorized_attribute_model).set_x_axis(
        "tdb", 20.0, 30.0, resolution=1.0
    )

    with pytest.raises(ValueError, match="cannot set axis parameter"):
        plot.set_params(tdb=25.0)


def test_plot_rejects_when_regions_not_set() -> None:
    plot = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
    )

    with pytest.raises(ValueError, match="Call set_regions"):
        plot.plot()


def test_plot_raises_for_missing_required_model_inputs() -> None:
    plot = (
        ThresholdPlot(required_input_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        plot.plot()


def test_plot_uses_provided_subplot_axis() -> None:
    fig, ax = plt.subplots()

    result = _new_plot().plot(ax=ax)

    assert isinstance(result, ThresholdPlotResult)
    assert result.ax is ax
    assert result.fig is fig
    assert len(result.fills) > 0


def test_plot_invalid_outer_margins_preserve_requested_limits() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(legend=False, show_lines=False)

    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))


def test_plot_invalid_regions_add_out_of_model_limits_legend_entry() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False)

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" in legend_labels


def test_plot_uses_default_invalid_color_in_legend() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False)

    assert result.legend is not None
    legend_patches = result.legend.get_patches()
    invalid_patch = next(
        patch
        for patch, text in zip(legend_patches, result.legend.get_texts(), strict=False)
        if text.get_text() == "Out of model limits"
    )
    assert invalid_patch.get_facecolor()[:3] == pytest.approx(
        to_rgb(OUT_OF_MODEL_LIMITS_COLOR), abs=1e-3
    )


def test_plot_allows_custom_invalid_color() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False, invalid_color="#ff00ff")

    assert result.legend is not None
    legend_patches = result.legend.get_patches()
    invalid_patch = next(
        patch
        for patch, text in zip(legend_patches, result.legend.get_texts(), strict=False)
        if text.get_text() == "Out of model limits"
    )
    assert invalid_patch.get_facecolor()[:3] == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)


def test_plot_rejects_invalid_invalid_color() -> None:
    with pytest.raises(ValueError, match="invalid_color"):
        _new_plot().plot(invalid_color="not-a-color")


def test_plot_allows_custom_legend_kwargs() -> None:
    result = _new_plot().plot(legend_kws={"ncol": 1, "loc": "upper right"})

    assert result.legend is not None


def test_plot_all_valid_regions_do_not_add_invalid_legend_entry() -> None:
    result = _new_plot().plot(show_lines=False)

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" not in legend_labels


def test_plot_internal_invalid_holes_preserve_requested_limits() -> None:
    plot = (
        ThresholdPlot(internal_hole_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(legend=False, show_lines=False)

    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))


def test_plot_all_invalid_surface_renders_and_preserves_requested_limits() -> None:
    plot = (
        ThresholdPlot(all_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot()

    assert isinstance(result, ThresholdPlotResult)
    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))
    assert result.legend is not None


def test_plot_supports_default_link_tr_from_tdb() -> None:
    result = (
        ThresholdPlot(linked_input_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[8.0])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_supports_default_link_tdb_from_tr() -> None:
    result = (
        ThresholdPlot(linked_input_model)
        .set_x_axis("tr", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[8.0])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_smoke_returns_editable_lines_present_fills_title_and_legend() -> None:
    result = _new_plot().plot(
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
    assert legend_labels == ["PMV < -0.5", "-0.5 ≤ PMV < 0.5", "PMV ≥ 0.5"]

    result.lines[0].set_linewidth(2.5)
    assert result.lines[0].get_linewidth() == 2.5


def test_plot_with_show_lines_false_returns_empty_lines_and_nonempty_fills() -> None:
    result = _new_plot().plot(show_lines=False)

    assert result.lines == []
    assert len(result.fills) > 0


def test_plot_uses_custom_labels_in_legend() -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=custom_labels)
        .plot(legend=True)
    )

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert legend_labels == custom_labels


def test_plot_rejects_wrong_color_count() -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(
                output="pmv",
                thresholds=[-0.5, 0.5],
                colors=["#4c78a8", "#e15759"],
            )
        )


def test_plot_rejects_wrong_label_count() -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(
                output="pmv",
                thresholds=[-0.5, 0.5],
                labels=["Cold", "Hot"],
            )
        )


def test_plot_rejects_contour_payload_size_mismatch() -> None:
    plot = (
        ThresholdPlot(mismatched_payload_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(
        ValueError, match="Model output shape does not match the contour grid"
    ):
        plot.plot()


@pytest.mark.parametrize(
    "fill_kws",
    [
        {"color": "red"},
        {"facecolor": "red"},
    ],
)
def test_plot_rejects_fill_kws_color_override(fill_kws: dict[str, str]) -> None:
    with pytest.raises(
        ValueError, match="fill_kws cannot include 'color' or 'facecolor'"
    ):
        _new_plot().plot(fill_kws=fill_kws)


def test_plot_extracts_output_from_mapping_result() -> None:
    result = (
        ThresholdPlot(vectorized_mapping_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_coarse_resolution_still_renders() -> None:
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 20.5, resolution=1.0)
        .set_y_axis("rh", 20.0, 25.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )
    assert isinstance(result, ThresholdPlotResult)
    assert result.ax.get_xlim() == pytest.approx((20.0, 20.5))
    assert result.ax.get_ylim() == pytest.approx((20.0, 25.0))


def test_plot_with_thresholds_config() -> None:
    config = ThresholdsConfig(
        thresholds=[-0.5, 0.5],
        labels=["Cool", "Comfortable", "Warm"],
        colors=["#A3D1FF", "#A8E6CF", "#FFB7B2"],
    )
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=config)
        .plot()
    )
    assert isinstance(result, ThresholdPlotResult)
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert legend_labels == ["Cool", "Comfortable", "Warm"]


def test_plot_rejects_thresholds_config_with_separate_labels() -> None:
    config = ThresholdsConfig(thresholds=[-0.5, 0.5])
    with pytest.raises(ValueError, match="must not be provided separately"):
        (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(output="pmv", thresholds=config, labels=["A", "B", "C"])
        )


def test_plot_grid_includes_exact_endpoints() -> None:
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=4.0)
        .set_y_axis("rh", 20.0, 75.0, resolution=8.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )
    assert result.ax.get_xlim() == pytest.approx((20.0, 30.0))
    assert result.ax.get_ylim() == pytest.approx((20.0, 75.0))
