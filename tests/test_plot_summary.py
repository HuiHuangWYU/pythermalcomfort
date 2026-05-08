from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from pythermalcomfort.plots.matplotlib.summary import (
    SummaryPlot,
    SummaryPlotResult,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


@pytest.fixture
def pmv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tdb": [20.0, 25.0, 30.0],
            "rh": [50.0, 50.0, 50.0],
            "pmv": [-0.6, 0.0, 0.7],
        }
    )


def _new_summary(pmv_df: pd.DataFrame) -> SummaryPlot:
    return SummaryPlot(pmv_df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SummaryPlot([1, 2, 3])


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        SummaryPlot(pd.DataFrame())


def test_plot_requires_set_regions(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Call set_regions"):
        SummaryPlot(pmv_df).plot()


def test_set_regions_rejects_empty_output_name(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        SummaryPlot(pmv_df).set_regions(output="   ", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_missing_output_column(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="output column 'utci' was not found in the DataFrame.",
    ):
        SummaryPlot(pmv_df).set_regions(output="utci", thresholds=[9, 26])


def test_set_regions_rejects_wrong_label_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        SummaryPlot(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cold", "Hot"],
        )


def test_set_regions_rejects_wrong_color_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        SummaryPlot(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            colors=["#4c78a8", "#e15759"],
        )


def test_plot_uses_provided_axis(pmv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()

    result = _new_summary(pmv_df).plot(ax=ax)

    assert result.ax is ax
    assert result.fig is fig


def test_plot_vertical_mode_executes(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True)

    assert isinstance(result, SummaryPlotResult)
    assert len(result.artists) > 0


def test_plot_returns_percentages(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    expected_labels = ["PMV < -0.5", "-0.5 ≤ PMV < 0.5", "PMV ≥ 0.5"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_plot_uses_custom_labels_when_provided(pmv_df: pd.DataFrame) -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = (
        SummaryPlot(pmv_df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=custom_labels,
        )
        .plot()
    )

    assert result.percentages.index.tolist() == custom_labels


def test_plot_supports_utci_like_existing_column() -> None:
    df = pd.DataFrame(
        {
            "tdb": [10.0, 24.0, 32.0],
            "utci": [5.0, 18.0, 28.0],
        }
    )

    result = SummaryPlot(df).set_regions(output="utci", thresholds=[9, 26]).plot()

    expected_labels = ["UTCI < 9", "9 ≤ UTCI < 26", "UTCI ≥ 26"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_set_regions_rejects_non_numeric_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, "bad", 0.2]})

    with pytest.raises(ValueError, match="non-numeric"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_non_finite_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, float("inf"), 0.2]})

    with pytest.raises(ValueError, match="non-finite"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_summary_with_custom_labels() -> None:
    df = pd.DataFrame({"pmv": [0.7, -0.3, 0.1, -0.8, 1.2]})
    result = (
        SummaryPlot(df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cool", "Comfortable", "Warm"],
        )
        .plot()
    )
    assert isinstance(result, SummaryPlotResult)
    assert list(result.percentages.index) == ["Cool", "Comfortable", "Warm"]


def test_summary_handles_numeric_string_column() -> None:
    df = pd.DataFrame({"pmv": ["0.7", "-0.3", "0.1", "-0.8", "1.2"]})
    result = SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5]).plot()
    assert isinstance(result, SummaryPlotResult)
    assert result.percentages.sum() > 99.9


def test_plot_legend_shown_by_default(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert isinstance(result.legend, Legend)


def test_plot_legend_none_when_disabled(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(legend=False)

    assert result.legend is None


def test_plot_result_has_no_data_attribute(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert not hasattr(result, "data")


def test_set_regions_empty_labels_suppresses_label_text(pmv_df: pd.DataFrame) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


def test_empty_labels_suppresses_label_text_via_thresholds(
    pmv_df: pd.DataFrame,
) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]
