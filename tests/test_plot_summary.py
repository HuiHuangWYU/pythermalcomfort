from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythermalcomfort.plots.matplotlib.summary import (
    SummaryPlot,
    SummaryPlotResult,
    ThresholdsConfig,
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


def test_plot_returns_processed_data(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert result.data is not pmv_df
    assert result.data["pmv"].tolist() == pmv_df["pmv"].tolist()
    assert "pmv_label" in result.data.columns
    assert result.data["pmv_label"].astype(str).tolist() == [
        "PMV < -0.5",
        "-0.5 <= PMV < 0.5",
        "PMV >= 0.5",
    ]


def test_plot_returns_region_percentages(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    expected_labels = ["PMV < -0.5", "-0.5 <= PMV < 0.5", "PMV >= 0.5"]
    assert result.region_percentages.index.tolist() == expected_labels
    assert result.region_percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


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

    assert result.region_percentages.index.tolist() == custom_labels
    assert list(result.data["pmv_label"].cat.categories) == custom_labels


def test_plot_supports_utci_like_existing_column() -> None:
    df = pd.DataFrame(
        {
            "tdb": [10.0, 24.0, 32.0],
            "utci": [5.0, 18.0, 28.0],
        }
    )

    result = SummaryPlot(df).set_regions(output="utci", thresholds=[9, 26]).plot()

    expected_labels = ["UTCI < 9", "9 <= UTCI < 26", "UTCI >= 26"]
    assert "utci_label" in result.data.columns
    assert result.region_percentages.index.tolist() == expected_labels
    assert result.region_percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_set_regions_rejects_non_numeric_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, "bad", 0.2]})

    with pytest.raises(ValueError, match="non-numeric"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_non_finite_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, float("inf"), 0.2]})

    with pytest.raises(ValueError, match="non-finite"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_summary_with_thresholds_config() -> None:
    config = ThresholdsConfig(
        thresholds=[-0.5, 0.5],
        labels=["Cool", "Comfortable", "Warm"],
    )
    df = pd.DataFrame({"pmv": [0.7, -0.3, 0.1, -0.8, 1.2]})
    result = SummaryPlot(df).set_regions(output="pmv", thresholds=config).plot()
    assert isinstance(result, SummaryPlotResult)
    assert list(result.region_percentages.index) == ["Cool", "Comfortable", "Warm"]


def test_summary_handles_numeric_string_column() -> None:
    df = pd.DataFrame({"pmv": ["0.7", "-0.3", "0.1", "-0.8", "1.2"]})
    result = SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5]).plot()
    assert isinstance(result, SummaryPlotResult)
    assert result.region_percentages.sum() > 99.9
