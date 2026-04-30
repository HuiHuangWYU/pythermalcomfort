from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythermalcomfort.plots.matplotlib.summary import Summary, SummaryPlotResult
from pythermalcomfort.plots.matplotlib.threshold import ThresholdsConfig

PMV_THRESHOLDS = ThresholdsConfig(thresholds=[-0.5, 0.5])


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


def test_data_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        Summary.data([1, 2, 3])


def test_data_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        Summary.data(pd.DataFrame())


def test_plot_requires_data_before_plot() -> None:
    with pytest.raises(ValueError, match="Call data\\(df\\) before plot"):
        Summary().plot(output="pmv", thresholds=PMV_THRESHOLDS)


def test_plot_requires_thresholds_config(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="thresholds"):
        Summary.data(pmv_df).plot(output="pmv")


def test_plot_rejects_non_thresholds_config(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="ThresholdsConfig"):
        Summary.data(pmv_df).plot(output="pmv", thresholds=[-0.5, 0.5])


def test_plot_rejects_invalid_threshold_config_values(
    pmv_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="at least one threshold"):
        Summary.data(pmv_df).plot(
            output="pmv",
            thresholds=ThresholdsConfig(thresholds=[]),
        )


def test_plot_rejects_empty_output_name(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        Summary.data(pmv_df).plot(output="   ", thresholds=PMV_THRESHOLDS)


def test_plot_rejects_missing_output_column(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="Summary requires an existing output column 'utci' in the DataFrame.",
    ):
        Summary.data(pmv_df).plot(
            output="utci",
            thresholds=ThresholdsConfig(thresholds=[9, 26]),
        )


def test_plot_rejects_wrong_label_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        Summary.data(pmv_df).plot(
            output="pmv",
            thresholds=ThresholdsConfig(
                thresholds=[-0.5, 0.5],
                labels=["Cold", "Hot"],
            ),
        )


def test_plot_rejects_wrong_color_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        Summary.data(pmv_df).plot(
            output="pmv",
            thresholds=ThresholdsConfig(
                thresholds=[-0.5, 0.5],
                colors=["#4c78a8", "#e15759"],
            ),
        )


def test_plot_uses_provided_axis(pmv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()

    result = Summary.data(pmv_df).plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        ax=ax,
    )

    assert result.ax is ax
    assert result.fig is fig


def test_plot_vertical_mode_executes(pmv_df: pd.DataFrame) -> None:
    result = Summary.data(pmv_df).plot(
        output="pmv",
        thresholds=PMV_THRESHOLDS,
        vertical=True,
    )

    assert isinstance(result, SummaryPlotResult)
    assert len(result.artists) > 0


def test_plot_returns_processed_data(pmv_df: pd.DataFrame) -> None:
    result = Summary.data(pmv_df).plot(output="pmv", thresholds=PMV_THRESHOLDS)

    assert result.data is not pmv_df
    assert result.data["pmv"].tolist() == pmv_df["pmv"].tolist()
    assert "pmv_label" in result.data.columns
    assert result.data["pmv_label"].astype(str).tolist() == [
        "PMV < -0.5",
        "-0.5 <= PMV <= 0.5",
        "PMV > 0.5",
    ]


def test_plot_returns_region_percentages(pmv_df: pd.DataFrame) -> None:
    result = Summary.data(pmv_df).plot(output="pmv", thresholds=PMV_THRESHOLDS)

    expected_labels = ["PMV < -0.5", "-0.5 <= PMV <= 0.5", "PMV > 0.5"]
    assert result.region_percentages.index.tolist() == expected_labels
    assert result.region_percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_plot_uses_custom_labels_when_provided(pmv_df: pd.DataFrame) -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = Summary.data(pmv_df).plot(
        output="pmv",
        thresholds=ThresholdsConfig(
            thresholds=[-0.5, 0.5],
            labels=custom_labels,
        ),
    )

    assert result.region_percentages.index.tolist() == custom_labels
    assert list(result.data["pmv_label"].cat.categories) == custom_labels


def test_plot_supports_pmv_like_existing_column(pmv_df: pd.DataFrame) -> None:
    result = Summary.data(pmv_df).plot(output="pmv", thresholds=PMV_THRESHOLDS)

    assert isinstance(result, SummaryPlotResult)
    assert "pmv_label" in result.data.columns


def test_plot_supports_utci_like_existing_column() -> None:
    df = pd.DataFrame(
        {
            "tdb": [10.0, 24.0, 32.0],
            "utci": [5.0, 18.0, 28.0],
        }
    )

    result = Summary.data(df).plot(
        output="utci",
        thresholds=ThresholdsConfig(thresholds=[9, 26]),
    )

    expected_labels = ["UTCI < 9", "9 <= UTCI <= 26", "UTCI > 26"]
    assert "utci_label" in result.data.columns
    assert result.region_percentages.index.tolist() == expected_labels
    assert result.region_percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])
