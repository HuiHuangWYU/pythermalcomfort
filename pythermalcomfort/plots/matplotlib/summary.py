"""Class-based summary plotting for threshold regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pythermalcomfort.plots.matplotlib._shared import (
    ThresholdsConfig,
    _build_region_labels,
    _is_light_color,
    _normalize_levels,
    _resolve_region_colors,
)


@dataclass
class SummaryPlotResult:
    """Container with handles and processed data from :meth:`Summary.plot`."""

    fig: Figure
    ax: Axes
    data: pd.DataFrame
    region_percentages: pd.Series
    artists: list[Any]


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame for summary plotting."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must contain at least one row.")


def _validate_output_column(df: pd.DataFrame, output: str) -> str:
    """Validate output column name and ensure it exists."""
    if not isinstance(output, str):
        raise TypeError("output must be a string.")

    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    if output_name not in df.columns:
        msg = f"Summary requires an existing output column '{output_name}' in the DataFrame."
        raise ValueError(msg)

    return output_name


def _validate_output_values(df: pd.DataFrame, output_column: str) -> None:
    """Ensure output column contains numeric finite values only."""
    numeric_values = pd.to_numeric(df[output_column], errors="coerce")
    invalid_mask = ~numeric_values.notna()
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        msg = (
            f"output column '{output_column}' contains {invalid_count} non-numeric "
            "or missing value(s)."
        )
        raise ValueError(msg)


def _categorize_output_values(
    df_copy: pd.DataFrame,
    *,
    output_column: str,
    label_column: str,
    levels: Sequence[float],
    region_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Assign each row to a threshold region and compute region percentages."""
    bins = [-float("inf"), *levels, float("inf")]
    df_copy[label_column] = pd.cut(
        df_copy[output_column],
        bins=bins,
        labels=region_labels,
        right=False,
    )

    region_percentages = (
        df_copy[label_column]
        .value_counts(normalize=True)
        .reindex(region_labels, fill_value=0.0)
        .mul(100)
        .round(1)
    )

    return df_copy, region_percentages


def _prepare_axis(ax: Axes, *, title: str | None) -> None:
    """Prepare a clean axis for summary bar rendering."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(title, fontsize=13, pad=10)


def _plot_horizontal_summary(
    ax: Axes,
    *,
    region_percentages: pd.Series,
    region_labels: Sequence[str],
    region_colors: Sequence[str],
) -> list[Any]:
    """Render horizontal stacked summary bar and annotations."""
    artists: list[Any] = []
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)

    left = 0.0
    bar_y = 0.0
    bar_height = 0.36

    for label, color in zip(region_labels, region_colors, strict=False):
        value = float(region_percentages[label])
        bar = ax.barh(
            y=bar_y,
            width=value,
            left=left,
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        artists.append(bar)

        if value > 0:
            is_light = _is_light_color(color)
            text_color = "black" if is_light else "white"
            artists.append(
                ax.text(
                    left + value / 2,
                    bar_y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )
            )
            label_color = "dimgray" if is_light else color
            artists.append(
                ax.text(
                    left + value / 2,
                    0.34,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color=label_color,
                )
            )

        left += value

    return artists


def _plot_vertical_summary(
    ax: Axes,
    *,
    region_percentages: pd.Series,
    region_labels: Sequence[str],
    region_colors: Sequence[str],
) -> list[Any]:
    """Render vertical stacked summary bar and annotations."""
    artists: list[Any] = []
    ax.set_xlim(-0.75, 0.9)
    ax.set_ylim(0, 100)

    bottom = 0.0
    x = 0.0
    width = 0.42

    for label, color in zip(region_labels, region_colors, strict=False):
        value = float(region_percentages[label])
        bar = ax.bar(
            x=x,
            height=value,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=1.0,
        )
        artists.append(bar)

        if value > 0:
            center_y = bottom + value / 2
            is_light = _is_light_color(color)
            text_color = "black" if is_light else "white"
            artists.append(
                ax.text(
                    x,
                    center_y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color=text_color,
                )
            )
            label_color = "dimgray" if is_light else color
            artists.append(
                ax.text(
                    x + 0.38,
                    center_y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=11,
                    color=label_color,
                )
            )

        bottom += value

    return artists


class Summary:
    """Build and render a threshold summary plot from tabular outputs."""

    def __init__(self, df: pd.DataFrame | None = None) -> None:
        self._df = df

    @classmethod
    def data(cls, df: pd.DataFrame) -> Summary:
        """Create a :class:`Summary` configured with a DataFrame."""
        _validate_dataframe(df)
        return cls(df)

    def _validate_plot_inputs(
        self, *, output: str, thresholds: ThresholdsConfig
    ) -> str:
        """Validate plot inputs and return normalized output column name."""
        if self._df is None:
            raise ValueError("No dataframe set. Call data(df) before plot(...).")

        output_name = _validate_output_column(self._df, output)
        _validate_output_values(self._df, output_name)

        if not isinstance(thresholds, ThresholdsConfig):
            raise TypeError("thresholds must be a ThresholdsConfig.")

        return output_name

    def plot(
        self,
        *,
        output: str,
        thresholds: ThresholdsConfig,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
    ) -> SummaryPlotResult:
        """Render a threshold summary plot for one output column."""
        output_name = self._validate_plot_inputs(output=output, thresholds=thresholds)

        normalized_levels = _normalize_levels(thresholds.thresholds)
        region_labels = _build_region_labels(
            output=output_name,
            levels=normalized_levels,
            labels=thresholds.labels,
        )
        region_colors = _resolve_region_colors(
            n_regions=len(region_labels),
            colors=thresholds.colors,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        else:
            fig = ax.figure

        output_column = output_name
        label_column = f"{output_column}_label"

        df_copy = self._df.copy()
        df_copy, region_percentages = _categorize_output_values(
            df_copy,
            output_column=output_column,
            label_column=label_column,
            levels=normalized_levels,
            region_labels=region_labels,
        )

        _prepare_axis(ax, title=title)
        if vertical:
            artists = _plot_vertical_summary(
                ax,
                region_percentages=region_percentages,
                region_labels=region_labels,
                region_colors=region_colors,
            )
        else:
            artists = _plot_horizontal_summary(
                ax,
                region_percentages=region_percentages,
                region_labels=region_labels,
                region_colors=region_colors,
            )

        return SummaryPlotResult(
            fig=fig,
            ax=ax,
            data=df_copy,
            region_percentages=region_percentages,
            artists=artists,
        )
