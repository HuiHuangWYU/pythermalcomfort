"""Class-based summary plotting for threshold regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pythermalcomfort.plots.matplotlib._shared import (
    _build_region_labels,
    _is_light_color,
    _normalize_levels,
    _resolve_region_colors,
)


@dataclass
class SummaryPlotResult:
    """Container with handles and processed data from :meth:`SummaryPlot.plot`.

    Attributes:
        fig: Matplotlib figure containing the summary plot.
        ax: Matplotlib axis containing the summary plot.
        data: Copy of input DataFrame with an added output label column.
        region_percentages: Percentage share per region label.
        artists: List of rendered artists for post-customization.
    """

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
        msg = f"output column '{output_name}' was not found in the DataFrame."
        raise ValueError(msg)

    return output_name


def _validate_output_values(df: pd.DataFrame, output_column: str) -> None:
    """Ensure output column contains numeric finite values only."""
    numeric_values = pd.to_numeric(df[output_column], errors="coerce")
    invalid_mask = ~numeric_values.notna() | ~np.isfinite(numeric_values.to_numpy())
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        msg = (
            f"output column '{output_column}' contains {invalid_count} non-numeric, "
            "non-finite, or missing value(s)."
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
    bins = [-np.inf, *levels, np.inf]
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


def _add_center_text(
    ax: Axes,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
) -> Any:
    """Add centered bold text annotation."""
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=color,
    )


def _add_region_annotations(
    ax: Axes,
    *,
    value: float,
    label: str,
    color: str,
    percentage_x: float,
    percentage_y: float,
    label_x: float,
    label_y: float,
    label_ha: str,
    label_va: str,
) -> list[Any]:
    """Add percentage and region-label annotations for one region."""
    is_light = _is_light_color(color)
    percentage_text_color = "black" if is_light else "white"
    label_color = "dimgray" if is_light else color

    artists: list[Any] = [
        _add_center_text(
            ax,
            x=percentage_x,
            y=percentage_y,
            text=f"{value:.1f}%",
            color=percentage_text_color,
        ),
        ax.text(
            label_x,
            label_y,
            label,
            ha=label_ha,
            va=label_va,
            fontsize=11,
            color=label_color,
        ),
    ]
    return artists


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
            artists.extend(
                _add_region_annotations(
                    ax,
                    value=value,
                    label=label,
                    color=color,
                    percentage_x=left + value / 2,
                    percentage_y=bar_y,
                    label_x=left + value / 2,
                    label_y=0.34,
                    label_ha="center",
                    label_va="bottom",
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
            artists.extend(
                _add_region_annotations(
                    ax,
                    value=value,
                    label=label,
                    color=color,
                    percentage_x=x,
                    percentage_y=center_y,
                    label_x=x + 0.38,
                    label_y=center_y,
                    label_ha="left",
                    label_va="center",
                )
            )

        bottom += value

    return artists


class SummaryPlot:
    """Build and render a threshold summary plot from tabular model outputs.

    The class works with an existing DataFrame that already contains the target
    model output column (e.g., ``pmv`` or ``utci``).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a summary plot builder from a DataFrame.

        Args:
            df: Input DataFrame containing at least one output column to summarize.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame.
            ValueError: If ``df`` is empty.
        """
        _validate_dataframe(df)
        self._df = df
        self._output_name: str | None = None
        self._thresholds: list[float] | None = None
        self._region_labels: list[str] | None = None
        self._region_colors: list[str] | None = None

    def set_regions(
        self,
        *,
        output: str,
        thresholds: Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> SummaryPlot:
        """Set output variable and threshold region configuration.

        Args:
            output: Name of the DataFrame column to categorize.
            thresholds: Threshold boundary values for region splitting.
            labels: Optional region labels. Must have length
                ``len(thresholds) + 1`` when provided.
            colors: Optional region colors. Must have length
                ``len(thresholds) + 1`` when provided.

        Returns:
            Self, to support method chaining.

        Raises:
            TypeError: If ``output`` is not a string.
            ValueError: If the output column is missing or has invalid values,
                or if thresholds/labels/colors are invalid.
        """
        output_name = _validate_output_column(self._df, output)
        _validate_output_values(self._df, output_name)

        normalized_levels = _normalize_levels(thresholds)
        region_labels = _build_region_labels(
            output=output_name,
            levels=normalized_levels,
            labels=labels,
        )
        region_colors = _resolve_region_colors(
            n_regions=len(normalized_levels) + 1,
            colors=colors,
        )

        self._output_name = output_name
        self._thresholds = normalized_levels
        self._region_labels = region_labels
        self._region_colors = region_colors
        return self

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
    ) -> SummaryPlotResult:
        """Render a threshold summary plot for the configured output column.

        Args:
            ax: Existing axis to draw on. If ``None``, a new figure/axis is created.
            title: Optional axis title.
            vertical: If ``True``, render a vertical stacked bar; otherwise horizontal.

        Returns:
            :class:`SummaryPlotResult` with figure, axis, processed data, and artists.

        Raises:
            ValueError: If regions are not configured first via :meth:`set_regions`.
        """
        if (
            self._output_name is None
            or self._thresholds is None
            or self._region_labels is None
            or self._region_colors is None
        ):
            raise ValueError(
                "Regions are not set. Call set_regions(...) before plot(...)."
            )
        output_name = self._output_name
        normalized_levels = self._thresholds
        region_labels = self._region_labels
        region_colors = self._region_colors

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
        else:
            fig = ax.figure

        label_column = f"{output_name}_label"
        df_copy = self._df.copy()
        df_copy, region_percentages = _categorize_output_values(
            df_copy,
            output_column=output_name,
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
