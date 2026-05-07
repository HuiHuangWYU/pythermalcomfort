"""Class-based summary plotting for threshold regions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from pythermalcomfort.plots.matplotlib._shared import (
    BasePlotResult,
    RegionConfig,
    ThresholdsConfig,
    _configure_regions,
    _is_light_color,
    _PlotDefaults,
)

# ── result container ───────────────────────────────────────────────────────


@dataclass
class SummaryPlotResult(BasePlotResult):
    """Container with handles and processed data from :meth:`SummaryPlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the summary plot.
    ax : Axes
        Matplotlib axis containing the summary plot.
    data : DataFrame
        Copy of input DataFrame with an added output label column.
    region_percentages : Series
        Percentage share per region label.
    artists : list
        List of rendered artists for post-customization.
    """

    data: pd.DataFrame
    region_percentages: pd.Series
    artists: list[Any]


# ── validation helpers ─────────────────────────────────────────────────────


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
    """Ensure output column contains numeric finite values only.

    Raises rather than silently dropping rows so callers are aware of missing
    data and can decide how to handle it before plotting.
    """
    numeric_values = pd.to_numeric(df[output_column], errors="coerce")
    invalid_mask = ~numeric_values.notna() | ~np.isfinite(numeric_values.to_numpy())
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        msg = (
            f"output column '{output_column}' contains {invalid_count} non-numeric, "
            "non-finite, or missing value(s)."
        )
        raise ValueError(msg)


# ── categorization ─────────────────────────────────────────────────────────


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
    df_copy[output_column] = pd.to_numeric(df_copy[output_column], errors="raise")
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


# ── axis preparation ───────────────────────────────────────────────────────


def _prepare_axis(ax: Axes, *, title: str | None) -> None:
    """Prepare a clean axis for summary bar rendering."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title is not None:
        ax.set_title(
            title,
            fontsize=_PlotDefaults.title_fontsize,
            pad=_PlotDefaults.Summary.title_pad,
        )


# ── annotation helpers ────────────────────────────────────────────────────


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
        fontsize=_PlotDefaults.Summary.percentage_fontsize,
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
            fontsize=_PlotDefaults.Summary.label_fontsize,
            color=label_color,
        ),
    ]
    return artists


# ── unified summary renderer ──────────────────────────────────────────────


def _plot_summary(
    ax: Axes,
    *,
    vertical: bool,
    region_percentages: pd.Series,
    region_labels: Sequence[str],
    region_colors: Sequence[str],
) -> list[Any]:
    """Render a stacked summary bar (horizontal *or* vertical) with annotations."""
    D = _PlotDefaults.Summary
    artists: list[Any] = []

    if vertical:
        ax.set_xlim(*D.v_xlim)
        ax.set_ylim(*D.v_ylim)
    else:
        ax.set_xlim(*D.h_xlim)
        ax.set_ylim(*D.h_ylim)

    cumulative = 0.0

    for label, color in zip(region_labels, region_colors, strict=False):
        value = float(region_percentages[label])

        if vertical:
            bar = ax.bar(
                x=D.v_bar_x,
                height=value,
                width=D.v_bar_width,
                bottom=cumulative,
                color=color,
                edgecolor=D.bar_edgecolor,
                linewidth=D.bar_linewidth,
            )
        else:
            bar = ax.barh(
                y=D.h_bar_y,
                width=value,
                left=cumulative,
                height=D.h_bar_height,
                color=color,
                edgecolor=D.bar_edgecolor,
                linewidth=D.bar_linewidth,
            )
        artists.append(bar)

        if value > 0:
            if vertical:
                center_y = cumulative + value / 2
                pct_x, pct_y = D.v_bar_x, center_y
                lbl_x, lbl_y = D.v_bar_x + D.v_label_x_offset, center_y
                lbl_ha, lbl_va = "left", "center"
            else:
                pct_x, pct_y = cumulative + value / 2, D.h_bar_y
                lbl_x, lbl_y = cumulative + value / 2, D.h_label_y
                lbl_ha, lbl_va = "center", "bottom"

            artists.extend(
                _add_region_annotations(
                    ax,
                    value=value,
                    label=label,
                    color=color,
                    percentage_x=pct_x,
                    percentage_y=pct_y,
                    label_x=lbl_x,
                    label_y=lbl_y,
                    label_ha=lbl_ha,
                    label_va=lbl_va,
                )
            )

        cumulative += value

    return artists


# ── public API ─────────────────────────────────────────────────────────────


class SummaryPlot:
    """Build and render a threshold summary plot from tabular model outputs.

    The class works with an existing DataFrame that already contains the target
    model output column (e.g., ``pmv`` or ``utci``).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a summary plot builder from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing at least one output column to summarize.

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If ``df`` is empty.
        """
        _validate_dataframe(df)
        self._df = df
        self._region_config: RegionConfig | None = None

    def set_regions(
        self,
        *,
        output: str,
        thresholds: ThresholdsConfig | Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> SummaryPlot:
        """Set output variable and threshold region configuration.

        Accepts either a pre-built :class:`ThresholdsConfig` or raw threshold
        values (with optional *labels* and *colors*).

        Parameters
        ----------
        output : str
            Name of the DataFrame column to categorize.
        thresholds : ThresholdsConfig or sequence of float
            A :class:`ThresholdsConfig` instance **or** a sequence of numeric
            boundary values.  When a ``ThresholdsConfig`` is supplied, *labels*
            and *colors* must not be given separately.
        labels : sequence of str, optional
            Region labels.  Ignored when *thresholds* is a ``ThresholdsConfig``.
            Must have length ``len(thresholds) + 1`` when provided.
        colors : sequence of str, optional
            Region colors.  Ignored when *thresholds* is a ``ThresholdsConfig``.
            Must have length ``len(thresholds) + 1`` when provided.

        Returns
        -------
        SummaryPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``output`` is not a string.
        ValueError
            If the output column is missing or has invalid values, if *labels*
            or *colors* are supplied together with a ``ThresholdsConfig``, or
            if thresholds/labels/colors are invalid.
        """
        output_name = _validate_output_column(self._df, output)
        _validate_output_values(self._df, output_name)

        if isinstance(thresholds, ThresholdsConfig):
            if labels is not None or colors is not None:
                raise ValueError(
                    "labels and colors must not be provided separately when "
                    "thresholds is a ThresholdsConfig instance.  Set them "
                    "inside the ThresholdsConfig instead."
                )
            config = thresholds
        else:
            config = ThresholdsConfig(
                thresholds=thresholds, labels=labels, colors=colors
            )

        self._region_config = _configure_regions(
            output=output_name,
            thresholds=config,
        )
        return self

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
    ) -> SummaryPlotResult:
        """Render a threshold summary plot for the configured output column.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is created
            with a default size of ``(7, 4)`` inches.
        title : str, optional
            Optional axis title.
        vertical : bool
            If ``True``, render a vertical stacked bar; otherwise horizontal.

        Returns
        -------
        SummaryPlotResult
            Result with figure, axis, processed data, and artists.

        Raises
        ------
        ValueError
            If regions are not configured first via :meth:`set_regions`.
        """
        if self._region_config is None:
            raise ValueError(
                "Regions are not set. Call set_regions(...) before plot(...)."
            )
        rc = self._region_config

        if ax is None:
            fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
        else:
            fig = ax.figure

        label_column = f"{rc.output_name}_label"
        df_copy = self._df.copy()
        df_copy, region_percentages = _categorize_output_values(
            df_copy,
            output_column=rc.output_name,
            label_column=label_column,
            levels=rc.thresholds,
            region_labels=rc.labels,
        )

        _prepare_axis(ax, title=title)
        artists = _plot_summary(
            ax,
            vertical=vertical,
            region_percentages=region_percentages,
            region_labels=rc.labels,
            region_colors=rc.colors,
        )

        return SummaryPlotResult(
            fig=fig,
            ax=ax,
            data=df_copy,
            region_percentages=region_percentages,
            artists=artists,
        )
