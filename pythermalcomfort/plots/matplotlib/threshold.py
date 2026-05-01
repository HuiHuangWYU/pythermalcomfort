"""Class-based threshold plotting with a contour backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, is_color_like
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._shared import (
    _apply_default_links_to_kwargs,
    _build_region_labels,
    _extract_output_by_name,
    _inspect_model_signature,
    _normalize_levels,
    _resolve_region_colors,
    _validate_model_kwargs,
)

OUT_OF_MODEL_LIMITS_COLOR = "#bdbdbd"


@dataclass
class _AxisConfig:
    """Axis plotting configuration."""

    name: str
    min_val: float
    max_val: float
    resolution: float


@dataclass
class ThresholdPlotResult:
    """Container with handles returned by :meth:`ThresholdPlot.plot`.

    Attributes:
        fig: Matplotlib figure containing the rendered threshold plot.
        ax: Matplotlib axis containing the rendered threshold plot.
        lines: Contour boundary lines as editable ``Line2D`` artists.
        fills: Filled threshold regions as ``PolyCollection`` artists.
        legend: Legend artist if ``legend=True``, otherwise ``None``.
    """

    fig: Figure
    ax: Axes
    lines: list[Line2D]
    fills: list[PolyCollection]
    legend: Legend | None


def _parse_axis_range(min_val: Any, max_val: Any) -> tuple[float, float]:
    """Validate and normalize axis bounds."""
    try:
        min_float = float(min_val)
        max_float = float(max_val)
    except (TypeError, ValueError) as exc:
        raise ValueError("Axis range values must be numeric.") from exc

    if min_float >= max_float:
        msg = f"Axis requires min < max (got {min_float} >= {max_float})."
        raise ValueError(msg)

    return min_float, max_float


def _validate_resolution(resolution: Any) -> float:
    """Validate axis resolution."""
    try:
        resolution_float = float(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError("Axis resolution must be numeric.") from exc
    if resolution_float <= 0:
        raise ValueError("Axis resolution must be positive.")
    return resolution_float


def _contour_paths_to_lines(
    ax: Axes,
    *,
    contour_set: Any,
    line_opts: Mapping[str, Any],
) -> list[Line2D]:
    """Convert contour paths into editable line artists."""
    lines: list[Line2D] = []

    for path in contour_set.get_paths():
        segments = path.to_polygons(closed_only=False)
        x_combined: list[float] = []
        y_combined: list[float] = []

        for segment in segments:
            if len(segment) == 0:
                continue
            x_combined.extend(segment[:, 0].tolist())
            x_combined.append(np.nan)
            y_combined.extend(segment[:, 1].tolist())
            y_combined.append(np.nan)

        if x_combined:
            (line,) = ax.plot(x_combined, y_combined, **line_opts)
            lines.append(line)

    contour_set.remove()
    return lines


class ThresholdPlot:
    """Configure and render threshold regions for a selected model function.

    The API is staged and explicit:

    1. configure x and y axes,
    2. set fixed model parameters,
    3. define output thresholds and optional labels/colors,
    4. render with :meth:`plot`.

    The returned result contains editable Matplotlib artists, so users can apply
    additional styling with standard Matplotlib code.

    Example
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.plots.matplotlib import ThresholdPlot

        result = (
            ThresholdPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 18.0, 34.0, resolution=0.2)
            .set_y_axis("rh", 20.0, 100.0, resolution=0.5)
            .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
            .plot(title="PMV Threshold Regions")
        )
        result.ax.set_xlabel("Air temperature [°C]")
    """

    def __init__(self, model_func: Any) -> None:
        """Initialize a threshold plot builder.

        Args:
            model_func: Callable model function (typically from
                ``pythermalcomfort.models``). Its signature is inspected to validate
                axis names and fixed parameters.
        """
        self.model_func = model_func
        self.x_axis: _AxisConfig | None = None
        self.y_axis: _AxisConfig | None = None
        self.fixed_values: dict[str, Any] = {}
        self._output_name: str | None = None
        self._thresholds: list[float] | None = None
        self._region_labels: list[str] | None = None
        self._region_colors: list[str] | None = None
        self._default_links: dict[str, str] = {"tr": "tdb", "tdb": "tr"}
        (
            _,
            self._allowed_args,
            self._required_args,
            self._accepts_var_kwargs,
        ) = _inspect_model_signature(model_func)

    def _set_axis(
        self,
        *,
        axis_kind: str,
        name: str,
        min_val: Any,
        max_val: Any,
        resolution: Any,
    ) -> ThresholdPlot:
        """Validate and set one axis configuration."""
        if not isinstance(name, str):
            raise TypeError("Axis name must be a string.")
        axis_name = name.strip()
        if not axis_name:
            raise ValueError("Axis name must be a non-empty string.")

        if axis_name not in self._allowed_args:
            msg = (
                f"{axis_kind} axis parameter '{axis_name}' was not found in the model "
                "function arguments."
            )
            raise ValueError(msg)

        other = self.y_axis if axis_kind == "x" else self.x_axis
        if other is not None and axis_name == other.name:
            raise ValueError("x and y axis parameters must be different.")

        if axis_name in self.fixed_values:
            msg = (
                f"set_params() already contains axis parameter '{axis_name}'. "
                f"Remove it before calling set_{axis_kind}_axis()."
            )
            raise ValueError(msg)

        min_float, max_float = _parse_axis_range(min_val, max_val)
        resolution_float = _validate_resolution(resolution)
        axis_config = _AxisConfig(
            name=axis_name,
            min_val=min_float,
            max_val=max_float,
            resolution=resolution_float,
        )

        axis_attr = "x_axis" if axis_kind == "x" else "y_axis"
        setattr(self, axis_attr, axis_config)
        return self

    def set_x_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> ThresholdPlot:
        """Set x-axis model parameter, range, and grid resolution.

        Args:
            name: Model argument name mapped to the x-axis.
            min_val: Minimum x-axis value.
            max_val: Maximum x-axis value.
            resolution: Grid step along x-axis used for contour evaluation.

        Returns:
            Self, to support method chaining.

        Raises:
            TypeError: If ``name`` is not a string.
            ValueError: If ``name`` is empty/invalid, conflicts with y-axis,
                conflicts with fixed params, or range/resolution are invalid.
        """
        return self._set_axis(
            axis_kind="x",
            name=name,
            min_val=min_val,
            max_val=max_val,
            resolution=resolution,
        )

    def set_y_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> ThresholdPlot:
        """Set y-axis model parameter, range, and grid resolution.

        Args:
            name: Model argument name mapped to the y-axis.
            min_val: Minimum y-axis value.
            max_val: Maximum y-axis value.
            resolution: Grid step along y-axis used for contour evaluation.

        Returns:
            Self, to support method chaining.

        Raises:
            TypeError: If ``name`` is not a string.
            ValueError: If ``name`` is empty/invalid, conflicts with x-axis,
                conflicts with fixed params, or range/resolution are invalid.
        """
        return self._set_axis(
            axis_kind="y",
            name=name,
            min_val=min_val,
            max_val=max_val,
            resolution=resolution,
        )

    def set_params(self, **kwargs: Any) -> ThresholdPlot:
        """Set fixed model parameters used during grid evaluation.

        Args:
            **kwargs: Fixed model inputs passed unchanged to model evaluations.

        Returns:
            Self, to support method chaining.

        Raises:
            ValueError: If parameter names are invalid for the model signature or
                conflict with configured x/y axis parameters.

        Note:
            **tr / tdb auto-link** — When the model accepts both ``tdb``
            (dry-bulb temperature) and ``tr`` (mean radiant temperature), and
            one of them is used as an axis parameter, the other is automatically
            set to the same per-cell value if it is not explicitly supplied here.
            For example, if ``tdb`` is the x-axis and ``tr`` is omitted, every
            model call receives ``tr = tdb``.  Supply ``tr=<value>`` explicitly
            to override this behaviour.
        """
        if not self._accepts_var_kwargs:
            invalid = sorted(key for key in kwargs if key not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = (
                    f"set_params() parameter(s) {invalid_str} were not found in the "
                    "model function arguments."
                )
                raise ValueError(msg)

        axis_names = {
            axis.name for axis in (self.x_axis, self.y_axis) if axis is not None
        }
        conflicting_axis_keys = sorted(key for key in kwargs if key in axis_names)
        if conflicting_axis_keys:
            conflict_key = conflicting_axis_keys[0]
            msg = (
                f"set_params() cannot set axis parameter '{conflict_key}'. "
                "Set it through set_x_axis(...) or set_y_axis(...)."
            )
            raise ValueError(msg)

        self.fixed_values.update(kwargs)

        return self

    def set_regions(
        self,
        *,
        output: str,
        thresholds: Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> ThresholdPlot:
        """Set output variable and threshold region configuration.

        Args:
            output: Output field name to extract from the model result.
            thresholds: Threshold boundary values for region splitting.
            labels: Optional region labels. Must have length
                ``len(thresholds) + 1`` when provided.
            colors: Optional region colors. Must have length
                ``len(thresholds) + 1`` when provided.

        Returns:
            Self, to support method chaining.

        Raises:
            TypeError: If ``output`` is not a string.
            ValueError: If output name is empty, thresholds/labels/colors are invalid.
        """
        if not isinstance(output, str):
            raise TypeError("output must be a string.")
        output_name = output.strip()
        if not output_name:
            raise ValueError("output must be a non-empty string.")

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

    def _validate_plot_inputs(self, *, fill_kws: Mapping[str, Any] | None) -> None:
        """Validate plot preconditions and plotting keyword constraints."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError(
                "Axes are not set. Call set_x_axis(...) and set_y_axis(...) first."
            )
        if (
            self._output_name is None
            or self._thresholds is None
            or self._region_labels is None
            or self._region_colors is None
        ):
            raise ValueError(
                "Regions are not set. Call set_regions(...) before plot(...)."
            )

        disallowed_fill_keys = {"color", "facecolor"}
        if fill_kws is not None and disallowed_fill_keys.intersection(fill_kws):
            raise ValueError(
                "fill_kws cannot include 'color' or 'facecolor'. "
                "Use set_regions(..., colors=...) to control region colors."
            )

    def _validate_invalid_color(self, invalid_color: str) -> None:
        """Validate color used to render out-of-model areas."""
        if not isinstance(invalid_color, str) or not is_color_like(invalid_color):
            raise ValueError("invalid_color must be a valid Matplotlib color string.")

    def _build_call_kwargs(self, x_value: Any, y_value: Any) -> dict[str, Any]:
        """Build validated kwargs for a model evaluation call."""
        call_kwargs: dict[str, Any] = dict(self.fixed_values)
        call_kwargs[self.x_axis.name] = x_value
        call_kwargs[self.y_axis.name] = y_value
        call_kwargs = _apply_default_links_to_kwargs(
            call_kwargs,
            allowed_args=self._allowed_args,
            default_links=self._default_links,
        )
        _validate_model_kwargs(
            call_kwargs,
            allowed_args=self._allowed_args,
            required_args=self._required_args,
            accepts_var_kwargs=self._accepts_var_kwargs,
        )
        return call_kwargs

    def _build_grid(
        self,
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Build contour mesh grid from axis configs."""
        x_min = float(self.x_axis.min_val)
        x_max = float(self.x_axis.max_val)
        y_min = float(self.y_axis.min_val)
        y_max = float(self.y_axis.max_val)

        x_vals = np.arange(x_min, x_max + 1e-12, self.x_axis.resolution)
        y_vals = np.arange(y_min, y_max + 1e-12, self.y_axis.resolution)
        if x_vals.size < 2 or y_vals.size < 2:
            msg = (
                "Axis resolution is too coarse for the chosen ranges. "
                "Each axis requires at least 2 grid points."
            )
            raise ValueError(msg)

        X, Y = np.meshgrid(x_vals, y_vals)
        return x_min, x_max, y_min, y_max, X, Y

    def _evaluate_grid_output(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        output_name: str,
    ) -> np.ndarray:
        """Evaluate the model on the contour grid and return shaped output."""
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()
        grid_kwargs = self._build_call_kwargs(x_flat, y_flat)
        try:
            result = self.model_func(**grid_kwargs)
        except Exception as exc:
            msg = f"Failed to evaluate model on contour grid: {exc}"
            raise ValueError(msg) from exc

        try:
            payload = _extract_output_by_name(result, output_name)
        except Exception as exc:
            msg = f"Failed to extract output '{output_name}' from contour result: {exc}"
            raise ValueError(msg) from exc

        z_flat = np.asarray(payload, dtype=float)
        if z_flat.size != x.size:
            msg = (
                "Model output shape does not match the contour grid. "
                f"Expected {x.size} values for the flattened grid, got {z_flat.size}."
            )
            raise ValueError(msg)
        return z_flat.reshape(x.shape)

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
        invalid_color: str = OUT_OF_MODEL_LIMITS_COLOR,
    ) -> ThresholdPlotResult:
        """Render threshold regions and contours on a Matplotlib axis.

        Args:
            ax: Existing axis to draw on. If ``None``, a new figure/axis is created.
            title: Optional axis title.
            legend: Whether to draw a legend.
            show_lines: Whether to draw threshold contour boundaries.
            line_kws: Keyword overrides forwarded to ``ax.plot`` for contour lines.
            fill_kws: Keyword overrides forwarded to ``ax.contourf`` for region fills.
                Keys ``color`` and ``facecolor`` are reserved and rejected.
            legend_kws: Keyword overrides forwarded to ``ax.legend``.
            invalid_color: Color used for out-of-model/invalid grid areas.

        Returns:
            :class:`ThresholdPlotResult` with axis and artist handles.

        Raises:
            ValueError: If required configuration is missing, plotting inputs are
                invalid, or model evaluation/output extraction fails.
        """
        self._validate_plot_inputs(fill_kws=fill_kws)
        self._validate_invalid_color(invalid_color)

        output_name = self._output_name
        normalized_levels = self._thresholds
        region_labels = self._region_labels
        region_colors = self._region_colors

        line_opts = dict(line_kws or {})
        line_opts.setdefault("color", "black")
        line_opts.setdefault("linewidth", 1.0)

        fill_opts = dict(fill_kws or {})
        fill_opts.setdefault("alpha", 1)
        fill_opts.setdefault("corner_mask", False)

        legend_opts = dict(legend_kws or {})
        legend_opts.setdefault("loc", "lower center")
        legend_opts.setdefault("bbox_to_anchor", (0.5, 1.02))
        legend_opts.setdefault("frameon", False)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            fig = ax.figure

        x_min, x_max, y_min, y_max, x, y = self._build_grid()
        z = self._evaluate_grid_output(x=x, y=y, output_name=output_name)

        finite = np.isfinite(z)
        invalid = ~finite
        z_masked = np.ma.masked_invalid(z)

        extended_levels = [-np.inf, *normalized_levels, np.inf]
        fills: list[PolyCollection] = []
        if finite.any():
            filled_contours = ax.contourf(
                x,
                y,
                z_masked,
                levels=extended_levels,
                colors=region_colors,
                extend="neither",
                **fill_opts,
            )

            if hasattr(filled_contours, "collections"):
                fills.extend(
                    cast(list[PolyCollection], list(filled_contours.collections))
                )
            else:
                fills.append(cast(PolyCollection, filled_contours))

        if invalid.any():
            invalid_cells = (
                invalid[:-1, :-1]
                | invalid[1:, :-1]
                | invalid[:-1, 1:]
                | invalid[1:, 1:]
            )
            invalid_mask = np.ma.masked_where(
                ~invalid_cells,
                np.ones_like(invalid_cells, dtype=float),
            )
            ax.pcolormesh(
                x,
                y,
                invalid_mask,
                cmap=ListedColormap([invalid_color]),
                shading="flat",
                zorder=1.5,
            )

        lines: list[Line2D] = []
        if show_lines and finite.any():
            contour_lines = ax.contour(
                x, y, z_masked, levels=normalized_levels, antialiased=True
            )
            lines = _contour_paths_to_lines(
                ax,
                contour_set=contour_lines,
                line_opts=line_opts,
            )

        legend_artist: Legend | None = None
        if legend:
            handles = [
                Patch(
                    facecolor=region_colors[index],
                    alpha=fill_opts.get("alpha"),
                    label=region_labels[index],
                )
                for index in range(len(region_labels))
            ]
            if invalid.any():
                handles.append(
                    Patch(
                        facecolor=invalid_color,
                        alpha=1.0,
                        label="Out of model limits",
                    )
                )
            legend_opts.setdefault("ncol", min(len(handles), 4))
            legend_artist = ax.legend(
                handles=handles,
                **legend_opts,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(self.x_axis.name)
        ax.set_ylabel(self.y_axis.name)
        if title is not None:
            ax.set_title(title)

        return ThresholdPlotResult(
            fig=fig,
            ax=ax,
            lines=lines,
            fills=fills,
            legend=legend_artist,
        )
