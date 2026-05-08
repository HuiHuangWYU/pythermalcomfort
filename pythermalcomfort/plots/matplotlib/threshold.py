"""Class-based threshold plotting with a contour backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    BasePlotResult,
    _apply_default_links_to_kwargs,
    _AxisConfig,
    _extract_output_by_name,
    _inspect_model_signature,
    _parse_axis_range,
    _PlotDefaults,
    _validate_model_kwargs,
    _validate_resolution,
)

#: Default color for grid cells that fall outside the model's applicability limits.
OUT_OF_MODEL_LIMITS_COLOR: str = _PlotDefaults.color_out_of_model


@dataclass
class ThresholdPlotResult(BasePlotResult):
    """Container with handles returned by :meth:`ThresholdPlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the rendered threshold plot.
    ax : Axes
        Matplotlib axis containing the rendered threshold plot.
    lines : list of Line2D
        Contour boundary lines as editable artists.
    fills : list of PolyCollection
        Filled threshold regions as artists.
    legend : Legend or None
        Legend artist if ``legend=True``, otherwise ``None``.
    """

    lines: list[Line2D]
    fills: list[PolyCollection]
    legend: Legend | None


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


class ThresholdPlot(BasePlot):
    """Configure and render threshold regions for a selected model function.

    The API is staged and explicit:

    1. configure x and y axes,
    2. set fixed model parameters,
    3. define output thresholds and optional labels/colors,
    4. render with :meth:`plot`.

    The returned result contains editable Matplotlib artists, so users can apply
    additional styling with standard Matplotlib code.

    Examples
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

        Parameters
        ----------
        model_func : callable
            Callable model function (typically from ``pythermalcomfort.models``).
            Its signature is inspected to validate axis names and fixed parameters.
        """
        super().__init__()
        self._model_func = model_func
        self._x_axis: _AxisConfig | None = None
        self._y_axis: _AxisConfig | None = None
        self._fixed_values: dict[str, Any] = {}
        self._default_links = _PlotDefaults.parameter_links
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

        other = self._y_axis if axis_kind == "x" else self._x_axis
        if other is not None and axis_name == other.name:
            raise ValueError("x and y axis parameters must be different.")

        if axis_name in self._fixed_values:
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

        axis_attr = "_x_axis" if axis_kind == "x" else "_y_axis"
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

        Parameters
        ----------
        name : str
            Model argument name mapped to the x-axis.
        min_val : float
            Minimum x-axis value.
        max_val : float
            Maximum x-axis value.
        resolution : float
            Grid step along x-axis used for contour evaluation.

        Returns
        -------
        ThresholdPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``name`` is not a string.
        ValueError
            If ``name`` is empty/invalid, conflicts with y-axis,
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

        Parameters
        ----------
        name : str
            Model argument name mapped to the y-axis.
        min_val : float
            Minimum y-axis value.
        max_val : float
            Maximum y-axis value.
        resolution : float
            Grid step along y-axis used for contour evaluation.

        Returns
        -------
        ThresholdPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``name`` is not a string.
        ValueError
            If ``name`` is empty/invalid, conflicts with x-axis,
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

        Parameters
        ----------
        **kwargs : Any
            Fixed model inputs passed unchanged to model evaluations.

        Returns
        -------
        ThresholdPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If parameter names are invalid for the model signature or conflict
            with configured x/y axis parameters.

        Notes
        -----
        **tr / tdb auto-link** — When the model accepts both ``tdb``
        (dry-bulb temperature) and ``tr`` (mean radiant temperature), and one
        of them is used as an axis parameter, the other is automatically set to
        the same per-cell value if it is not explicitly supplied here.  For
        example, if ``tdb`` is the x-axis and ``tr`` is omitted, every model
        call receives ``tr = tdb``.  Supply ``tr=<value>`` explicitly to
        override this behaviour.
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
            axis.name for axis in (self._x_axis, self._y_axis) if axis is not None
        }
        conflicting_axis_keys = sorted(key for key in kwargs if key in axis_names)
        if conflicting_axis_keys:
            conflict_key = conflicting_axis_keys[0]
            msg = (
                f"set_params() cannot set axis parameter '{conflict_key}'. "
                "Set it through set_x_axis(...) or set_y_axis(...)."
            )
            raise ValueError(msg)

        self._fixed_values.update(kwargs)

        return self

    def _validate_plot_inputs(self, *, fill_kws: Mapping[str, Any] | None) -> None:
        """Validate plot preconditions and plotting keyword constraints."""
        if self._x_axis is None or self._y_axis is None:
            raise ValueError(
                "Axes are not set. Call set_x_axis(...) and set_y_axis(...) first."
            )
        if self._region_config is None:
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
        call_kwargs: dict[str, Any] = dict(self._fixed_values)
        call_kwargs[self._x_axis.name] = x_value
        call_kwargs[self._y_axis.name] = y_value
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
        x_min = float(self._x_axis.min_val)
        x_max = float(self._x_axis.max_val)
        y_min = float(self._y_axis.min_val)
        y_max = float(self._y_axis.max_val)

        x_vals = np.arange(x_min, x_max, self._x_axis.resolution)
        if x_vals[-1] < x_max:
            x_vals = np.append(x_vals, x_max)

        y_vals = np.arange(y_min, y_max, self._y_axis.resolution)
        if y_vals[-1] < y_max:
            y_vals = np.append(y_vals, y_max)

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
            result = self._model_func(**grid_kwargs)
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
        invalid_color: str = _PlotDefaults.color_out_of_model,
    ) -> ThresholdPlotResult:
        """Render threshold regions and contours on a Matplotlib axis.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is
            created with a default size of ``(7, 4)`` inches.
        title : str, optional
            Optional axis title.
        legend : bool
            Whether to draw a legend.
        show_lines : bool
            Whether to draw threshold contour boundaries.
        line_kws : dict, optional
            Keyword overrides forwarded to ``ax.plot`` for contour lines.
        fill_kws : dict, optional
            Keyword overrides forwarded to ``ax.contourf`` for region fills.
            Keys ``color`` and ``facecolor`` are reserved and rejected.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.
        invalid_color : str
            Color used for out-of-model/invalid grid areas.

        Returns
        -------
        ThresholdPlotResult
            Result with axis and artist handles.

        Raises
        ------
        ValueError
            If required configuration is missing, plotting inputs are invalid,
            or model evaluation/output extraction fails.
        """
        self._validate_plot_inputs(fill_kws=fill_kws)
        self._validate_invalid_color(invalid_color)

        rc = self._region_config

        line_opts = dict(line_kws or {})
        line_opts.setdefault("color", _PlotDefaults.Threshold.line_color)
        line_opts.setdefault("linewidth", _PlotDefaults.Threshold.line_linewidth)

        fill_opts = dict(fill_kws or {})
        fill_opts.setdefault("corner_mask", _PlotDefaults.Threshold.fill_corner_mask)

        legend_opts = dict(legend_kws or {})
        legend_opts.setdefault("loc", _PlotDefaults.Threshold.legend_loc)
        legend_opts.setdefault(
            "bbox_to_anchor", _PlotDefaults.Threshold.legend_bbox_to_anchor
        )
        legend_opts.setdefault("frameon", _PlotDefaults.Threshold.legend_frameon)

        if ax is None:
            fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
        else:
            fig = ax.figure

        x_min, x_max, y_min, y_max, x, y = self._build_grid()
        z = self._evaluate_grid_output(x=x, y=y, output_name=rc.output_name)

        finite = np.isfinite(z)
        invalid = ~finite
        z_masked = np.ma.masked_invalid(z)

        extended_levels = [-np.inf, *rc.thresholds, np.inf]
        fills: list[PolyCollection] = []
        if finite.any():
            filled_contours = ax.contourf(
                x,
                y,
                z_masked,
                levels=extended_levels,
                colors=rc.colors,
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
                zorder=_PlotDefaults.Threshold.zorder_invalid,
            )

        lines: list[Line2D] = []
        if show_lines and finite.any():
            contour_lines = ax.contour(
                x, y, z_masked, levels=rc.thresholds, antialiased=True
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
                    facecolor=color,
                    alpha=fill_opts.get("alpha", _PlotDefaults.fill_alpha),
                    label=label,
                )
                for label, color in zip(rc.labels, rc.colors, strict=False)
            ]
            if invalid.any():
                handles.append(
                    Patch(
                        facecolor=invalid_color,
                        alpha=1.0,
                        label="Out of model limits",
                    )
                )
            legend_opts.setdefault(
                "ncol", min(len(handles), _PlotDefaults.Threshold.legend_ncol_max)
            )
            legend_artist = ax.legend(
                handles=handles,
                **legend_opts,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(self._x_axis.name)
        ax.set_ylabel(self._y_axis.name)
        if title is not None:
            ax.set_title(title)

        return ThresholdPlotResult(
            fig=fig,
            ax=ax,
            lines=lines,
            fills=fills,
            legend=legend_artist,
        )
