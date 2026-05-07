"""Class-based psychrometric charting with contour threshold regions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from pythermalcomfort.plots.matplotlib._shared import (
    _apply_default_links_to_kwargs,
    _AxisConfig,
    _extract_output_by_name,
    _parse_axis_range,
    _validate_model_kwargs,
    _validate_resolution,
)
from pythermalcomfort.plots.matplotlib.threshold import (
    OUT_OF_MODEL_LIMITS_COLOR,
    ThresholdPlot,
    ThresholdPlotResult,
)
from pythermalcomfort.utilities import p_sat, psy_ta_rh


class PsychrometricPlot(ThresholdPlot):
    """Configure and render a psychrometric chart with threshold regions.

    Inherits from :class:`ThresholdPlot` but strictly enforces ``tdb``
    (dry-bulb temperature) on the x-axis and ``hr`` (humidity ratio) on the
    y-axis.  Grid evaluation converts humidity ratio back to relative humidity
    before calling the underlying model.  Constant-RH background curves are
    drawn on top of the threshold regions.

    Example
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.plots.matplotlib import PsychrometricPlot

        result = (
            PsychrometricPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 10.0, 36.0, resolution=0.2)
            .set_y_axis("hr", 0.0, 0.030, resolution=0.0005)
            .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
            .plot(title="PMV — Psychrometric Chart")
        )
    """

    def set_x_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set x-axis; must be ``'tdb'`` (dry-bulb temperature).

        Args:
            name: Must be ``'tdb'``.
            min_val: Minimum dry-bulb temperature.
            max_val: Maximum dry-bulb temperature.
            resolution: Grid step along the x-axis.

        Returns:
            Self, to support method chaining.

        Raises:
            ValueError: If ``name`` is not ``'tdb'``, or if range/resolution
                are invalid.
        """
        if name != "tdb":
            raise ValueError(
                "PsychrometricPlot requires the x-axis to be 'tdb' (dry-bulb temperature)."
            )
        return super().set_x_axis(name, min_val, max_val, resolution=resolution)

    def set_y_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set y-axis; must be ``'hr'`` (humidity ratio).

        The standard model-argument check is bypassed because thermal comfort
        models accept ``rh`` (relative humidity), not ``hr`` directly.  Grid
        evaluation handles the conversion internally.

        Args:
            name: Must be ``'hr'``.
            min_val: Minimum humidity ratio (kg/kg).
            max_val: Maximum humidity ratio (kg/kg).
            resolution: Grid step along the y-axis.

        Returns:
            Self, to support method chaining.

        Raises:
            ValueError: If ``name`` is not ``'hr'``, conflicts with a fixed
                parameter set via :meth:`set_params`, or if range/resolution
                are invalid.
        """
        if name != "hr":
            raise ValueError(
                "PsychrometricPlot requires the y-axis to be 'hr' (humidity ratio)."
            )
        if name in self._fixed_values:
            msg = (
                f"set_params() already contains axis parameter '{name}'. "
                "Remove it before calling set_y_axis()."
            )
            raise ValueError(msg)
        if self._x_axis is not None and name == self._x_axis.name:
            msg = "x and y axis parameters must be different. "
            raise ValueError(msg)

        min_float, max_float = _parse_axis_range(min_val, max_val)
        resolution_float = _validate_resolution(resolution)
        self._y_axis = _AxisConfig(
            name=name,
            min_val=min_float,
            max_val=max_float,
            resolution=resolution_float,
        )
        return self

    def _evaluate_grid_output(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        output_name: str,
    ) -> np.ndarray:
        """Evaluate the model on the psychrometric grid and return shaped output.

        Converts the humidity-ratio grid (*y*) to relative humidity before
        calling the model.  Cells where RH > 100 % or RH < 0 % are physically
        impossible and are returned as NaN so the parent renders them as
        out-of-model areas.
        """
        x_flat = np.asarray(x).ravel()  # tdb
        y_flat = np.asarray(y).ravel()  # hr (kg/kg)

        # Reverse-calculate RH from humidity ratio and dry-bulb temperature.
        p_atm = 101325.0
        p_vap = (y_flat * p_atm) / (0.62198 + y_flat)
        p_sat_values = p_sat(x_flat)
        rh_flat = (p_vap / p_sat_values) * 100.0

        invalid_mask = (rh_flat < 0.0) | (rh_flat > 100.0)
        # Clamp to a safe value so the model does not receive out-of-range RH.
        rh_safe = np.where(invalid_mask, 50.0, rh_flat)

        grid_kwargs: dict[str, Any] = dict(self._fixed_values)
        grid_kwargs["tdb"] = x_flat
        grid_kwargs["rh"] = rh_safe
        grid_kwargs = _apply_default_links_to_kwargs(
            grid_kwargs,
            allowed_args=self._allowed_args,
            default_links=self._default_links,
        )
        _validate_model_kwargs(
            grid_kwargs,
            allowed_args=self._allowed_args,
            required_args=self._required_args,
            accepts_var_kwargs=self._accepts_var_kwargs,
        )

        try:
            result = self._model_func(**grid_kwargs)
        except Exception as exc:
            msg = f"Failed to evaluate model on psychrometric grid: {exc}"
            raise ValueError(msg) from exc

        try:
            payload = _extract_output_by_name(result, output_name)
        except Exception as exc:
            msg = f"Failed to extract output '{output_name}' from psychrometric result: {exc}"
            raise ValueError(msg) from exc

        z_flat = np.asarray(payload, dtype=float)
        if z_flat.size != x.size:
            msg = (
                "Model output shape does not match the contour grid. "
                f"Expected {x.size} values for the flattened grid, got {z_flat.size}."
            )
            raise ValueError(msg)

        z_flat[invalid_mask] = np.nan
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
        """Render the psychrometric chart with threshold regions and RH curves.

        Delegates to :meth:`ThresholdPlot.plot` for contour rendering, then
        overlays:

        - A white fill masking the physically impossible RH > 100 % area.
        - Dotted constant-RH background curves at 10 % intervals.

        Args:
            ax: Existing axis to draw on. If ``None``, a new figure/axis is
                created.
            title: Optional axis title.
            legend: Whether to draw a legend.
            show_lines: Whether to draw threshold contour boundaries.
            line_kws: Keyword overrides forwarded to ``ax.plot`` for contour
                lines.
            fill_kws: Keyword overrides forwarded to ``ax.contourf`` for region
                fills.  Keys ``color`` and ``facecolor`` are reserved and
                rejected.
            legend_kws: Keyword overrides forwarded to ``ax.legend``.
            invalid_color: Color used for out-of-model/invalid grid areas.

        Returns:
            :class:`ThresholdPlotResult` with axis and artist handles.
        """
        result = super().plot(
            ax=ax,
            title=title,
            legend=legend,
            show_lines=show_lines,
            line_kws=line_kws,
            fill_kws=fill_kws,
            legend_kws=legend_kws,
            invalid_color=invalid_color,
        )
        ax = result.ax

        tdb_dense = np.linspace(self._x_axis.min_val, self._x_axis.max_val, 500)
        label_offset = (self._y_axis.max_val - self._y_axis.min_val) * 0.01

        # White fill hides the grey out-of-model patch above the RH = 100% curve.
        hr_100 = psy_ta_rh(tdb_dense, np.full_like(tdb_dense, 100.0)).hr
        ax.fill_between(
            tdb_dense,
            hr_100,
            self._y_axis.max_val,
            color="white",
            zorder=1.6,
            edgecolor="none",
        )

        for rh_target in range(10, 110, 10):
            hr_line = psy_ta_rh(tdb_dense, np.full_like(tdb_dense, float(rh_target))).hr
            in_range = hr_line <= self._y_axis.max_val
            if not in_range.any():
                continue
            ax.plot(
                tdb_dense[in_range],
                hr_line[in_range],
                color="#a0a0a0",
                linestyle=":",
                linewidth=0.8,
                zorder=2.0,
            )
            last_idx = int(np.where(in_range)[0][-1])
            ax.text(
                tdb_dense[last_idx],
                hr_line[last_idx] + label_offset,
                f"{rh_target}%",
                color="#a0a0a0",
                fontsize=8,
                zorder=2.0,
            )

        ax.set_xlim(self._x_axis.min_val, self._x_axis.max_val)
        ax.set_ylim(self._y_axis.min_val, self._y_axis.max_val)

        return result
