"""Review-only class-based threshold plotting prototype with contour backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots._shared import (
    ThresholdsConfig,
    _apply_default_links_to_kwargs,
    _build_region_labels,
    _extract_output_payload,
    _inspect_model_signature,
    _normalize_levels,
    _resolve_region_colors,
    _validate_model_kwargs,
)


@dataclass
class _AxisConfig:
    name: str
    min_val: float
    max_val: float


@dataclass
class ThresholdPlotResult:
    ax: Axes
    lines: list[Line2D]
    fills: list[PolyCollection]
    legend: Legend | None


def _parse_axis_range(param_name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, tuple | list) or len(value) != 2:
        msg = f"Axis '{param_name}' must be a tuple/list of length 2: (min, max)."
        raise ValueError(msg)

    try:
        min_val = float(value[0])
        max_val = float(value[1])
    except (TypeError, ValueError) as exc:
        msg = f"Axis '{param_name}' range values must be numeric."
        raise ValueError(msg) from exc

    if min_val >= max_val:
        msg = f"Axis '{param_name}' requires min < max (got {min_val} >= {max_val})."
        raise ValueError(msg)

    return min_val, max_val


def _contour_paths_to_lines(
    ax: Axes,
    *,
    contour_set: Any,
    line_opts: Mapping[str, Any],
) -> list[Line2D]:
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


class Threshold:
    def __init__(self, model_func: Any) -> None:
        self.model_func = model_func
        self.x_axis: _AxisConfig | None = None
        self.y_axis: _AxisConfig | None = None
        self.fixed_values: dict[str, Any] = {}
        self._default_links: dict[str, str] = {"tr": "tdb", "tdb": "tr"}
        (
            _,
            self._allowed_args,
            self._required_args,
            self._accepts_var_kwargs,
        ) = _inspect_model_signature(model_func)

    def _extract_axis_spec(
        self,
        *,
        axis_label: str,
        axis_range: dict[str, Any],
        other_axis: _AxisConfig | None = None,
    ) -> _AxisConfig:
        if len(axis_range) != 1:
            msg = f"{axis_label}() requires exactly one keyword range."
            raise ValueError(msg)

        ((axis_name, value),) = axis_range.items()

        if other_axis is not None and axis_name == other_axis.name:
            raise ValueError("x and y axis parameters must be different.")

        if axis_name not in self._allowed_args:
            msg = (
                f"{axis_label}() received invalid parameter: {axis_name}. "
                "Use model argument names from the selected function."
            )
            raise ValueError(msg)

        if axis_name in self.fixed_values:
            msg = (
                f"parameters() already contains axis parameter '{axis_name}'. "
                f"Remove it from parameters() when using {axis_label}()."
            )
            raise ValueError(msg)

        min_val, max_val = _parse_axis_range(axis_name, value)
        return _AxisConfig(name=axis_name, min_val=min_val, max_val=max_val)

    def x(self, **axis_range: Any) -> Threshold:
        self.x_axis = self._extract_axis_spec(
            axis_label="x",
            axis_range=axis_range,
            other_axis=self.y_axis,
        )
        return self

    def y(self, **axis_range: Any) -> Threshold:
        self.y_axis = self._extract_axis_spec(
            axis_label="y",
            axis_range=axis_range,
            other_axis=self.x_axis,
        )
        return self

    def parameters(self, **kwargs: Any) -> Threshold:
        if not self._accepts_var_kwargs:
            invalid = sorted(key for key in kwargs if key not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = (
                    f"parameters() received invalid parameter(s): {invalid_str}. "
                    "Use allowed_parameters() to inspect valid keys."
                )
                raise ValueError(msg)

        x_name = self.x_axis.name if self.x_axis is not None else None
        y_name = self.y_axis.name if self.y_axis is not None else None
        for key, value in kwargs.items():
            if key in (x_name, y_name):
                msg = (
                    f"parameters() cannot set axis parameter '{key}'. "
                    "Set it through x(...) or y(...)."
                )
                raise ValueError(msg)
            self.fixed_values[key] = value

        return self

    def allowed_parameters(self) -> list[str]:
        return sorted(self._allowed_args)

    def required_parameters(self) -> list[str]:
        return sorted(self._required_args)

    def _build_call_kwargs(self, x_value: Any, y_value: Any) -> dict[str, Any]:
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call x(...) and y(...) first.")

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

    def plot(
        self,
        *,
        output: str,
        thresholds: ThresholdsConfig,
        x_resolution: float,
        y_resolution: float,
        ax: Axes | None = None,
        title: str | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
    ) -> ThresholdPlotResult:
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call x(...) and y(...) first.")
        if x_resolution <= 0 or y_resolution <= 0:
            raise ValueError("x_resolution and y_resolution must be positive.")

        output_name = output.strip()
        if not output_name:
            raise ValueError("output must be a non-empty string.")
        if not isinstance(thresholds, ThresholdsConfig):
            raise TypeError("thresholds must be a ThresholdsConfig.")

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

        line_opts = dict(line_kws or {})
        line_opts.setdefault("color", "black")
        line_opts.setdefault("linewidth", 1.0)

        fill_opts = dict(fill_kws or {})
        if "color" in fill_opts or "facecolor" in fill_opts:
            raise ValueError(
                "fill_kws cannot include 'color' or 'facecolor'. "
                "Use ThresholdsConfig.colors to control region colors."
            )
        fill_opts.setdefault("alpha", 0.65)
        fill_opts.setdefault("corner_mask", False)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        x_min = float(self.x_axis.min_val)
        x_max = float(self.x_axis.max_val)
        y_min = float(self.y_axis.min_val)
        y_max = float(self.y_axis.max_val)

        x_vals = np.arange(x_min, x_max + 1e-12, x_resolution)
        y_vals = np.arange(y_min, y_max + 1e-12, y_resolution)
        X, Y = np.meshgrid(x_vals, y_vals)

        grid_kwargs = self._build_call_kwargs(X.ravel().tolist(), Y.ravel().tolist())
        try:
            result = self.model_func(**grid_kwargs)
        except Exception as exc:
            msg = f"Failed to evaluate model on contour grid: {exc}"
            raise ValueError(msg) from exc

        try:
            payload = _extract_output_payload(result, output_name)
        except Exception as exc:
            msg = f"Failed to extract output '{output_name}' from contour result: {exc}"
            raise ValueError(msg) from exc

        z_flat = np.asarray(payload, dtype=float)
        if z_flat.size != X.size:
            msg = (
                "Model output shape does not match the contour grid. "
                f"Expected {X.size} values for the flattened grid, got {z_flat.size}."
            )
            raise ValueError(msg)
        Z = z_flat.reshape(X.shape)
        finite = np.isfinite(Z)
        invalid = ~finite
        Z_masked = np.ma.masked_invalid(Z)

        extended_levels = [-np.inf, *normalized_levels, np.inf]
        fills: list[PolyCollection] = []
        if finite.any():
            filled_contours = ax.contourf(
                X,
                Y,
                Z_masked,
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
                X,
                Y,
                invalid_mask,
                cmap=ListedColormap(["#bdbdbd"]),
                shading="flat",
                antialiased=False,
                zorder=1.5,
            )

        lines: list[Line2D] = []
        if show_lines and finite.any():
            contour_lines = ax.contour(X, Y, Z_masked, levels=normalized_levels)
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
                        facecolor="#bdbdbd",
                        alpha=1.0,
                        label="Out of model limits",
                    )
                )
            legend_artist = ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(handles), 3),
                frameon=False,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(self.x_axis.name)
        ax.set_ylabel(self.y_axis.name)
        if title is not None:
            ax.set_title(title)

        return ThresholdPlotResult(
            ax=ax,
            lines=lines,
            fills=fills,
            legend=legend_artist,
        )
