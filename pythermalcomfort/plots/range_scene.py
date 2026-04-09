"""Class-based threshold-region scene plotting."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass
class AxisConfig:
    name: str
    min_val: float
    max_val: float


@dataclass
class PlotResult:
    ax: plt.Axes
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


def _extract_output_value(result: Any, output: str) -> float:
    """Extract a scalar output value from a model result using a requested name."""
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    candidates = [output_name]
    lowered = output_name.lower()
    if lowered != output_name:
        candidates.append(lowered)

    for candidate in candidates:
        if hasattr(result, candidate):
            return float(getattr(result, candidate))

    if isinstance(result, Mapping):
        for candidate in candidates:
            if candidate in result:
                return float(result[candidate])

    if isinstance(result, Number) and not isinstance(result, bool):
        if lowered in {"value", "scalar"}:
            return float(result)
        raise ValueError(
            f"Could not extract output '{output_name}' from scalar model result. "
            "Use output='value' for scalar-returning models."
        )

    available_outputs: list[str] = []
    if isinstance(result, Mapping):
        available_outputs = [str(key) for key in result.keys()]
    else:
        result_dict = getattr(result, "__dict__", None)
        if isinstance(result_dict, dict):
            available_outputs = [
                name
                for name, value in result_dict.items()
                if not name.startswith("_") and not callable(value)
            ]

    msg = f"Could not extract output '{output_name}' from model result."
    if available_outputs:
        shown = ", ".join(sorted(available_outputs)[:6])
        if len(available_outputs) > 6:
            shown = f"{shown}, ..."
        msg = f"{msg} Available outputs: {shown}."
    raise ValueError(msg)


def _default_region_colors(n_regions: int) -> list[str]:
    """Return default region colors in a cool-neutral-warm progression."""
    if n_regions < 1:
        raise ValueError("n_regions must be at least 1.")
    if n_regions == 1:
        return ["#f2f2f2"]
    if n_regions == 2:
        return ["#4c78a8", "#e15759"]
    if n_regions == 3:
        return ["#4c78a8", "#f2f2f2", "#e15759"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "range_scene_blue_neutral_red",
        ["#4c78a8", "#f2f2f2", "#e15759"],
    )
    positions = np.linspace(0.0, 1.0, n_regions)
    return [mcolors.to_hex(cmap(v)) for v in positions]


class RangeScene:
    def __init__(self, model_func: Any) -> None:
        self.model_func = model_func

        self.x_axis: AxisConfig | None = None
        self.y_axis: AxisConfig | None = None
        self._default_links: dict[str, str] = {"tr": "tdb"}
        self.fixed_values: dict[str, Any] = {}

        self._signature: inspect.Signature | None = None
        self._allowed_args: set[str] = set()
        self._required_args: set[str] = set()
        self._accepts_var_kwargs: bool = False

    def allowed_parameters(self) -> list[str]:
        """Return sorted parameter names accepted by the model signature."""
        self._read_model_signature()
        return sorted(self._allowed_args)

    def required_parameters(self) -> list[str]:
        """Return sorted required parameter names from the model signature."""
        self._read_model_signature()
        return sorted(self._required_args)

    def _extract_axis_spec(
        self,
        axis_label: str,
        axis_range: dict[str, Any],
        other_axis: AxisConfig | None = None,
    ) -> AxisConfig:
        self._read_model_signature()

        if len(axis_range) != 1:
            raise ValueError(f"{axis_label}() requires exactly one keyword range.")

        ((axis_name, value),) = axis_range.items()

        if other_axis is not None and axis_name == other_axis.name:
            raise ValueError("x and y axis parameters must be different.")

        if axis_name not in self._allowed_args:
            msg = (
                f"{axis_label}() received invalid parameter: {axis_name}. "
                "Use model argument names from the selected function."
            )
            raise ValueError(msg)

        min_val, max_val = _parse_axis_range(axis_name, value)

        if axis_name in self.fixed_values:
            msg = (
                f"parameters() already contains axis parameter '{axis_name}'. "
                f"Remove it from parameters() when using {axis_label}()."
            )
            raise ValueError(msg)

        return AxisConfig(name=axis_name, min_val=min_val, max_val=max_val)

    def x(self, **axis_range: Any) -> RangeScene:
        self.x_axis = self._extract_axis_spec(
            axis_label="x",
            axis_range=axis_range,
            other_axis=self.y_axis,
        )
        return self

    def y(self, **axis_range: Any) -> RangeScene:
        self.y_axis = self._extract_axis_spec(
            axis_label="y",
            axis_range=axis_range,
            other_axis=self.x_axis,
        )
        return self

    def parameters(self, **kwargs: Any) -> RangeScene:
        """Set fixed model parameters that are not used as axes.

        Example:
            scene.parameters(vr=0.1, met=1.2, clo=0.5)
        """
        self._read_model_signature()
        parameter_kwargs = dict(kwargs)

        if not self._accepts_var_kwargs:
            invalid = sorted(k for k in parameter_kwargs if k not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = (
                    f"parameters() received invalid parameter(s): {invalid_str}. "
                    "Use scene.allowed_parameters() to inspect valid keys."
                )
                raise ValueError(msg)

        x_name = self.x_axis.name if self.x_axis is not None else None
        y_name = self.y_axis.name if self.y_axis is not None else None
        for key, value in parameter_kwargs.items():
            if key in (x_name, y_name):
                msg = (
                    f"parameters() cannot set axis parameter '{key}'. "
                    "Set it through x(...) or y(...)."
                )
                raise ValueError(msg)
            self.fixed_values[key] = value
        return self

    def _read_model_signature(self) -> None:
        """Read model signature once and cache required/allowed argument names."""
        if self._signature is not None:
            return

        self._signature = inspect.signature(self.model_func)
        self._allowed_args = set(self._signature.parameters.keys())
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._signature.parameters.values()
        )

        self._required_args = {
            name
            for name, p in self._signature.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.default is inspect.Signature.empty
        }

    def _validate_call_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate model-call kwargs against signature."""
        self._read_model_signature()

        if not self._accepts_var_kwargs:
            invalid = sorted(k for k in kwargs if k not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = f"Model does not accept parameter(s): {invalid_str}"
                raise ValueError(msg)

        missing = sorted(k for k in self._required_args if k not in kwargs)
        if missing:
            missing_str = ", ".join(missing)
            msg = f"Missing required parameter(s): {missing_str}"
            raise ValueError(msg)

    def _build_call_kwargs(self, x_value: float, y_value: float) -> dict[str, Any]:
        """Build model kwargs for one x/y point."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call x(...) and y(...) first.")
        self._read_model_signature()

        call_kwargs: dict[str, Any] = dict(self.fixed_values)
        call_kwargs[self.x_axis.name] = float(x_value)
        call_kwargs[self.y_axis.name] = float(y_value)

        if "tr" in self._allowed_args and "tdb" in self._allowed_args:
            if "tr" not in call_kwargs and "tdb" in call_kwargs:
                call_kwargs["tr"] = call_kwargs["tdb"]
            elif "tdb" not in call_kwargs and "tr" in call_kwargs:
                call_kwargs["tdb"] = call_kwargs["tr"]
        else:
            for target, source in self._default_links.items():
                if (
                    target in self._allowed_args
                    and source in self._allowed_args
                    and target not in call_kwargs
                    and source in call_kwargs
                ):
                    call_kwargs[target] = call_kwargs[source]

        self._validate_call_kwargs(call_kwargs)
        return call_kwargs

    def _compute_threshold_curves(
        self,
        *,
        output: str,
        thresholds: list[float],
        x_step: float,
        y_step: float,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Compute x(y) threshold curves by scanning temperature for each RH."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call x(...) and y(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")

        if (
            self.x_axis.min_val is None
            or self.x_axis.max_val is None
            or self.y_axis.min_val is None
            or self.y_axis.max_val is None
        ):
            raise ValueError("Axes are not fully set. Call x(...) and y(...) first.")

        self._build_call_kwargs(
            float(self.x_axis.min_val),
            float(self.y_axis.min_val),
        )

        x_values = np.arange(self.x_axis.min_val, self.x_axis.max_val + 1e-12, x_step)
        y_values = np.arange(self.y_axis.min_val, self.y_axis.max_val + 1e-12, y_step)

        fill_curves: list[np.ndarray] = []
        line_curves: list[np.ndarray] = []
        x_min = float(self.x_axis.min_val)
        x_max = float(self.x_axis.max_val)
        has_any_valid_output = False

        for _ in thresholds:
            fill_curves.append(np.full(len(y_values), np.nan, dtype=float))
            line_curves.append(np.full(len(y_values), np.nan, dtype=float))

        for i, y in enumerate(y_values):
            z = np.full(len(x_values), np.nan, dtype=float)
            for j, x in enumerate(x_values):
                kwargs = self._build_call_kwargs(float(x), float(y))
                try:
                    result = self.model_func(**kwargs)
                except Exception:
                    z[j] = np.nan
                    continue

                z_val = _extract_output_value(result, output)
                z[j] = z_val
                if np.isfinite(z_val):
                    has_any_valid_output = True

            if not np.isfinite(z).any():
                continue

            for k, threshold in enumerate(thresholds):
                r = z - float(threshold)

                crossing_x = np.nan
                found_crossing = False
                for j in range(len(x_values) - 1):
                    x0, x1 = x_values[j], x_values[j + 1]
                    r0, r1 = r[j], r[j + 1]

                    if not np.isfinite(r0) or not np.isfinite(r1):
                        continue

                    if r0 == 0.0:
                        crossing_x = float(x0)
                        found_crossing = True
                        break

                    if r0 * r1 < 0.0:
                        crossing_x = float(x0 + (-r0) * (x1 - x0) / (r1 - r0))
                        found_crossing = True
                        break

                    if r1 == 0.0:
                        crossing_x = float(x1)
                        found_crossing = True
                        break

                if found_crossing:
                    fill_curves[k][i] = crossing_x
                    line_curves[k][i] = crossing_x
                    continue

                finite = np.isfinite(r)
                if not finite.any():
                    continue

                rf = r[finite]
                if np.all(rf < 0.0):
                    fill_curves[k][i] = x_max
                elif np.all(rf > 0.0):
                    fill_curves[k][i] = x_min
                else:
                    continue

        if not has_any_valid_output:
            raise ValueError(
                "No valid model outputs could be computed for this plot. "
                "Check fixed parameters and axis ranges."
            )

        return y_values, fill_curves, line_curves

    def plot(
        self,
        *,
        output: str,
        levels: list[float],
        colors: list[str] | None = None,
        x_step: float,
        y_step: float,
        ax: plt.Axes | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: dict[str, Any] | None = None,
        fill_kws: dict[str, Any] | None = None,
    ) -> PlotResult:
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call x(...) and y(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")
        if len(levels) == 0:
            raise ValueError("levels must contain at least one threshold.")
        if (
            self.x_axis.min_val is None
            or self.x_axis.max_val is None
            or self.y_axis.min_val is None
            or self.y_axis.max_val is None
        ):
            raise ValueError("Axes are not fully set. Call x(...) and y(...) first.")

        sorted_levels = sorted(float(v) for v in levels)

        n_regions = len(sorted_levels) + 1
        if colors is None:
            region_colors = _default_region_colors(n_regions)
        else:
            if len(colors) != n_regions:
                msg = f"colors must have length {n_regions} (got {len(colors)})."
                raise ValueError(msg)
            region_colors = colors

        line_opts = dict(line_kws or {})
        fill_opts = dict(fill_kws or {})
        if "color" in fill_opts or "facecolor" in fill_opts:
            raise ValueError(
                "fill_kws cannot include 'color' or 'facecolor'. "
                "Use colors=[...] to control region colors."
            )
        fill_opts.setdefault("alpha", 0.65)

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        y_values, fill_curves, line_curves = self._compute_threshold_curves(
            output=output,
            thresholds=sorted_levels,
            x_step=x_step,
            y_step=y_step,
        )

        x_lo = float(self.x_axis.min_val)
        x_hi = float(self.x_axis.max_val)
        left_const = np.full_like(y_values, x_lo, dtype=float)
        right_const = np.full_like(y_values, x_hi, dtype=float)

        fills: list[PolyCollection] = []
        lines: list[Line2D] = []
        legend_artist: Legend | None = None

        region_pairs: list[tuple[np.ndarray, np.ndarray]] = [
            (left_const, fill_curves[0]),
            *[
                (fill_curves[i], fill_curves[i + 1])
                for i in range(len(fill_curves) - 1)
            ],
            (fill_curves[-1], right_const),
        ]

        for i, (x_left, x_right) in enumerate(region_pairs):
            valid = np.isfinite(x_left) & np.isfinite(x_right)
            if valid.any():
                region_fill_opts = dict(fill_opts)
                region_fill_opts["color"] = region_colors[i]
                poly = ax.fill_betweenx(
                    y_values[valid],
                    x_left[valid],
                    x_right[valid],
                    **region_fill_opts,
                )
                fills.append(poly)

        if show_lines:
            for curve in line_curves:
                valid = np.isfinite(curve)
                if valid.any():
                    (line,) = ax.plot(
                        curve[valid],
                        y_values[valid],
                        **line_opts,
                    )
                    lines.append(line)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(float(self.y_axis.min_val), float(self.y_axis.max_val))
        ax.set_xlabel(self.x_axis.name)
        ax.set_ylabel(self.y_axis.name)

        if legend:
            out_name = output.strip().upper()
            labels: list[str] = []
            for i in range(len(region_colors)):
                if i == 0:
                    labels.append(f"{out_name} < {sorted_levels[0]:g}")
                elif i == len(region_colors) - 1:
                    labels.append(f"{out_name} > {sorted_levels[-1]:g}")
                else:
                    lo = sorted_levels[i - 1]
                    hi = sorted_levels[i]
                    labels.append(f"{lo:g} <= {out_name} <= {hi:g}")

            handles = [
                Patch(
                    facecolor=region_colors[i],
                    alpha=fill_opts.get("alpha"),
                    label=labels[i],
                )
                for i in range(len(region_colors))
            ]
            legend_artist = ax.legend(
                handles=handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=min(len(handles), 3),
                frameon=False,
            )

        return PlotResult(ax=ax, lines=lines, fills=fills, legend=legend_artist)
