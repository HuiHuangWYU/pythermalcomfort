"""Shared helpers for Matplotlib threshold and summary plots."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
from matplotlib import colors as mcolors


@dataclass
class ThresholdsConfig:
    """Threshold and optional label/color configuration.

    Attributes:
        thresholds: Sorted threshold boundaries used to split regions.
        labels: Optional labels for each region. Length must be ``len(thresholds) + 1``.
        colors: Optional colors for each region. Length must be ``len(thresholds) + 1``.
    """

    thresholds: Sequence[float]
    labels: Sequence[str] | None = None
    colors: Sequence[str] | None = None


def _inspect_model_signature(
    model_func: Any,
) -> tuple[inspect.Signature, set[str], set[str], bool]:
    """Inspect a model callable and return argument metadata."""
    signature = inspect.signature(model_func)
    allowed_args = set(signature.parameters.keys())
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    required_args = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Signature.empty
    }
    return signature, allowed_args, required_args, accepts_var_kwargs


def _validate_model_kwargs(
    kwargs: Mapping[str, Any],
    *,
    allowed_args: set[str],
    required_args: set[str],
    accepts_var_kwargs: bool,
) -> None:
    """Validate kwargs against a model signature contract."""
    if not accepts_var_kwargs:
        invalid = sorted(key for key in kwargs if key not in allowed_args)
        if invalid:
            invalid_str = ", ".join(invalid)
            msg = f"Model does not accept parameter(s): {invalid_str}"
            raise ValueError(msg)

    missing = sorted(key for key in required_args if key not in kwargs)
    if missing:
        missing_str = ", ".join(missing)
        msg = f"Missing required parameter(s): {missing_str}"
        raise ValueError(msg)


def _extract_output_by_name(result: Any, output: str) -> Any:
    """Extract an output payload from a model result by name."""
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    candidates = [output_name]
    lowered = output_name.lower()
    if lowered != output_name:
        candidates.append(lowered)

    for candidate in candidates:
        if hasattr(result, candidate):
            return getattr(result, candidate)

    if isinstance(result, Mapping):
        for candidate in candidates:
            if candidate in result:
                return result[candidate]

    if isinstance(result, Number) and not isinstance(result, bool):
        if lowered in {"value", "scalar"}:
            return result
        msg = (
            f"Could not extract output '{output_name}' from scalar model result. "
            "Use output='value' for scalar-returning models."
        )
        raise ValueError(msg)

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


def _extract_output_value(result: Any, output: str) -> float:
    """Extract a scalar output value from a model result."""
    return float(_extract_output_by_name(result, output))


def _extract_output_payload(result: Any, output: str) -> Any:
    """Extract a scalar or vector-capable output payload for contour plotting."""
    payload = _extract_output_by_name(result, output)
    if isinstance(payload, list | tuple | np.ndarray | Number) and not isinstance(
        payload, bool
    ):
        return payload
    return payload


def _normalize_levels(levels: Sequence[float]) -> list[float]:
    """Validate and normalize threshold levels."""
    if len(levels) == 0:
        raise ValueError("thresholds must contain at least one threshold.")

    try:
        normalized = sorted(float(level) for level in levels)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must contain only numeric values.") from exc

    if not all(np.isfinite(normalized)):
        raise ValueError("thresholds must contain only finite values.")

    if any(
        right <= left for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("thresholds must be strictly increasing after sorting.")

    return normalized


def _build_region_labels(
    *,
    output: str,
    levels: Sequence[float],
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Build labels for threshold regions."""
    normalized_levels = _normalize_levels(levels)
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    n_regions = len(normalized_levels) + 1
    if labels is not None:
        if len(labels) != n_regions:
            msg = f"labels must have length {n_regions} (got {len(labels)})."
            raise ValueError(msg)
        return [str(label) for label in labels]

    out_name = output_name.upper()
    region_labels = [f"{out_name} < {normalized_levels[0]:g}"]
    for lower, upper in zip(normalized_levels, normalized_levels[1:], strict=False):
        region_labels.append(f"{lower:g} <= {out_name} < {upper:g}")
    region_labels.append(f"{out_name} >= {normalized_levels[-1]:g}")
    return region_labels


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
        "summary_blue_neutral_red",
        ["#4c78a8", "#f2f2f2", "#e15759"],
    )
    positions = np.linspace(0.0, 1.0, n_regions)
    return [mcolors.to_hex(cmap(value)) for value in positions]


def _resolve_region_colors(
    *,
    n_regions: int,
    colors: Sequence[str] | None = None,
) -> list[str]:
    """Resolve user or default region colors and validate length/format."""
    if colors is None:
        return _default_region_colors(n_regions)

    resolved = [str(color) for color in colors]
    if len(resolved) != n_regions:
        msg = f"colors must have length {n_regions} (got {len(resolved)})."
        raise ValueError(msg)
    invalid = [color for color in resolved if not mcolors.is_color_like(color)]
    if invalid:
        msg = f"Invalid color value(s): {', '.join(invalid)}."
        raise ValueError(msg)
    return resolved


def _apply_default_links_to_kwargs(
    kwargs: dict[str, Any],
    *,
    allowed_args: set[str],
    default_links: Mapping[str, str],
) -> dict[str, Any]:
    resolved = dict(kwargs)
    for target, source in default_links.items():
        if (
            (not allowed_args or target in allowed_args)
            and (not allowed_args or source in allowed_args)
            and target not in resolved
            and source in resolved
        ):
            resolved[target] = resolved[source]
    return resolved


def _is_light_color(color: str) -> bool:
    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance > 0.7
