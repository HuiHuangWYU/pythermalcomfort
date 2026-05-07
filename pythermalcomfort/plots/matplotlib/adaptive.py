"""Adaptive comfort chart plotting for ASHRAE 55 and EN 16798."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._shared import BasePlotResult

# ── constants ──────────────────────────────────────────────────────────────

_N_POINTS: int = 200

_CENTER_LINE_LABEL: str = "Comfort Temperature"
_CENTER_LINE_DEFAULTS: dict[str, Any] = {
    "color": "#333333",
    "linewidth": 1.5,
    "linestyle": "--",
}


# ── band configuration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BandDef:
    """Internal definition of one comfort band."""

    key: str
    low_field: str
    up_field: str
    default_label: str
    default_color: str


_STANDARD_CONFIGS: dict[str, dict[str, Any]] = {
    "ashrae": {
        "center_field": "tmp_cmf",
        "t_rm_range": (10.0, 33.5),
        "required_params": {"tdb", "tr", "v"},
        "bands": [
            _BandDef(
                "80", "tmp_cmf_80_low", "tmp_cmf_80_up", "80% Acceptability", "#B3D9FF"
            ),
            _BandDef(
                "90", "tmp_cmf_90_low", "tmp_cmf_90_up", "90% Acceptability", "#6BB3FF"
            ),
        ],
    },
    "en": {
        "center_field": "tmp_cmf",
        "t_rm_range": (10.0, 33.5),
        "required_params": {"tdb", "tr", "v"},
        "bands": [
            _BandDef(
                "cat_iii",
                "tmp_cmf_cat_iii_low",
                "tmp_cmf_cat_iii_up",
                "Category III",
                "#C5E0B4",
            ),
            _BandDef(
                "cat_ii",
                "tmp_cmf_cat_ii_low",
                "tmp_cmf_cat_ii_up",
                "Category II",
                "#A9D18E",
            ),
            _BandDef(
                "cat_i",
                "tmp_cmf_cat_i_low",
                "tmp_cmf_cat_i_up",
                "Category I",
                "#70AD47",
            ),
        ],
    },
}

_BAND_KEYS: dict[str, list[str]] = {
    std: [b.key for b in cfg["bands"]] for std, cfg in _STANDARD_CONFIGS.items()
}


# ── public config ──────────────────────────────────────────────────────────


@dataclass
class BandsConfig:
    """Reusable comfort band configuration.

    Controls which bands are displayed and their appearance.
    Follows the same pattern as :class:`ThresholdsConfig`.

    Attributes:
        show: Band keys to display.  If ``None``, all bands are shown.

            - ASHRAE keys: ``"80"``, ``"90"``
            - EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``

        labels: Custom labels for the visible bands.  Must have the same
            length as *show* (or the total number of bands if *show* is
            ``None``).  If ``None``, default labels are used.
        colors: Custom colors for the visible bands.  Same length rule
            as *labels*.  If ``None``, default colors are used.

    Example::

        # Show only 90% band with custom styling
        config = BandsConfig(
            show=["90"],
            labels=["90% Comfort Zone"],
            colors=["#FF6B6B"],
        )

        # Customize all ASHRAE bands
        config = BandsConfig(
            labels=["Wider Zone", "Narrower Zone"],
            colors=["#AECDE1", "#5BA3CF"],
        )
    """

    show: Sequence[str] | None = None
    labels: Sequence[str] | None = None
    colors: Sequence[str] | None = None

    def _validate(self, standard: str) -> None:
        """Validate against a specific standard's band keys."""
        valid_keys = set(_BAND_KEYS[standard])
        n_visible = (
            len(self.show) if self.show is not None else len(_BAND_KEYS[standard])
        )

        if self.show is not None:
            invalid = [k for k in self.show if k not in valid_keys]
            if invalid:
                msg = (
                    f"Invalid band key(s): {', '.join(invalid)}. "
                    f"Valid keys for '{standard}': {', '.join(sorted(valid_keys))}"
                )
                raise ValueError(msg)

        if self.labels is not None:
            if len(self.labels) != n_visible:
                msg = f"labels must have length {n_visible} (got {len(self.labels)})."
                raise ValueError(msg)

        if self.colors is not None:
            if len(self.colors) != n_visible:
                msg = f"colors must have length {n_visible} (got {len(self.colors)})."
                raise ValueError(msg)
            bad = [c for c in self.colors if not is_color_like(c)]
            if bad:
                msg = f"Invalid color value(s): {', '.join(str(c) for c in bad)}"
                raise ValueError(msg)


# ── resolved band (internal) ──────────────────────────────────────────────


@dataclass(frozen=True)
class _ResolvedBand:
    """A band with all overrides applied, ready to render."""

    low_field: str
    up_field: str
    label: str
    color: str


# ── result container ───────────────────────────────────────────────────────


@dataclass
class AdaptivePlotResult(BasePlotResult):
    """Result from :meth:`AdaptivePlot.plot`.

    Attributes:
        fig: Matplotlib figure.
        ax: Matplotlib axes.
        center_line: The comfort temperature center line artist, or ``None``.
        fills: Filled comfort band artists (outermost first).
        legend: Legend artist, or ``None``.
    """

    center_line: Line2D | None
    fills: list[PolyCollection]
    legend: Legend | None


# ── main class ─────────────────────────────────────────────────────────────


class AdaptivePlot:
    """Adaptive comfort chart for ASHRAE 55 or EN 16798.

    The chart displays comfort bands as filled regions on a plot of
    operative temperature (y-axis) versus prevailing mean outdoor
    temperature (x-axis).

    .. note::
        The model parameters ``tdb``, ``tr``, and ``v`` influence the
        **cooling effect (ce)**, which shifts the upper boundary of each
        comfort band upward when operative temperature exceeds 25 °C and
        air speed exceeds 0.6 m/s.  Different ``tdb``/``tr``/``v`` values
        will produce different upper boundary positions.  The lower
        boundaries and the center line are not affected by ce.

    Example::

        from pythermalcomfort.plots.matplotlib import AdaptivePlot

        result = (
            AdaptivePlot("ashrae")
            .set_params(tdb=25, tr=25, v=0.1)
            .plot(title="Adaptive Comfort (ASHRAE 55)")
        )

    Band keys for selection and customization:

    - **ASHRAE**: ``"80"`` (80% acceptability), ``"90"`` (90% acceptability)
    - **EN**: ``"cat_i"`` (Category I), ``"cat_ii"`` (Category II),
      ``"cat_iii"`` (Category III)
    """

    def __init__(
        self,
        standard: Literal["ashrae", "en"],
        *,
        t_running_mean_range: tuple[float, float] | None = None,
    ) -> None:
        """Initialize an adaptive comfort chart builder.

        Args:
            standard: Comfort standard, ``'ashrae'`` or ``'en'``.
            t_running_mean_range: Optional ``(min, max)`` override for the
                x-axis.  Defaults to the standard's applicability range
                (10-33.5 °C).
        """
        std = standard.lower().strip()
        if std not in _STANDARD_CONFIGS:
            msg = f"Unknown standard '{standard}'. Must be 'ashrae' or 'en'."
            raise ValueError(msg)

        self._standard = std
        self._cfg = _STANDARD_CONFIGS[std]
        self._fixed_params: dict[str, Any] = {}
        self._bands_config: BandsConfig | None = None

        if t_running_mean_range is not None:
            lo, hi = float(t_running_mean_range[0]), float(t_running_mean_range[1])
            if lo >= hi:
                raise ValueError("t_running_mean_range must have min < max.")
            self._t_rm_range = (lo, hi)
        else:
            self._t_rm_range = self._cfg["t_rm_range"]

    def set_params(self, **kwargs: Any) -> AdaptivePlot:
        """Set fixed model parameters.

        At minimum ``tdb``, ``tr``, and ``v`` are required.  Additional
        parameters such as ``units`` or ``limit_inputs`` are forwarded
        to the model unchanged.

        .. note::
            The values of ``tdb``, ``tr``, and ``v`` affect the **cooling
            effect (ce)**.  When operative temperature exceeds 25 °C and
            ``v`` ≥ 0.6 m/s, the upper boundaries of comfort bands shift
            upward.

        Returns:
            Self, to support method chaining.
        """
        self._fixed_params.update(kwargs)
        return self

    def set_bands(
        self,
        *,
        show: BandsConfig | Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> AdaptivePlot:
        """Customize which comfort bands are displayed and their appearance.

        Accepts either a pre-built :class:`BandsConfig` or raw parameters.

        Args:
            show: Controls which bands are shown and, optionally, their
                appearance.  Accepts three forms:

                - **BandsConfig** — a fully configured instance; *labels*
                  and *colors* must not be supplied separately.
                - **list of band keys** — selects which bands to display;
                  use *labels* / *colors* for appearance.
                - **None** — all bands are shown.

                ASHRAE keys: ``"80"``, ``"90"``.
                EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``.

            labels: Custom labels for the visible bands.  Must have the
                same length as *show* (or the total band count if *show*
                is ``None``).
            colors: Custom colors for the visible bands.  Same length
                rule as *labels*.

        Returns:
            Self, to support method chaining.

        Example::

            # Raw parameters
            .set_bands(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])

            # BandsConfig (reusable)
            config = BandsConfig(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
            .set_bands(show=config)
        """
        if isinstance(show, BandsConfig):
            if labels is not None or colors is not None:
                raise ValueError(
                    "labels and colors must not be provided separately when "
                    "show is a BandsConfig instance.  Set them inside the "
                    "BandsConfig instead."
                )
            config = show
        else:
            config = BandsConfig(show=show, labels=labels, colors=colors)

        config._validate(self._standard)
        self._bands_config = config
        return self

    def _resolve_bands(self) -> list[_ResolvedBand]:
        """Return bands with all user overrides applied."""
        all_defs: list[_BandDef] = self._cfg["bands"]
        cfg = self._bands_config

        # Filter by show list
        if cfg is not None and cfg.show is not None:
            show_set = set(cfg.show)
            visible_defs = [d for d in all_defs if d.key in show_set]
        else:
            visible_defs = list(all_defs)

        # Apply label/color overrides by position
        resolved: list[_ResolvedBand] = []
        for i, d in enumerate(visible_defs):
            label = d.default_label
            color = d.default_color
            if cfg is not None:
                if cfg.labels is not None:
                    label = str(cfg.labels[i])
                if cfg.colors is not None:
                    color = str(cfg.colors[i])
            resolved.append(
                _ResolvedBand(
                    low_field=d.low_field,
                    up_field=d.up_field,
                    label=label,
                    color=color,
                )
            )
        return resolved

    def _load_model(self) -> Any:
        """Import and return the model function."""
        if self._standard == "ashrae":
            from pythermalcomfort.models import adaptive_ashrae

            return adaptive_ashrae
        from pythermalcomfort.models import adaptive_en

        return adaptive_en

    def _validate_params(self) -> None:
        """Ensure all required fixed parameters have been set."""
        required: set[str] = self._cfg["required_params"]
        missing = sorted(required - set(self._fixed_params))
        if missing:
            msg = (
                f"Missing required parameter(s): {', '.join(missing)}. "
                "Call set_params() first."
            )
            raise ValueError(msg)

    def _evaluate(self, t_rm: np.ndarray) -> Any:
        """Call the model across the t_running_mean array."""
        model = self._load_model()
        kwargs = dict(self._fixed_params)
        kwargs["t_running_mean"] = t_rm
        try:
            return model(**kwargs)
        except Exception as exc:
            msg = f"Model evaluation failed: {exc}"
            raise ValueError(msg) from exc

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        xlabel: str | None = "Prevailing Mean Outdoor Temperature [°C]",
        ylabel: str | None = "Operative Temperature [°C]",
        legend: bool = True,
        grid: bool = True,
        show_center_line: bool = True,
        center_line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
    ) -> AdaptivePlotResult:
        """Render the adaptive comfort chart.

        Args:
            ax: Existing axes.  If ``None``, a new figure is created.
            title: Optional chart title.
            xlabel: X-axis label.  ``None`` to omit.
            ylabel: Y-axis label.  ``None`` to omit.
            legend: Whether to draw a legend.
            grid: Whether to display background grid lines.
            show_center_line: Whether to draw the comfort temperature
                center line.
            center_line_kws: Overrides for the center line (``ax.plot``).
            fill_kws: Shared overrides for all bands (``ax.fill_between``).
                Per-band colors are set via :meth:`set_bands`.
            legend_kws: Overrides for the legend (``ax.legend``).

        Returns:
            :class:`AdaptivePlotResult` with figure, axes, and artists.
        """
        self._validate_params()
        bands = self._resolve_bands()

        t_rm = np.linspace(self._t_rm_range[0], self._t_rm_range[1], _N_POINTS)
        result = self._evaluate(t_rm)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        fill_opts = dict(fill_kws or {})
        fill_opts.setdefault("alpha", 0.7)

        # Draw bands (outermost first)
        fills: list[PolyCollection] = []
        for band in bands:
            low = np.asarray(getattr(result, band.low_field), dtype=float)
            up = np.asarray(getattr(result, band.up_field), dtype=float)
            valid = np.isfinite(low) & np.isfinite(up)
            if not valid.any():
                continue
            fill = ax.fill_between(
                t_rm[valid],
                low[valid],
                up[valid],
                color=band.color,
                **fill_opts,
            )
            fills.append(fill)

        # Center line
        center_line_artist: Line2D | None = None
        if show_center_line:
            center_vals = np.asarray(
                getattr(result, self._cfg["center_field"]),
                dtype=float,
            )
            valid = np.isfinite(center_vals)
            if valid.any():
                cl_opts = dict(_CENTER_LINE_DEFAULTS)
                if center_line_kws:
                    cl_opts.update(center_line_kws)
                (center_line_artist,) = ax.plot(
                    t_rm[valid],
                    center_vals[valid],
                    **cl_opts,
                )

        # Legend (innermost first, then center line)
        legend_artist: Legend | None = None
        if legend:
            lg_opts = dict(legend_kws or {})
            lg_opts.setdefault("loc", "lower right")
            lg_opts.setdefault("frameon", True)
            lg_opts.setdefault("framealpha", 0.9)

            handles: list[Any] = []
            # Reversed so the narrowest (strictest) band appears first in the legend.
            for band in reversed(bands):
                handles.append(
                    Patch(
                        facecolor=band.color,
                        alpha=fill_opts.get("alpha", 0.7),
                        label=band.label,
                    )
                )
            if center_line_artist is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        label=_CENTER_LINE_LABEL,
                        **dict(_CENTER_LINE_DEFAULTS),
                    )
                )
            legend_artist = ax.legend(handles=handles, **lg_opts)

        # Grid
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        # Labels and limits
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.set_xlim(self._t_rm_range)

        return AdaptivePlotResult(
            fig=fig,
            ax=ax,
            center_line=center_line_artist,
            fills=fills,
            legend=legend_artist,
        )
