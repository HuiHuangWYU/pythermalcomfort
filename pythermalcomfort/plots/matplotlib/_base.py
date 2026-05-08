"""Abstract base class shared by all Matplotlib plot classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from matplotlib.axes import Axes

from pythermalcomfort.plots.matplotlib._shared import (
    BasePlotResult,
    RegionConfig,
    ThresholdsConfig,
    _configure_regions,
)


class BasePlot(ABC):
    """Abstract base for all pythermalcomfort Matplotlib plot classes.

    Provides the shared :meth:`set_regions` implementation and enforces the
    :meth:`plot` contract via :func:`~abc.abstractmethod`.

    Subclasses must implement :meth:`plot`.  They may override
    :meth:`set_regions` to add input-specific validation (e.g. DataFrame
    column checks in :class:`~pythermalcomfort.plots.matplotlib.SummaryPlot`)
    before delegating to this base implementation via ``super()``.
    """

    def __init__(self) -> None:
        self._region_config: RegionConfig | None = None

    def set_regions(
        self,
        *,
        output: str,
        thresholds: ThresholdsConfig | Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> BasePlot:
        """Configure output regions.

        Accepts either a pre-built :class:`ThresholdsConfig` or raw threshold
        values (with optional *labels* and *colors*).

        Parameters
        ----------
        output : str
            Output field or column name.
        thresholds : ThresholdsConfig or sequence of float
            A :class:`ThresholdsConfig` instance **or** a sequence of numeric
            boundary values.  When a ``ThresholdsConfig`` is supplied, *labels*
            and *colors* must not be given separately.
        labels : sequence of str, optional
            Region labels.  Must have length ``len(thresholds) + 1`` when
            provided.
        colors : sequence of str, optional
            Region colors.  Must have length ``len(thresholds) + 1`` when
            provided.

        Returns
        -------
        BasePlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``output`` is not a string.
        ValueError
            If output name is empty, if *labels* or *colors* are supplied
            together with a ``ThresholdsConfig``, or if thresholds/labels/colors
            are invalid.
        """
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

        self._region_config = _configure_regions(output=output, thresholds=config)
        return self

    @abstractmethod
    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> BasePlotResult:
        """Render the plot.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is
            created.
        title : str, optional
            Optional chart title.

        Returns
        -------
        BasePlotResult
            Result with figure, axis, and plot-specific artist handles.
        """
