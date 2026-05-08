=====================================
Adding a Matplotlib Plot Class
=====================================

This guide defines the conventions that every Matplotlib plot class in
``pythermalcomfort.plots.matplotlib`` must follow.  Read it before adding a
new plot class or making structural changes to an existing one.

Plot families
=============

There are three distinct plot families.  Choose the right base class before
you start:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Family
     - When to use
     - Current examples
   * - **Grid plot**
     - Model is evaluated on a 2-D (x, y) mesh; output is rendered as filled
       contour regions.
     - ``ThresholdPlot``, ``PsychrometricPlot``
   * - **Line plot**
     - Band boundaries are smooth lines computed directly from an equation;
       no grid evaluation.
     - ``AdaptivePlot``
   * - **Summary plot**
     - Input is a pre-computed DataFrame; no model evaluation at all.
     - ``SummaryPlot``

Class contract
==============

Every plot class must implement the following public interface.  Method names
and return types are **not negotiable** — consistency across plot types is the
primary goal.

Required methods (all plot types)
----------------------------------

``__init__(model_func_or_data, ...)``
    Accept the model function (grid/line plots) or a DataFrame (summary
    plots) as the first positional argument.  Additional keyword-only
    arguments are allowed but must have defaults.

``set_regions(*, ...)``
    Configure which regions to display and how to label/colour them.
    Must return ``self`` to support method chaining.

    * **Grid and summary plots**: accept a ``ThresholdsConfig`` or raw
      sequence of numeric boundary values.
    * **Line (adaptive) plots**: accept a ``RegionsConfig`` that selects
      named comfort bands by key.

``plot(*, ax=None, title=None, ...)``
    Render the plot and return a ``<Name>PlotResult`` dataclass.  When
    ``ax`` is ``None``, create a new figure using the default size from
    ``_PlotDefaults.figsize``.

Additional methods (grid plots only)
--------------------------------------

``set_x_axis(name, min_val, max_val, *, resolution)``
    Set the x-axis parameter name, range, and grid resolution.
    Returns ``self``.

``set_y_axis(name, min_val, max_val, *, resolution)``
    Set the y-axis parameter name, range, and grid resolution.
    Returns ``self``.

``set_params(**kwargs)``
    Set fixed model parameters (those not mapped to an axis).
    Returns ``self``.

Additional methods (line plots only)
--------------------------------------

``set_x_axis(min_val, max_val)``
    Set the x-axis range.  No ``name`` or ``resolution`` — the axis variable
    is fixed by the standard.  Returns ``self``.

``set_y_axis(min_val, max_val)``
    Set the y-axis display range.  Returns ``self``.

``set_params(*, ...)``
    Set any model parameters that affect the rendered lines (e.g. air speed
    for the adaptive model).  Returns ``self``.

Naming conventions
==================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - What
     - Rule
   * - Region configuration method
     - Always ``set_regions``; never ``set_bands``, ``set_zones``, or
       ``set_thresholds``.
   * - Axis configuration methods
     - Always ``set_x_axis`` / ``set_y_axis``; never ``configure_axis``
       or ``set_range``.
   * - Configuration dataclasses
     - ``ThresholdsConfig`` for numeric thresholds (shared across grid/summary
       plots); ``RegionsConfig`` for named comfort bands (adaptive plots).
   * - Result dataclasses
     - ``<ClassName>Result`` (e.g. ``ThresholdPlotResult``); inherit from
       ``BasePlotResult``.
   * - Private helpers
     - Prefix with ``_`` (e.g. ``_resolve_regions``, ``_build_grid``).
   * - Visual defaults
     - All defaults live in ``_PlotDefaults`` in ``_shared.py``.  Add a
       nested class ``_PlotDefaults.<ClassName>`` for plot-specific values.
       Never hard-code numbers inside ``plot()``.

Fluent API pattern
==================

All setter methods must return ``self`` so callers can chain:

.. code-block:: python

    result = (
        MyPlot(some_model)
        .set_x_axis("tdb", 18.0, 34.0, resolution=0.2)
        .set_y_axis("rh", 20.0, 100.0, resolution=0.5)
        .set_params(met=1.2, clo=0.5)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot(title="My Plot")
    )

``plot()`` returns a frozen result dataclass, not ``self``.

Result dataclasses
==================

Every ``plot()`` method returns a dataclass that inherits from
``BasePlotResult`` (defined in ``_shared.py``):

.. code-block:: python

    @dataclass
    class MyPlotResult(BasePlotResult):
        """Result from :meth:`MyPlot.plot`.

        Attributes
        ----------
        fig : Figure
            Matplotlib figure.
        ax : Axes
            Matplotlib axes.
        # ... plot-specific artists
        """
        fills: list[PolyCollection]
        legend: Legend | None

Return enough artist handles (fills, lines, legend) that users can
post-process without re-running the plot.

Visual defaults
===============

All default values (colours, sizes, line widths, alpha values) must be
defined in ``_PlotDefaults`` inside ``pythermalcomfort/plots/matplotlib/_shared.py``.

* Top-level attributes (e.g. ``figsize``, ``fill_alpha``, ``title_fontsize``)
  are shared across all plot types.
* Nested classes (e.g. ``_PlotDefaults.Threshold``) hold plot-specific values.

Never embed magic numbers directly inside ``plot()``.  Reference a
``_PlotDefaults`` attribute instead.

Matplotlib style (rcParams)
----------------------------

Global Matplotlib style settings (spine visibility, legend frame, grid style)
are defined in the module-level dict ``_PYTHERMALCOMFORT_RC`` in ``_shared.py``
and applied inside every ``plot()`` call via ``mpl.rc_context(_PYTHERMALCOMFORT_RC)``.
This keeps individual ``plot()`` implementations free of boilerplate style calls.

If you add a new plot class, wrap the body of its ``plot()`` method the same way::

    import matplotlib as mpl
    from pythermalcomfort.plots.matplotlib._shared import _PYTHERMALCOMFORT_RC

    def plot(self, ...) -> MyPlotResult:
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            ...

Add new package-wide style settings to ``_PYTHERMALCOMFORT_RC``; add
plot-specific numeric defaults to a nested ``_PlotDefaults.<ClassName>`` class.

Base class hierarchy (target state)
=====================================

The intended inheritance tree after the planned refactor is:

.. code-block:: text

    BasePlot                      (_shared.py or _base.py)
    ├── GridBasePlot              (grid evaluation, set_x_axis, set_y_axis, set_params)
    │   ├── ThresholdPlot
    │   └── PsychrometricPlot
    ├── AdaptivePlot              (line-based; set_x_axis/set_y_axis simplified)
    └── SummaryPlot               (DataFrame-based; no axis config)

``BasePlot`` provides ``set_regions`` and the abstract ``plot`` stub.
``GridBasePlot`` provides axis configuration and the model grid evaluation
pipeline.

During the transition, existing classes may not yet fully conform.  The
refactor is tracked in the repository issues.

Step-by-step: adding a new plot
=================================

1. Decide which family the plot belongs to (grid, line, or summary).
2. Create ``pythermalcomfort/plots/matplotlib/<name>.py``.
3. Inherit from the appropriate base class.
4. Implement the required interface (see `Class contract`_ above).
5. Add a nested class ``_PlotDefaults.<ClassName>`` in ``_shared.py`` with
   every default value the new plot needs.
6. Export the class and its result dataclass from
   ``pythermalcomfort/plots/matplotlib/__init__.py``.
7. Add a notebook example under
   ``docs/documentation/plots/matplotlib/<name>.ipynb``.

Skeleton
--------

.. code-block:: python

    # pythermalcomfort/plots/matplotlib/my_plot.py
    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Any

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from pythermalcomfort.plots.matplotlib._shared import (
        BasePlotResult,
        ThresholdsConfig,
        _PlotDefaults,
        _configure_regions,
    )


    @dataclass
    class MyPlotResult(BasePlotResult):
        """Result from :meth:`MyPlot.plot`."""
        # add artist fields here


    class MyPlot:
        """One-line summary.

        Examples
        --------
        .. code-block:: python

            from pythermalcomfort.models import some_model
            from pythermalcomfort.plots.matplotlib import MyPlot

            result = (
                MyPlot(some_model)
                .set_regions(output="pmv", thresholds=[-0.5, 0.5])
                .plot(title="My Plot")
            )
        """

        def __init__(self, model_func: Any) -> None:
            self._model_func = model_func
            self._region_config = None

        def set_regions(
            self,
            *,
            output: str,
            thresholds: ThresholdsConfig | list[float],
            labels: list[str] | None = None,
            colors: list[str] | None = None,
        ) -> MyPlot:
            """Configure output regions.

            Returns
            -------
            MyPlot
                Self, to support method chaining.
            """
            if not isinstance(thresholds, ThresholdsConfig):
                thresholds = ThresholdsConfig(
                    thresholds=thresholds, labels=labels, colors=colors
                )
            self._region_config = _configure_regions(output=output, thresholds=thresholds)
            return self

        def plot(
            self,
            *,
            ax: Axes | None = None,
            title: str | None = None,
        ) -> MyPlotResult:
            """Render the plot.

            Returns
            -------
            MyPlotResult
                Result with figure, axes, and artists.
            """
            if self._region_config is None:
                raise ValueError("Call set_regions() before plot().")

            if ax is None:
                fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
            else:
                fig = ax.figure

            # ... rendering logic ...

            if title is not None:
                ax.set_title(title, fontsize=_PlotDefaults.title_fontsize)

            return MyPlotResult(fig=fig, ax=ax)

Testing plot classes
====================

* Test the fluent API: call the full chain and assert the returned figure is a
  ``matplotlib.figure.Figure`` and the axes is a ``matplotlib.axes.Axes``.
* Test that ``plot()`` raises ``ValueError`` when ``set_regions()`` has not
  been called.
* Test that invalid colours, out-of-order thresholds, and wrong label counts
  raise ``ValueError`` at ``set_regions()`` time, not at ``plot()`` time.
* Test ``set_x_axis`` / ``set_y_axis`` validation (min ≥ max, non-numeric
  resolution, axis name not in model signature).
* Do **not** assert on pixel-level rendering; assert on data and artist counts.

Reference implementations
==========================

* ``pythermalcomfort/plots/matplotlib/threshold.py`` — canonical grid plot.
* ``pythermalcomfort/plots/matplotlib/adaptive.py`` — canonical line plot;
  shows how constants are imported from model modules.
* ``pythermalcomfort/plots/matplotlib/_shared.py`` — ``_PlotDefaults``,
  ``ThresholdsConfig``, ``_configure_regions``, ``BasePlotResult``.
