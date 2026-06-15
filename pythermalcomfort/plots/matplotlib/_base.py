"""Abstract base classes shared by all Matplotlib plot classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes

from pythermalcomfort.plots.matplotlib._shared import (
    BasePlotResult,
    RegionConfig,
    _apply_default_links_to_kwargs,
    _AxisConfig,
    _extract_output_by_name,
    _inspect_model_signature,
    _parse_axis_range,
    _PlotDefaults,
    _validate_model_kwargs,
    _validate_resolution,
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


class GridBasePlot(BasePlot):
    """Intermediate base for grid-evaluated contour and psychrometric plots.

    Adds model-signature inspection, axis configuration, fixed-parameter
    management, and mesh-grid evaluation on top of :class:`BasePlot`.
    Concrete subclasses implement :meth:`plot` to render the grid output.

    Examples
    --------
    .. code-block:: python

        # Subclass and implement plot():
        class MyPlot(GridBasePlot):
            def plot(self, *, ax=None, title=None, **kwargs):
                x_min, x_max, y_min, y_max, X, Y = self._build_grid()
                Z = self._evaluate_grid_output(x=X, y=Y, output_name="pmv")
                ...
    """

    def __init__(self, model_func: Any) -> None:
        """Initialize the grid plot builder.

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
    ) -> GridBasePlot:
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
    ) -> GridBasePlot:
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
        GridBasePlot
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
    ) -> GridBasePlot:
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
        GridBasePlot
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

    def set_params(self, **kwargs: Any) -> GridBasePlot:
        """Set fixed model parameters used during grid evaluation.

        Parameters
        ----------
        **kwargs : Any
            Fixed model inputs passed unchanged to model evaluations.

        Returns
        -------
        GridBasePlot
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
