"""Class-based psychrometric charting with contour threshold regions."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes

from pythermalcomfort.plots.matplotlib._shared import (
    _apply_default_links_to_kwargs,
    _extract_output_by_name,
)
from pythermalcomfort.plots.matplotlib.threshold import (
    ThresholdPlot,
    ThresholdPlotResult,
    _AxisConfig,
    _parse_axis_range,
    _validate_resolution,
)
from pythermalcomfort.utilities import p_sat, psy_ta_rh


class PsychrometricPlot(ThresholdPlot):
    """Configure and render a psychrometric chart with threshold regions.
    
    This class inherits from ThresholdPlot but strictly enforces 'tdb' (dry-bulb
    temperature) on the x-axis and 'hr' (humidity ratio) on the y-axis. It handles
    the reverse calculation of relative humidity for model evaluation and draws
    constant relative humidity lines in the background.
    """

    def __init__(self, model_func: Any) -> None:
        super().__init__(model_func)

    def set_x_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set x-axis strictly to dry-bulb temperature (tdb)."""
        if name != "tdb":
            raise ValueError("PsychrometricPlot strictly requires the x-axis to be 'tdb'.")
        return super().set_x_axis(name, min_val, max_val, resolution=resolution)

    def set_y_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set y-axis strictly to humidity ratio (hr) bypassing standard validation."""
        if name != "hr":
            raise ValueError("PsychrometricPlot strictly requires the y-axis to be 'hr'.")
        
        # Bypass _allowed_args check since thermal models typically don't accept 'hr'
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
        """Evaluate the grid by converting HR back to RH before calling the model."""
        x_flat = np.asarray(x).ravel()  # tdb
        y_flat = np.asarray(y).ravel()  # hr

        # 1. Reverse-calculate RH from Humidity Ratio and Tdb
        p_atm = 101325.0
        p_vap = (y_flat * p_atm) / (0.62198 + y_flat)
        p_sat_values = p_sat(x_flat)
        rh_flat = (p_vap / p_sat_values) * 100.0

        # 2. Mask physically impossible combinations (RH > 100% or RH < 0%)
        invalid_mask = (rh_flat < 0) | (rh_flat > 100)
        
        # Prevent math warnings/errors in the thermal model by capping invalid RH
        rh_safe = np.where(invalid_mask, 50.0, rh_flat)

        # 3. Build model kwargs using 'rh' instead of 'hr' (Using _fixed_values)
        grid_kwargs: dict[str, Any] = dict(self._fixed_values)
        grid_kwargs["tdb"] = x_flat
        grid_kwargs["rh"] = rh_safe
        
        grid_kwargs = _apply_default_links_to_kwargs(
            grid_kwargs,
            allowed_args=self._allowed_args,
            default_links=self._default_links,
        )

        try:
            result = self._model_func(**grid_kwargs)
        except Exception as exc:
            msg = f"Failed to evaluate model on psychrometric grid: {exc}"
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

        # 4. Re-apply physical bounds mask (set impossible values to NaN)
        z_flat[invalid_mask] = np.nan
        return z_flat.reshape(x.shape)

    def plot(self, **kwargs: Any) -> ThresholdPlotResult:
        """Render the psychrometric chart and overlay constant RH lines."""
        # Call the parent method to render the model contours and invalid mask
        result = super().plot(**kwargs)
        ax: Axes = result.ax

        tdb_dense = np.linspace(self._x_axis.min_val, self._x_axis.max_val, 500)
        
        # 1. Mask out the physically impossible region (RH > 100%) with white.
        # This gracefully hides the base ThresholdPlot's out-of-model grey mask in the top-left area.
        hr_100 = psy_ta_rh(tdb_dense, np.full_like(tdb_dense, 100.0)).hr
        ax.fill_between(
            tdb_dense,
            hr_100,
            self._y_axis.max_val,
            color="white",
            zorder=1.6,  # Sits above the base grey mask (z=1.5) but below grid lines (z=2.0)
            edgecolor="none"
        )

        # 2. Draw the constant Relative Humidity (RH) background curves
        for rh_target in range(10, 110, 10):
            rh_array = np.full_like(tdb_dense, rh_target)
            psy_vals = psy_ta_rh(tdb_dense, rh_array)
            hr_line = psy_vals.hr
            
            # Only plot the segment of the line that falls within the chart's Y-limits
            valid_hr_mask = hr_line <= self._y_axis.max_val
            ax.plot(
                tdb_dense[valid_hr_mask], 
                hr_line[valid_hr_mask], 
                color="#a0a0a0",  # Solid light grey chosen to avoid alpha rendering artifacts
                linestyle=":", 
                linewidth=0.8, 
                zorder=2.0
            )

            # Optional: Add text labels for the RH lines at the top curve bounds
            if np.any(valid_hr_mask):
                last_idx = np.where(valid_hr_mask)[0][-1]
                ax.text(
                    tdb_dense[last_idx], 
                    hr_line[last_idx] + 0.0002, 
                    f"{rh_target}%", 
                    color="#a0a0a0", 
                    fontsize=8,
                    zorder=2.0
                )

        # Enforce proper axis limits after drawing background lines
        ax.set_xlim(self._x_axis.min_val, self._x_axis.max_val)
        ax.set_ylim(self._y_axis.min_val, self._y_axis.max_val)

        return result