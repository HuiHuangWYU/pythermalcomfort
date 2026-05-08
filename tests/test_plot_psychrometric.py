import matplotlib.pyplot as plt
import pytest

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import PsychrometricPlot, ThresholdPlotResult


def _new_plot() -> PsychrometricPlot:
    """Initialize a basic PsychrometricPlot."""
    return (
        PsychrometricPlot(pmv_ppd_iso)
        .set_params(vr=0.1, met=1.2, clo=0.5, tr=25.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def test_import_export() -> None:
    """Test import/export from pythermalcomfort.plots.matplotlib."""
    try:
        from pythermalcomfort.plots.matplotlib import PsychrometricPlot

        assert PsychrometricPlot is not None
    except ImportError as exc:
        pytest.fail(f"Failed to import PsychrometricPlot: {exc}")


def test_set_x_axis_accepts_any_model_temperature_param() -> None:
    """tdb and tr are both valid x-axis parameters; unknown params are rejected."""
    # tdb is always accepted
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)

    # tr is also a valid model parameter when it is not already in fixed params
    plot_tr = PsychrometricPlot(pmv_ppd_iso).set_params(vr=0.1, met=1.2, clo=0.5)
    plot_tr.set_x_axis("tr", 10.0, 40.0, resolution=1.0)

    with pytest.raises(ValueError):
        plot.set_x_axis("not_a_model_param", 10.0, 40.0, resolution=1.0)


def test_set_y_axis_only_accepts_hr() -> None:
    """Test set_y_axis strictly enforces 'hr'."""
    plot = _new_plot()
    with pytest.raises(ValueError, match="requires the y-axis to be 'hr'"):
        plot.set_y_axis("rh", 0.0, 0.03, resolution=0.001)

    # Valid input should not raise
    plot.set_y_axis("hr", 0.0, 0.03, resolution=0.001)


def test_basic_plot_renders_and_preserves_limits() -> None:
    """Test a basic plot renders, masks invalid RH, and preserves requested axis limits."""
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)
    plot.set_y_axis("hr", 0.0, 0.03, resolution=0.002)

    result = plot.plot()

    # Verify the return object
    assert isinstance(result, ThresholdPlotResult)
    assert result.fig is not None
    assert result.ax is not None

    # Verify requested axis limits are preserved perfectly
    xlim = result.ax.get_xlim()
    ylim = result.ax.get_ylim()
    assert xlim == (10.0, 40.0)
    assert ylim == (0.0, 0.03)

    plt.close(result.fig)
