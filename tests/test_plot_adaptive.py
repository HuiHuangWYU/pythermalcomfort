"""Tests for adaptive comfort chart plotting."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythermalcomfort.plots.matplotlib.adaptive import (
    AdaptivePlot,
    AdaptivePlotResult,
    BandsConfig,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


# ── mock models ────────────────────────────────────────────────────────────


def _mock_ashrae(tdb, tr, t_running_mean, v, **kwargs):
    """Mimics adaptive_ashrae return structure."""
    t_rm = np.asarray(t_running_mean, dtype=float)
    t_cmf = 0.31 * t_rm + 17.8
    return SimpleNamespace(
        tmp_cmf=t_cmf,
        tmp_cmf_80_low=t_cmf - 3.5,
        tmp_cmf_80_up=t_cmf + 3.5,
        tmp_cmf_90_low=t_cmf - 2.5,
        tmp_cmf_90_up=t_cmf + 2.5,
        acceptability_80=np.ones_like(t_rm, dtype=bool),
        acceptability_90=np.ones_like(t_rm, dtype=bool),
    )


def _mock_en(tdb, tr, t_running_mean, v, **kwargs):
    """Mimics adaptive_en return structure."""
    t_rm = np.asarray(t_running_mean, dtype=float)
    t_cmf = 0.33 * t_rm + 18.8
    return SimpleNamespace(
        tmp_cmf=t_cmf,
        tmp_cmf_cat_i_low=t_cmf - 3.0,
        tmp_cmf_cat_i_up=t_cmf + 2.0,
        tmp_cmf_cat_ii_low=t_cmf - 4.0,
        tmp_cmf_cat_ii_up=t_cmf + 3.0,
        tmp_cmf_cat_iii_low=t_cmf - 5.0,
        tmp_cmf_cat_iii_up=t_cmf + 4.0,
        acceptability_cat_i=np.ones_like(t_rm, dtype=bool),
        acceptability_cat_ii=np.ones_like(t_rm, dtype=bool),
        acceptability_cat_iii=np.ones_like(t_rm, dtype=bool),
    )


def _mock_ashrae_with_nan(tdb, tr, t_running_mean, v, **kwargs):
    """Return NaN outside 15-30 range to simulate limit_inputs=True."""
    t_rm = np.asarray(t_running_mean, dtype=float)
    t_cmf = 0.31 * t_rm + 17.8
    invalid = (t_rm < 15) | (t_rm > 30)
    t_cmf = np.where(invalid, np.nan, t_cmf)
    return SimpleNamespace(
        tmp_cmf=t_cmf,
        tmp_cmf_80_low=t_cmf - 3.5,
        tmp_cmf_80_up=t_cmf + 3.5,
        tmp_cmf_90_low=t_cmf - 2.5,
        tmp_cmf_90_up=t_cmf + 2.5,
        acceptability_80=np.where(invalid, False, True),
        acceptability_90=np.where(invalid, False, True),
    )


# ── helper to patch model loading ──────────────────────────────────────────


def _new_ashrae_plot(monkeypatch=None, mock=None) -> AdaptivePlot:
    plot = AdaptivePlot("ashrae")
    if monkeypatch and mock:
        monkeypatch.setattr(plot, "_load_model", lambda: mock)
    plot.set_params(tdb=25, tr=25, v=0.1)
    return plot


def _new_en_plot(monkeypatch=None, mock=None) -> AdaptivePlot:
    plot = AdaptivePlot("en")
    if monkeypatch and mock:
        monkeypatch.setattr(plot, "_load_model", lambda: mock)
    plot.set_params(tdb=25, tr=25, v=0.1)
    return plot


# Constructor tests


def test_constructor_rejects_invalid_standard() -> None:
    with pytest.raises(ValueError, match="Unknown standard"):
        AdaptivePlot("invalid")


def test_constructor_accepts_ashrae() -> None:
    plot = AdaptivePlot("ashrae")
    assert plot._standard == "ashrae"


def test_constructor_accepts_en() -> None:
    plot = AdaptivePlot("en")
    assert plot._standard == "en"


def test_constructor_custom_t_rm_range() -> None:
    plot = AdaptivePlot("ashrae", t_running_mean_range=(15, 30))
    assert plot._t_rm_range == (15.0, 30.0)


def test_constructor_rejects_invalid_t_rm_range() -> None:
    with pytest.raises(ValueError, match="min < max"):
        AdaptivePlot("ashrae", t_running_mean_range=(30, 10))


# set_params tests


def test_set_params_stores_values() -> None:
    plot = AdaptivePlot("ashrae").set_params(tdb=25, tr=25, v=0.1)
    assert plot._fixed_params == {"tdb": 25, "tr": 25, "v": 0.1}


def test_set_params_chaining() -> None:
    plot = AdaptivePlot("ashrae").set_params(tdb=25).set_params(tr=25, v=0.1)
    assert plot._fixed_params == {"tdb": 25, "tr": 25, "v": 0.1}


def test_plot_rejects_missing_params(monkeypatch) -> None:
    plot = AdaptivePlot("ashrae")
    monkeypatch.setattr(plot, "_load_model", lambda: _mock_ashrae)
    with pytest.raises(ValueError, match="Missing required parameter"):
        plot.plot()


def test_plot_rejects_partial_params(monkeypatch) -> None:
    plot = AdaptivePlot("ashrae").set_params(tdb=25)
    monkeypatch.setattr(plot, "_load_model", lambda: _mock_ashrae)
    with pytest.raises(ValueError, match="Missing required parameter"):
        plot.plot()


# BandsConfig tests


def test_bands_config_default() -> None:
    cfg = BandsConfig()
    assert cfg.show is None
    assert cfg.labels is None
    assert cfg.colors is None


def test_bands_config_validates_show_keys() -> None:
    cfg = BandsConfig(show=["invalid_key"])
    with pytest.raises(ValueError, match="Invalid band key"):
        cfg._validate("ashrae")


def test_bands_config_validates_label_length() -> None:
    cfg = BandsConfig(show=["90"], labels=["A", "B"])
    with pytest.raises(ValueError, match="labels must have length 1"):
        cfg._validate("ashrae")


def test_bands_config_validates_color_length() -> None:
    cfg = BandsConfig(show=["90"], colors=["#ff0000", "#00ff00"])
    with pytest.raises(ValueError, match="colors must have length 1"):
        cfg._validate("ashrae")


def test_bands_config_validates_color_values() -> None:
    cfg = BandsConfig(colors=["not-a-color", "#ff0000"])
    with pytest.raises(ValueError, match="Invalid color"):
        cfg._validate("ashrae")


def test_bands_config_validates_all_bands_label_length() -> None:
    # ASHRAE has 2 bands; providing 3 labels without show should fail
    cfg = BandsConfig(labels=["A", "B", "C"])
    with pytest.raises(ValueError, match="labels must have length 2"):
        cfg._validate("ashrae")


def test_bands_config_valid_en() -> None:
    cfg = BandsConfig(
        show=["cat_i", "cat_ii"],
        labels=["Best", "OK"],
        colors=["#00ff00", "#ffff00"],
    )
    cfg._validate("en")
    assert cfg._validated


# set_bands tests


def test_set_bands_raw_params() -> None:
    plot = AdaptivePlot("ashrae")
    plot.set_bands(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
    assert plot._bands_config is not None
    assert plot._bands_config.show == ["90"]


def test_set_bands_with_config() -> None:
    config = BandsConfig(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
    plot = AdaptivePlot("ashrae")
    plot.set_bands(show=config)
    assert plot._bands_config is config


def test_set_bands_config_rejects_separate_labels() -> None:
    config = BandsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot("ashrae").set_bands(show=config, labels=["X"])


def test_set_bands_config_rejects_separate_colors() -> None:
    config = BandsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot("ashrae").set_bands(show=config, colors=["#ff0000"])


def test_set_bands_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot("ashrae").set_bands(show=["cat_i"])


def test_set_bands_chaining() -> None:
    plot = (
        AdaptivePlot("ashrae").set_params(tdb=25, tr=25, v=0.1).set_bands(show=["90"])
    )
    assert plot._bands_config is not None


# ASHRAE plot tests


def test_ashrae_plot_smoke(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(title="ASHRAE Test")
    assert isinstance(result, AdaptivePlotResult)
    assert result.ax.get_title() == "ASHRAE Test"
    assert len(result.fills) == 2  # 80% and 90%
    assert result.center_line is not None
    assert result.legend is not None


def test_ashrae_plot_default_labels(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "90% Acceptability" in legend_labels
    assert "80% Acceptability" in legend_labels
    assert "Comfort Temperature" in legend_labels


def test_ashrae_plot_custom_labels(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot.set_bands(labels=["Wide", "Narrow"])
    result = plot.plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Wide" in legend_labels
    assert "Narrow" in legend_labels


def test_ashrae_plot_show_only_90(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot.set_bands(show=["90"])
    result = plot.plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "90% Acceptability" in legend_labels
    assert "80% Acceptability" not in legend_labels


def test_ashrae_plot_custom_colors(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot.set_bands(colors=["#FF0000", "#00FF00"])
    result = plot.plot()
    assert len(result.fills) == 2


def test_ashrae_plot_no_center_line(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(show_center_line=False)
    assert result.center_line is None


def test_ashrae_plot_no_legend(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(legend=False)
    assert result.legend is None


def test_ashrae_plot_no_grid(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(grid=False)
    # Grid lines should not be visible
    assert not result.ax.xaxis.get_gridlines()[0].get_visible()


def test_ashrae_plot_with_grid(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(grid=True)
    assert result.ax.xaxis.get_gridlines()[0].get_visible()


def test_ashrae_plot_xlim(monkeypatch) -> None:
    plot = AdaptivePlot("ashrae", t_running_mean_range=(15, 30))
    monkeypatch.setattr(plot, "_load_model", lambda: _mock_ashrae)
    plot.set_params(tdb=25, tr=25, v=0.1)
    result = plot.plot()
    assert result.ax.get_xlim() == pytest.approx((15.0, 30.0))


def test_ashrae_plot_uses_provided_axis(monkeypatch) -> None:
    fig, ax = plt.subplots()
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(ax=ax)
    assert result.ax is ax
    assert result.fig is fig


def test_ashrae_plot_xlabel_ylabel(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(xlabel="X Label", ylabel="Y Label")
    assert result.ax.get_xlabel() == "X Label"
    assert result.ax.get_ylabel() == "Y Label"


def test_ashrae_plot_no_xlabel_ylabel(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(xlabel=None, ylabel=None)
    assert result.ax.get_xlabel() == ""
    assert result.ax.get_ylabel() == ""


def test_ashrae_plot_center_line_kws(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(center_line_kws={"color": "red", "linewidth": 3.0})
    assert result.center_line is not None
    assert result.center_line.get_color() == "red"
    assert result.center_line.get_linewidth() == 3.0


def test_ashrae_plot_with_bands_config(monkeypatch) -> None:
    config = BandsConfig(
        show=["90"],
        labels=["Narrow Zone"],
        colors=["#FF6B6B"],
    )
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot.set_bands(show=config)
    result = plot.plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Narrow Zone" in legend_labels


def test_ashrae_plot_handles_nan_gracefully(monkeypatch) -> None:
    plot = AdaptivePlot("ashrae")
    monkeypatch.setattr(plot, "_load_model", lambda: _mock_ashrae_with_nan)
    plot.set_params(tdb=25, tr=25, v=0.1)
    result = plot.plot()
    assert isinstance(result, AdaptivePlotResult)
    assert len(result.fills) > 0


# EN plot tests


def test_en_plot_smoke(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    result = plot.plot(title="EN Test")
    assert isinstance(result, AdaptivePlotResult)
    assert result.ax.get_title() == "EN Test"
    assert len(result.fills) == 3  # Cat I, II, III
    assert result.center_line is not None
    assert result.legend is not None


def test_en_plot_default_labels(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    result = plot.plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in legend_labels
    assert "Category II" in legend_labels
    assert "Category III" in legend_labels


def test_en_plot_show_only_cat_i(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    plot.set_bands(show=["cat_i"])
    result = plot.plot()
    assert len(result.fills) == 1
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in legend_labels
    assert "Category II" not in legend_labels


def test_en_plot_custom_labels(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    plot.set_bands(labels=["Best", "OK", "Min"])
    result = plot.plot()
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Best" in legend_labels
    assert "OK" in legend_labels
    assert "Min" in legend_labels


def test_en_plot_custom_colors(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    plot.set_bands(colors=["#AA0000", "#00AA00", "#0000AA"])
    result = plot.plot()
    assert len(result.fills) == 3


def test_en_plot_show_two_bands(monkeypatch) -> None:
    plot = _new_en_plot(monkeypatch, _mock_en)
    plot.set_bands(
        show=["cat_i", "cat_ii"],
        labels=["Strict", "Normal"],
        colors=["#00FF00", "#FFFF00"],
    )
    result = plot.plot()
    assert len(result.fills) == 2
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Strict" in legend_labels
    assert "Normal" in legend_labels


def test_en_rejects_ashrae_band_keys() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot("en").set_bands(show=["80"])


# BandsConfig reuse tests


def test_bands_config_reuse_across_plots(monkeypatch) -> None:
    config = BandsConfig(show=["90"], labels=["Comfort"], colors=["#FF6B6B"])

    plot1 = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot1.set_bands(show=config)
    result1 = plot1.plot()

    plot2 = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    plot2.set_bands(show=config)
    result2 = plot2.plot()

    labels1 = [t.get_text() for t in result1.legend.get_texts()]
    labels2 = [t.get_text() for t in result2.legend.get_texts()]
    assert labels1 == labels2
    assert "Comfort" in labels1


# Edge cases


def test_plot_without_set_bands_uses_defaults(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot()
    assert len(result.fills) == 2
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert "80% Acceptability" in legend_labels
    assert "90% Acceptability" in legend_labels


def test_plot_legend_kws(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(legend_kws={"loc": "upper left"})
    assert result.legend is not None


def test_plot_fill_kws(monkeypatch) -> None:
    plot = _new_ashrae_plot(monkeypatch, _mock_ashrae)
    result = plot.plot(fill_kws={"alpha": 0.3})
    assert len(result.fills) > 0
