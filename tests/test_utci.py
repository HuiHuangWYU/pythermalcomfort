import numpy as np

from pythermalcomfort.models import utci
from pythermalcomfort.models.utci import _utci_optimized
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_utci(get_test_url, retrieve_data) -> None:
    """Test that the UTCI function calculates correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.UTCI.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = utci(**inputs)

        validate_result(result, outputs, tolerance)


def test_utci_optimized() -> None:
    """Test that the optimized UTCI function calculates correctly for various inputs."""
    np.testing.assert_equal(
        np.around(_utci_optimized([25, 27], 1, 1, 1.5), 2),
        [24.73, 26.57],
    )


def test_utci_ip_uses_si_thresholds_for_stress_category() -> None:
    """Test that IP stress categories use the underlying SI UTCI value."""
    result = utci(tdb=77, tr=77, v=3.28084, rh=50, units="IP")

    assert result.utci == 76.4
    assert result.stress_category == "no thermal stress"


def test_utci_ip_vector_stress_category() -> None:
    """Test that IP vector stress categories use SI UTCI values."""
    result = utci(
        tdb=[77, 104],
        tr=[77, 104],
        v=[3.28084, 3.28084],
        rh=[50, 50],
        units="IP",
    )

    np.testing.assert_allclose(result.utci, [76.4, 110.7])
    np.testing.assert_array_equal(
        result.stress_category,
        ["no thermal stress", "very strong heat stress"],
    )


def test_utci_stress_category_uses_rounded_si_value_by_default() -> None:
    """Test that default SI category mapping preserves rounded-output behavior."""
    rounded = utci(tdb=26.24, tr=26.24, v=1, rh=50)
    unrounded = utci(tdb=26.24, tr=26.24, v=1, rh=50, round_output=False)

    assert rounded.utci == 26.0
    assert rounded.stress_category == "no thermal stress"
    assert unrounded.utci > 26.0
    assert unrounded.stress_category == "moderate heat stress"


def test_utci_ip_stress_category_uses_unrounded_si_value_when_not_rounding() -> None:
    """Test that unrounded IP output maps categories from unrounded SI values."""
    result = utci(
        tdb=79.232,
        tr=79.232,
        v=3.28084,
        rh=50,
        units="IP",
        round_output=False,
    )

    assert result.utci > 78.8
    assert result.stress_category == "moderate heat stress"


def test_utci_ip_out_of_range_stress_category_is_nan() -> None:
    """Test that out-of-range IP inputs produce NaN stress categories."""
    result = utci(tdb=1000, tr=1000, v=3.28084, rh=50, units="IP")

    assert np.isnan(result.utci).item()
    assert np.isnan(result.stress_category).item()

    vector_result = utci(
        tdb=[77, 1000],
        tr=[77, 1000],
        v=[3.28084, 3.28084],
        rh=[50, 50],
        units="IP",
    )

    np.testing.assert_allclose(vector_result.utci, [76.4, np.nan], equal_nan=True)
    assert vector_result.stress_category[0] == "no thermal stress"
    assert np.isnan(vector_result.stress_category[1])
