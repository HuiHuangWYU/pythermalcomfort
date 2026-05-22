import warnings

import numpy as np
import pytest

from pythermalcomfort.models.vertical_tmp_grad_ppd import vertical_tmp_grad_ppd
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_vertical_tmp_grad_ppd(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the output correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.VERTICAL_TMP_GRAD_PPD.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = vertical_tmp_grad_ppd(**inputs)

        validate_result(result, outputs, tolerance)

    # Test for ValueError
    np.isclose(
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7).ppd_vg,
        np.nan,
        equal_nan=True,
    )


def test_vertical_tmp_grad_ppd_limit_inputs_false_suppresses_warning_and_nan() -> None:
    """limit_inputs=False bypasses range checks: no UserWarning, no NaN mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = vertical_tmp_grad_ppd(
            50,
            25,
            0.1,
            50,
            1.2,
            0.5,
            7,
            limit_inputs=False,
        )
    assert not np.isnan(result.ppd_vg)


def test_vertical_tmp_grad_ppd_limit_inputs_true_still_warns_and_nans() -> None:
    """limit_inputs=True (default) keeps current behavior: warning + NaN mask."""
    with pytest.warns(UserWarning):
        result = vertical_tmp_grad_ppd(50, 25, 0.1, 50, 1.2, 0.5, 7)
    assert np.isnan(result.ppd_vg)
