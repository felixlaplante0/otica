"""Tests for otica utilities."""

import numpy as np
import pytest

from otica.utils import amari_index


@pytest.mark.parametrize(
    ("mixing_true", "unmixing_pred", "expected"),
    [
        (
            np.array([[1.0, 0.4], [-0.3, 1.0]]),
            np.array([[0.0, -2.0], [3.0, 0.0]])
            @ np.linalg.inv(np.array([[1.0, 0.4], [-0.3, 1.0]])),
            0.0,
        ),
        (np.eye(2), np.array([[1.0, 0.5], [0.25, 1.0]]), 0.375),
        (np.array([[0.5]]), np.array([[2.0]]), 0.0),
    ],
)
def test_amari(mixing_true, unmixing_pred, expected):
    """Computes the Amari index."""
    assert amari_index(mixing_true, unmixing_pred) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("mixing_true", "unmixing_pred", "message"),
    [
        (np.ones((2, 3)), np.eye(2), "mixing_true must be a square array"),
        (np.eye(2), np.ones((2, 3)), "unmixing_pred must be a square array"),
        (np.eye(2), np.eye(3), "must have the same shape"),
        (np.zeros((2, 2)), np.eye(2), "all-zero row or column"),
    ],
)
def test_amari_validation(mixing_true, unmixing_pred, message):
    """Rejects matrices that do not define a valid Amari index."""
    with pytest.raises(ValueError, match=message):
        amari_index(mixing_true, unmixing_pred)
