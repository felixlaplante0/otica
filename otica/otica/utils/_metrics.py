import numpy as np
from sklearn.utils.validation import check_array  # type: ignore


def amari_index(
    mixing_true: np.typing.ArrayLike,
    unmixing_pred: np.typing.ArrayLike,
) -> float:
    r"""Computes the Amari index between true mixing and estimated unmixing matrices.

    The index is zero when the estimated unmixing matrix recovers the sources up to the
    permutation, scaling, and sign indeterminacies of independent component analysis.

    Args:
        mixing_true (np.typing.ArrayLike): True square mixing matrix.
        unmixing_pred (np.typing.ArrayLike): Estimated square unmixing matrix.

    Returns:
        float: Amari index.

    Raises:
        ValueError: If either matrix is not square, their shapes differ, or the
            unmixing-mixing product has a zero row or column.
    """
    mixing_true_array = check_array(mixing_true)
    unmixing_pred_array = check_array(unmixing_pred)

    if mixing_true_array.shape[0] != mixing_true_array.shape[1]:
        raise ValueError(
            f"mixing_true must be a square array, got shape {mixing_true_array.shape}."
        )
    if unmixing_pred_array.shape[0] != unmixing_pred_array.shape[1]:
        raise ValueError(
            "unmixing_pred must be a square array, "
            f"got shape {unmixing_pred_array.shape}."
        )
    if mixing_true_array.shape != unmixing_pred_array.shape:
        raise ValueError(
            "mixing_true and unmixing_pred must have the same shape, "
            f"got {mixing_true_array.shape} and {unmixing_pred_array.shape}."
        )

    product = np.abs(unmixing_pred_array @ mixing_true_array)
    row_maxima = product.max(axis=1, keepdims=True)
    column_maxima = product.max(axis=0, keepdims=True)
    if np.any(row_maxima == 0.0) or np.any(column_maxima == 0.0):
        raise ValueError(
            "unmixing_pred @ mixing_true must not contain an all-zero row or column."
        )

    d = product.shape[0]
    if d == 1:
        return 0.0

    return float(
        (np.sum(product / row_maxima) + np.sum(product / column_maxima) - 2 * d)
        / (2 * d * (d - 1))
    )
