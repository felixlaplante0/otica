import numpy as np


def _gen_sources(n: int, d: int, distribution: str) -> np.ndarray:
    """Generates independent sources from a specified distribution.

    Args:
        n (int): Number of observations.
        d (int): Number of sources.
        distribution (str): Source distribution name.

    Returns:
        np.ndarray: Independent sources.
    """
    if distribution == "Laplace":
        return np.random.laplace(scale=1.0 / np.sqrt(2.0), size=(n, d))
    if distribution == "Uniform":
        return np.random.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n, d))
    if distribution == "Exponential":
        return np.random.exponential(size=(n, d))

    return np.random.choice(
        (-1.5540997179090625, 0.0, 1.5540997179090625),
        size=(n, d),
        p=(0.2070199699025653, 0.5859600601948694, 0.2070199699025653),
    )


def gen_data(
    n: int,
    d: int,
    distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates observations from a linear independent component model.

    Args:
        n (int): Number of observations.
        d (int): Number of sources and observed variables.
        distribution (str): Source distribution name.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and mixing matrix.
    """
    mixing = np.random.normal(size=(d, d))

    return _gen_sources(n, d, distribution) @ mixing.T, mixing


def gen_gaussianity(
    n: int,
    d: int,
    distribution: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates ICA observations with sources approaching Gaussianity.

    Args:
        n (int): Number of observations.
        d (int): Number of sources and observed variables.
        distribution (str): Non-Gaussian source distribution name.
        epsilon (float): Gaussian contribution between zero and one.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and mixing matrix.
    """
    non_gaussian = _gen_sources(n, d, distribution)
    if distribution == "Exponential":
        non_gaussian -= 1.0
    sources = np.sqrt(1.0 - epsilon) * non_gaussian + np.sqrt(
        epsilon
    ) * np.random.normal(size=(n, d))
    mixing = np.random.normal(size=(d, d))

    return sources @ mixing.T, mixing


def amari_index(unmixing: np.ndarray, mixing: np.ndarray) -> float:
    """Calculates the Amari index between unmixing and mixing matrices.

    Args:
        unmixing (np.ndarray): Estimated unmixing matrix.
        mixing (np.ndarray): True mixing matrix.

    Returns:
        float: Amari index, where zero indicates exact recovery up to ICA
            indeterminacies.
    """
    product = np.abs(unmixing @ mixing)
    d = product.shape[0]

    return float(
        (
            np.sum(product / product.max(axis=1, keepdims=True))
            + np.sum(product / product.max(axis=0, keepdims=True))
            - 2 * d
        )
        / (2 * d * (d - 1))
    )
