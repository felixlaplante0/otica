import numpy as np
from scipy.stats import t


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
    condition_number: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates observations from a linear independent component model.

    Args:
        n (int): Number of observations.
        d (int): Number of sources and observed variables.
        distribution (str): Source distribution name.
        condition_number (float | None, optional): Mixing-matrix condition number.
            A Gaussian mixing matrix is used when this is `None`. Defaults to `None`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and mixing matrix.
    """
    mixing = np.random.normal(size=(d, d))
    if condition_number is not None:
        left, _, right = np.linalg.svd(mixing)
        mixing = left @ np.diag(np.linspace(1.0, condition_number, d)) @ right

    return _gen_sources(n, d, distribution) @ mixing.T, mixing


def gen_t(
    n: int,
    d: int,
    dfs: np.typing.ArrayLike,
    condition_number: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates ICA observations with heterogeneously scaled Student-t sources.

    Args:
        n (int): Number of observations.
        d (int): Number of sources and observed variables.
        dfs (np.typing.ArrayLike): Degrees of freedom for the sources.
        condition_number (float): Mixing-matrix condition number.

    Returns:
        tuple[np.ndarray, np.ndarray]: Observations and mixing matrix.
    """
    dfs = np.asarray(dfs)
    scales = np.random.uniform(0.5, 2.0, size=d)
    noise = (
        np.column_stack([t.rvs(df, size=n) / np.sqrt(df / (df - 2)) for df in dfs])
        * scales
    )
    mixing = np.random.normal(size=(d, d))
    left, _, right = np.linalg.svd(mixing)
    mixing = left @ np.diag(np.linspace(1.0, condition_number, d)) @ right

    return noise @ mixing.T, mixing


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
