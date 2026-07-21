import numpy as np


def _gen_quadrature(order: int, distribution: str) -> tuple[np.ndarray, np.ndarray]:
    """Generates quadrature nodes and weights for a source distribution.

    Args:
        order (int): Quadrature order.
        distribution (str): Source distribution name.

    Returns:
        tuple[np.ndarray, np.ndarray]: Quadrature nodes and weights.
    """
    if distribution == "Gaussian":
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)
    if distribution == "Uniform":
        nodes, weights = np.polynomial.legendre.leggauss(order)
        return np.sqrt(3.0) * nodes, weights / 2.0

    nodes, weights = np.polynomial.laguerre.laggauss(order)
    if distribution == "Laplace":
        return (
            np.concatenate((-nodes, nodes)) / np.sqrt(2.0),
            np.tile(weights, 2) / 2.0,
        )
    if distribution == "Exponential":
        return nodes - 1.0, weights

    uniform_nodes, uniform_weights = _gen_quadrature(order, "Uniform")
    mixture_weight = 0.6239786746633258
    return (
        np.concatenate((uniform_nodes, nodes - 1.0)),
        np.concatenate(
            (mixture_weight * uniform_weights, (1.0 - mixture_weight) * weights)
        ),
    )


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

    sources = np.random.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=(n, d))
    exponential = np.random.random(size=(n, d)) >= 0.6239786746633258
    sources[exponential] = np.random.exponential(size=exponential.sum()) - 1.0

    return sources


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
