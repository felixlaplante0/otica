import torch


def gauss_quantiles(n: int) -> torch.Tensor:
    """Computes equally weighted standard-normal quantiles.

    Args:
        n (int): Number of quantile bins.

    Returns:
        torch.Tensor: Mean standard-normal quantile in each bin.
    """
    z = torch.special.ndtri(torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64))
    phi = torch.exp(-0.5 * z.square()) / torch.sqrt(
        torch.tensor(2.0 * torch.pi, dtype=torch.float64)
    )
    phi[0] = 0.0
    phi[-1] = 0.0
    return n * (phi[:-1] - phi[1:])
