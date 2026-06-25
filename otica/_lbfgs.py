import numpy as np
import torch
from torch.nn.utils import parametrizations


class LBFGSMixin:
    """Provides a torch-based orthogonal L-BFGS solver."""

    lr: float
    max_iter: int
    history_size: int
    tolerance_grad: float
    tolerance_change: float
    orthogonal_map: str

    def _solve(
        self,
        X: np.ndarray,
        quantiles: torch.Tensor,
        init_unmixing_: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Maximizes Wasserstein non-Gaussianity with torch L-BFGS.

        Args:
            X (np.ndarray): Whitened observations arranged by feature.
            quantiles (torch.Tensor): Reference Gaussian quantiles for each sample.
            init_unmixing_ (np.ndarray): Initial orthogonal unmixing matrix.

        Returns:
            tuple[np.ndarray, int, float]: Final orthogonal unmixing matrix, number of
            iterations performed, and final objective value.
        """
        X = torch.as_tensor(X, dtype=torch.float64)
        n_components = X.shape[1]

        module = torch.nn.Linear(
            n_components,
            n_components,
            bias=False,
            dtype=torch.float64,
        )
        module.weight.data = torch.as_tensor(init_unmixing_, dtype=torch.float64)
        parametrizations.orthogonal(
            module,
            "weight",
            orthogonal_map=self.orthogonal_map,
        )

        optimizer = torch.optim.LBFGS(
            module.parameters(),
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            sources = X @ module.weight.T
            differences = torch.sort(sources, dim=0).values - quantiles[:, None]
            loss = -differences.square().mean(dim=0).sum()
            loss.backward()
            
            return loss

        optimizer.step(closure)

        state = optimizer.state[optimizer.param_groups[0]["params"][0]]
        n_iter = int(state.get("n_iter", self.max_iter))
        with torch.no_grad():
            sources = X @ module.weight.T
            differences = torch.sort(sources, dim=0).values - quantiles[:, None]
            objective = float(differences.square().mean(dim=0).sum().item())

        return module.weight.detach().cpu().numpy(), n_iter, objective
