"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimiser.

Estimates gradients using only 2 function evaluations regardless of
the number of parameters.
"""

from __future__ import annotations

import numpy as np
from typing import List

from triflowqc.optim.base_optimizer import BaseOptimizer


class SPSA(BaseOptimizer):
    """
    SPSA optimiser.

    Parameters
    ----------
    lr : float
        Learning rate / step size.
    perturbation : float
        Perturbation magnitude for gradient estimation.
    """

    def __init__(self, lr: float = 0.01, perturbation: float = 0.1) -> None:
        super().__init__(lr)
        self.perturbation = perturbation
        self._k: int = 0

    def step(self, parameters: List[float], gradients: np.ndarray) -> List[float]:
        """Standard step using externally provided gradients."""
        params = np.asarray(parameters, dtype=np.float64)
        grads = np.asarray(gradients, dtype=np.float64)
        self._k += 1
        params = params - self.lr * grads
        return params.tolist()

    def estimate_gradient(
        self,
        fn: "Callable[[List[float]], float]",
        parameters: List[float],
    ) -> np.ndarray:
        """
        Estimate gradient using SPSA (2 evaluations).

        Parameters
        ----------
        fn : callable
            Function mapping parameters → scalar loss.
        parameters : list of float
            Current parameter values.

        Returns
        -------
        ndarray
            Estimated gradient.
        """
        self._k += 1
        params = np.asarray(parameters, dtype=np.float64)
        n = len(params)

        # Bernoulli perturbation
        delta = np.random.choice([-1.0, 1.0], size=n)
        c_k = self.perturbation / (self._k ** 0.101)

        params_plus = params + c_k * delta
        params_minus = params - c_k * delta

        f_plus = fn(params_plus.tolist())
        f_minus = fn(params_minus.tolist())

        grad = (f_plus - f_minus) / (2.0 * c_k * delta)
        return grad

    def reset(self) -> None:
        self._k = 0
