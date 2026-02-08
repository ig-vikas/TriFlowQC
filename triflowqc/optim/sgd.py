"""Stochastic Gradient Descent optimiser."""

from __future__ import annotations

import numpy as np
from typing import List

from triflowqc.optim.base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
    """
    SGD with optional momentum.

    Parameters
    ----------
    lr : float
        Learning rate.
    momentum : float
        Momentum factor.
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(lr)
        self.momentum = momentum
        self._velocity: np.ndarray = np.array([])

    def step(self, parameters: List[float], gradients: np.ndarray) -> List[float]:
        params = np.asarray(parameters, dtype=np.float64)
        grads = np.asarray(gradients, dtype=np.float64)

        if self._velocity.shape != params.shape:
            self._velocity = np.zeros_like(params)

        self._velocity = self.momentum * self._velocity - self.lr * grads
        params = params + self._velocity
        return params.tolist()

    def reset(self) -> None:
        self._velocity = np.array([])
