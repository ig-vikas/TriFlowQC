"""
Adam optimiser adapted for quantum circuit parameters.
"""

from __future__ import annotations

import numpy as np
from typing import List

from triflowqc.optim.base_optimizer import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam optimiser (Kingma & Ba, 2015).

    Parameters
    ----------
    lr : float
        Learning rate (default 0.01).
    beta1 : float
        Exponential decay rate for first moment (default 0.9).
    beta2 : float
        Exponential decay rate for second moment (default 0.999).
    epsilon : float
        Numerical stability constant (default 1e-8).
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._m: np.ndarray = np.array([])
        self._v: np.ndarray = np.array([])
        self._t: int = 0

    def step(self, parameters: List[float], gradients: np.ndarray) -> List[float]:
        params = np.asarray(parameters, dtype=np.float64)
        grads = np.asarray(gradients, dtype=np.float64)

        if self._m.shape != params.shape:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grads
        self._v = self.beta2 * self._v + (1 - self.beta2) * grads ** 2

        m_hat = self._m / (1 - self.beta1 ** self._t)
        v_hat = self._v / (1 - self.beta2 ** self._t)

        params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params.tolist()

    def reset(self) -> None:
        self._m = np.array([])
        self._v = np.array([])
        self._t = 0
