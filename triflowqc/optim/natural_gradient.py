"""
Quantum natural gradient optimiser.

Uses the Fubini-Study metric tensor to precondition gradients.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, List

from triflowqc.optim.base_optimizer import BaseOptimizer


class NaturalGradient(BaseOptimizer):
    """
    Quantum Natural Gradient optimiser.

    Computes the metric tensor g_{ij} from parameter-shift overlaps
    and updates θ → θ − lr · g⁻¹ · ∇L.

    Parameters
    ----------
    lr : float
        Learning rate.
    regularisation : float
        Tikhonov regularisation λ added to diagonal of metric tensor.
    """

    def __init__(self, lr: float = 0.01, regularisation: float = 1e-3) -> None:
        super().__init__(lr)
        self.regularisation = regularisation
        self._metric: NDArray[np.floating] = np.array([])

    def step(self, parameters: List[float], gradients: np.ndarray) -> List[float]:
        """
        Natural gradient step: θ ← θ − lr · g⁻¹ · grad.

        If metric tensor is not set, falls back to vanilla gradient descent.
        """
        params = np.asarray(parameters, dtype=np.float64)
        grads = np.asarray(gradients, dtype=np.float64)

        if self._metric.shape == (len(params), len(params)):
            g_reg = self._metric + self.regularisation * np.eye(len(params))
            try:
                g_inv_grad = np.linalg.solve(g_reg, grads)
            except np.linalg.LinAlgError:
                g_inv_grad = grads
        else:
            g_inv_grad = grads

        params = params - self.lr * g_inv_grad
        return params.tolist()

    def compute_metric_tensor(
        self, model: Any, parameters: List[float], X_sample: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Estimate the Fubini-Study metric tensor via parameter shifts.

        g_{ij} = Re[⟨∂_i ψ|∂_j ψ⟩ − ⟨∂_i ψ|ψ⟩⟨ψ|∂_j ψ⟩]

        Uses the parameter-shift-based overlap formula.

        Parameters
        ----------
        model : Sequential or similar
        parameters : list of float
        X_sample : ndarray
            A single input sample.

        Returns
        -------
        ndarray, shape (n_params, n_params)
        """
        n = len(parameters)
        shift = 2.0 * np.pi / 3.0

        # Get reference prediction
        model.all_parameters = list(parameters)
        ref = model.predict(X_sample.reshape(1, -1), shots=0)[0]

        g = np.zeros((n, n), dtype=np.float64)
        shifted_preds = {}

        for k in range(n):
            # + shift
            p_plus = list(parameters)
            p_plus[k] += shift
            model.all_parameters = p_plus
            shifted_preds[(k, "+")] = model.predict(X_sample.reshape(1, -1), shots=0)[0]

            # − shift
            p_minus = list(parameters)
            p_minus[k] -= shift
            model.all_parameters = p_minus
            shifted_preds[(k, "-")] = model.predict(X_sample.reshape(1, -1), shots=0)[0]

        for i in range(n):
            for j in range(i, n):
                # Approximate metric element from overlaps of shifted distributions
                dp_i = (shifted_preds[(i, "+")] - shifted_preds[(i, "-")]) / (2 * np.sin(shift))
                dp_j = (shifted_preds[(j, "+")] - shifted_preds[(j, "-")]) / (2 * np.sin(shift))
                g[i, j] = np.dot(dp_i, dp_j)
                g[j, i] = g[i, j]

        model.all_parameters = list(parameters)
        self._metric = g
        return g

    def reset(self) -> None:
        self._metric = np.array([])
