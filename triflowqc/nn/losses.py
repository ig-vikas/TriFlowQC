"""
Loss functions for qutrit quantum neural networks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class MSELoss:
    """Mean Squared Error loss."""

    name = "mse"

    def __call__(self, predictions: NDArray[np.floating], targets: NDArray) -> float:
        """
        Parameters
        ----------
        predictions : ndarray, shape (n_samples, n_classes) or (n_samples,)
        targets : ndarray, shape (n_samples,) or (n_samples, n_classes)
            Integer labels or one-hot encoded.
        """
        predictions = np.atleast_2d(predictions)
        targets_oh = self._to_onehot(targets, predictions.shape[-1])
        return float(np.mean((predictions - targets_oh) ** 2))

    @staticmethod
    def _to_onehot(targets: NDArray, n_classes: int) -> NDArray[np.floating]:
        targets = np.asarray(targets)
        if targets.ndim == 1:
            oh = np.zeros((targets.shape[0], n_classes), dtype=np.float64)
            for i, t in enumerate(targets):
                oh[i, int(t)] = 1.0
            return oh
        return targets.astype(np.float64)


class CrossEntropyLoss:
    """Cross-entropy loss for classification."""

    name = "cross_entropy"

    def __call__(self, predictions: NDArray[np.floating], targets: NDArray) -> float:
        """
        Parameters
        ----------
        predictions : ndarray, shape (n_samples, n_classes)
            Predicted probabilities.
        targets : ndarray, shape (n_samples,) integer labels or one-hot.
        """
        predictions = np.atleast_2d(predictions).astype(np.float64)
        eps = 1e-12
        predictions = np.clip(predictions, eps, 1.0 - eps)

        targets_arr = np.asarray(targets)
        if targets_arr.ndim == 1:
            # Integer labels
            n_samples = predictions.shape[0]
            loss = 0.0
            for i in range(n_samples):
                loss -= np.log(predictions[i, int(targets_arr[i])])
            return float(loss / n_samples)
        else:
            # One-hot
            targets_arr = targets_arr.astype(np.float64)
            return float(-np.mean(np.sum(targets_arr * np.log(predictions), axis=1)))

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"


class FidelityLoss:
    """
    Fidelity-based loss: 1 - F, where F is the average fidelity between
    predicted and target probability distributions.
    """

    name = "fidelity_loss"

    def __call__(self, predictions: NDArray[np.floating], targets: NDArray) -> float:
        predictions = np.atleast_2d(predictions).astype(np.float64)
        targets_arr = np.asarray(targets)
        if targets_arr.ndim == 1:
            n_classes = predictions.shape[-1]
            targets_oh = np.zeros_like(predictions)
            for i, t in enumerate(targets_arr):
                targets_oh[i, int(t)] = 1.0
        else:
            targets_oh = targets_arr.astype(np.float64)

        # Classical fidelity: F = (Σ √(p_i q_i))²
        fid = np.sum(np.sqrt(predictions * targets_oh + 1e-15), axis=1) ** 2
        return float(1.0 - np.mean(fid))
