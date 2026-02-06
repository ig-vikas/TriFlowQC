"""
Metrics for qutrit quantum neural networks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Accuracy:
    """Classification accuracy metric."""

    name = "accuracy"

    def __call__(self, predictions: NDArray[np.floating], targets: NDArray) -> float:
        """
        Parameters
        ----------
        predictions : ndarray, shape (n_samples, n_classes)
        targets : ndarray, shape (n_samples,) or (n_samples, n_classes)

        Returns
        -------
        float
            Fraction of correct predictions.
        """
        predictions = np.atleast_2d(predictions)
        pred_labels = np.argmax(predictions, axis=1)
        targets_arr = np.asarray(targets)
        if targets_arr.ndim > 1:
            target_labels = np.argmax(targets_arr, axis=1)
        else:
            target_labels = targets_arr.astype(np.int64)
        return float(np.mean(pred_labels == target_labels))


class FidelityMetric:
    """
    Classical fidelity between predicted and target distributions.

    F = mean over samples of (Σ_i √(p_i q_i))²
    """

    name = "fidelity"

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

        fid = np.sum(np.sqrt(predictions * targets_oh + 1e-15), axis=1) ** 2
        return float(np.mean(fid))
