"""
Data preprocessor for qutrit QNN training.

Provides scaling, normalisation, and feature-engineering utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class FitStatistics:
    """Stores fitted statistics for inverse transforms."""
    mean: Optional[NDArray] = None
    std: Optional[NDArray] = None
    min_val: Optional[NDArray] = None
    max_val: Optional[NDArray] = None
    scale_range: Tuple[float, float] = (0.0, 1.0)


class Preprocessor:
    """
    Feature preprocessor for classical data before quantum encoding.

    Supports:
    - StandardScaler: zero mean, unit variance.
    - MinMaxScaler: scale to [a, b] (default [0, π]).
    - Normalise: L2-normalise each sample.
    - PCA: principal component analysis for dimensionality reduction.
    - One-hot encoding for integer labels → qutrit probability targets.

    Parameters
    ----------
    method : str
        'standard', 'minmax', 'normalize', or 'none'.
    feature_range : tuple
        (min, max) for minmax scaling. Default (0, π).
    """

    def __init__(
        self,
        method: str = "minmax",
        feature_range: Tuple[float, float] = (0.0, np.pi),
    ):
        self.method = method
        self.feature_range = feature_range
        self._stats = FitStatistics()
        self._fitted = False

    def fit(self, x: NDArray[np.floating]) -> "Preprocessor":
        """
        Fit the preprocessor to training data.

        Parameters
        ----------
        x : ndarray (n_samples, n_features)

        Returns
        -------
        self
        """
        x = np.asarray(x, dtype=np.float64)

        if self.method == "standard":
            self._stats.mean = np.mean(x, axis=0)
            self._stats.std = np.std(x, axis=0)
            self._stats.std[self._stats.std < 1e-12] = 1.0

        elif self.method == "minmax":
            self._stats.min_val = np.min(x, axis=0)
            self._stats.max_val = np.max(x, axis=0)
            rng = self._stats.max_val - self._stats.min_val
            rng[rng < 1e-12] = 1.0
            self._stats.max_val = self._stats.min_val + rng
            self._stats.scale_range = self.feature_range

        self._fitted = True
        return self

    def transform(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Transform features using fitted statistics.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """
        x = np.asarray(x, dtype=np.float64)

        if self.method == "none":
            return x

        if self.method == "normalize":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            return x / norms

        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        if self.method == "standard":
            return (x - self._stats.mean) / self._stats.std

        if self.method == "minmax":
            a, b = self._stats.scale_range
            x_std = (x - self._stats.min_val) / (
                self._stats.max_val - self._stats.min_val
            )
            return x_std * (b - a) + a

        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """Fit and transform in one step."""
        return self.fit(x).transform(x)

    def inverse_transform(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Inverse-transform processed features back to original scale.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        ndarray
        """
        x = np.asarray(x, dtype=np.float64)

        if self.method == "none" or self.method == "normalize":
            return x

        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")

        if self.method == "standard":
            return x * self._stats.std + self._stats.mean

        if self.method == "minmax":
            a, b = self._stats.scale_range
            x_std = (x - a) / (b - a)
            return x_std * (self._stats.max_val - self._stats.min_val) + self._stats.min_val

        raise ValueError(f"Unknown method: {self.method}")

    @staticmethod
    def one_hot(
        labels: NDArray[np.integer], num_classes: int = 3
    ) -> NDArray[np.float64]:
        """
        One-hot encode integer labels.

        Parameters
        ----------
        labels : ndarray of int
        num_classes : int

        Returns
        -------
        ndarray of shape (n_samples, num_classes)
        """
        labels = np.asarray(labels, dtype=int)
        n = len(labels)
        oh = np.zeros((n, num_classes), dtype=np.float64)
        oh[np.arange(n), labels] = 1.0
        return oh

    @staticmethod
    def ternary_labels(labels: NDArray[np.integer]) -> NDArray[np.integer]:
        """
        Convert labels to ternary (base-3) for qutrit systems.

        If labels have more than 3 classes, maps them modulo 3.
        """
        return np.asarray(labels, dtype=int) % 3

    @staticmethod
    def pca_reduce(
        x: NDArray[np.floating],
        n_components: int,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Simple PCA dimensionality reduction.

        Parameters
        ----------
        x : ndarray (n_samples, n_features)
        n_components : int

        Returns
        -------
        x_reduced : ndarray (n_samples, n_components)
        components : ndarray (n_components, n_features) — principal axes
        """
        x = np.asarray(x, dtype=np.float64)
        x_centered = x - np.mean(x, axis=0)
        cov = np.cov(x_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        components = eigenvectors[:, idx[:n_components]].T
        x_reduced = x_centered @ components.T

        return x_reduced, components

    def __repr__(self) -> str:
        return f"Preprocessor(method={self.method!r}, fitted={self._fitted})"
