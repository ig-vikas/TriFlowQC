"""
Data loader and batch generator for qutrit QNN training.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple, Iterator
from dataclasses import dataclass


@dataclass
class Batch:
    """A single training batch."""
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    indices: NDArray[np.intp]


class DataLoader:
    """
    Mini-batch data loader with shuffling and train/val/test splits.

    Parameters
    ----------
    x : ndarray
        Input features of shape (n_samples, n_features).
    y : ndarray
        Labels of shape (n_samples,) or (n_samples, n_classes).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle at the start of each epoch.
    validation_split : float
        Fraction of data reserved for validation (0–1).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        batch_size: int = 32,
        shuffle: bool = True,
        validation_split: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        n = len(self.x)
        if validation_split > 0:
            n_val = max(1, int(n * validation_split))
            indices = self.rng.permutation(n)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            self.x_train = self.x[train_idx]
            self.y_train = self.y[train_idx]
            self.x_val = self.x[val_idx]
            self.y_val = self.y[val_idx]
        else:
            self.x_train = self.x
            self.y_train = self.y
            self.x_val = np.empty((0,) + self.x.shape[1:], dtype=np.float64)
            self.y_val = np.empty((0,) + self.y.shape[1:], dtype=np.float64)

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    @property
    def n_val(self) -> int:
        return len(self.x_val)

    @property
    def n_batches(self) -> int:
        return int(np.ceil(self.n_train / self.batch_size))

    def train_batches(self) -> Iterator[Batch]:
        """Iterate over training batches."""
        n = self.n_train
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            yield Batch(
                x=self.x_train[batch_idx],
                y=self.y_train[batch_idx],
                indices=batch_idx,
            )

    def val_batches(self) -> Iterator[Batch]:
        """Iterate over validation batches."""
        n = self.n_val
        if n == 0:
            return
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = np.arange(start, end)
            yield Batch(
                x=self.x_val[batch_idx],
                y=self.y_val[batch_idx],
                indices=batch_idx,
            )

    def all_data(self) -> Tuple[NDArray, NDArray]:
        """Return all data (no batching)."""
        return self.x, self.y

    @staticmethod
    def from_numpy(
        x_train: NDArray,
        y_train: NDArray,
        x_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> "DataLoader":
        """
        Create a DataLoader from pre-split numpy arrays.
        """
        loader = DataLoader(x_train, y_train, batch_size=batch_size, shuffle=shuffle)
        if x_val is not None and y_val is not None:
            loader.x_val = np.asarray(x_val, dtype=np.float64)
            loader.y_val = np.asarray(y_val, dtype=np.float64)
        return loader

    def __repr__(self) -> str:
        return (
            f"DataLoader(n_train={self.n_train}, n_val={self.n_val}, "
            f"batch_size={self.batch_size})"
        )

    def __len__(self) -> int:
        return self.n_batches
