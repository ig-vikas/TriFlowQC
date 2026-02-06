"""
Hybrid classical + quantum model.

Wraps a classical pre-processing head (numpy) and a quantum Sequential model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Sequence, Tuple

from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.layers.base_layer import BaseQutritLayer


class HybridModel(Sequential):
    """
    Hybrid classical-quantum model.

    Adds a classical pre-processing step before the quantum layers.
    The classical head is a simple linear layer W @ x + b that maps
    the input features to the number of qutrits.

    Parameters
    ----------
    input_dim : int
        Classical input dimension.
    layers : list of BaseQutritLayer
        Quantum layers (first should be an embedding layer).
    seed : int or None
        Random seed for weight initialisation.
    """

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[BaseQutritLayer],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(layers)
        nq = self.num_qutrits
        rng = np.random.default_rng(seed)
        self._classical_W = rng.standard_normal((nq, input_dim)) * 0.1
        self._classical_b = np.zeros(nq)
        self._input_dim = input_dim

    def _preprocess(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Classical linear transformation."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return self._classical_W @ x + self._classical_b

    def _forward_single(
        self, x: NDArray[np.floating], shots: int = 0
    ) -> NDArray[np.floating]:
        z = self._preprocess(x)
        return super()._forward_single(z, shots=shots)

    @property
    def num_parameters(self) -> int:
        classical = self._classical_W.size + self._classical_b.size
        return classical + super().num_parameters

    @property
    def all_parameters(self) -> List[float]:
        params = list(self._classical_W.ravel()) + list(self._classical_b)
        params.extend(super().all_parameters)
        return params

    @all_parameters.setter
    def all_parameters(self, values: List[float]) -> None:
        nq = self.num_qutrits
        n_W = self._classical_W.size
        n_b = self._classical_b.size
        self._classical_W = np.array(values[:n_W]).reshape(self._classical_W.shape)
        self._classical_b = np.array(values[n_W:n_W + n_b])
        # Set quantum parameters
        idx = n_W + n_b
        for layer in self._layers:
            n = layer.num_parameters
            layer.parameters = values[idx: idx + n]
            idx += n

    def __repr__(self) -> str:
        return (
            f"HybridModel(input_dim={self._input_dim}, "
            f"layers={len(self._layers)}, params={self.num_parameters})"
        )
