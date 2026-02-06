"""
Hybrid classical-quantum layer.

Applies a classical linear transformation to the input before encoding
it into the quantum circuit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class HybridLayer(BaseQutritLayer):
    """
    Classical linear → quantum layer.

    Applies W @ x + b (classical) then encodes the result via angle embedding.

    Parameters
    ----------
    input_dim : int
        Classical input dimension.
    num_qutrits : int
        Number of output qutrits.
    gm_index : int
        Gell-Mann rotation axis for encoding.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        input_dim: int,
        num_qutrits: int,
        gm_index: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_qutrits, name="HybridLayer")
        self._input_dim = input_dim
        self._gm_index = gm_index
        rng = np.random.default_rng(seed)
        # Classical parameters: weight matrix + bias
        self._W = rng.standard_normal((num_qutrits, input_dim)) * 0.1
        self._b = np.zeros(num_qutrits)
        # Store as flat list
        self._params = list(self._W.ravel()) + list(self._b)

    @property
    def num_parameters(self) -> int:
        return self._num_qutrits * self._input_dim + self._num_qutrits

    def _sync_matrices(self) -> None:
        """Rebuild W and b from flat param list."""
        n = self._num_qutrits * self._input_dim
        self._W = np.array(self._params[:n]).reshape(self._num_qutrits, self._input_dim)
        self._b = np.array(self._params[n:n + self._num_qutrits])

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        if x is None:
            raise ValueError("HybridLayer requires input x.")
        self._sync_matrices()
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] != self._input_dim:
            raise ValueError(
                f"Expected input dim {self._input_dim}, got {x.shape[0]}"
            )
        # Classical forward pass
        z = self._W @ x + self._b
        # Encode via rotations
        for i in range(self._num_qutrits):
            circuit.r_gm(self._gm_index, float(z[i]), i)
        return circuit
