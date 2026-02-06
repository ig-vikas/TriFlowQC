"""
Quantum pooling layer — reduces qutrit count by measuring some and
conditioning rotations on the outcomes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class PoolingLayer(BaseQutritLayer):
    """
    Quantum pooling: entangle pairs of qutrits, then measure one.

    For each pair (q_i, q_{i+1}), apply a CSUM and then mark q_i
    for measurement. This effectively halves the number of active qutrits.

    Parameters
    ----------
    num_qutrits : int
        Number of input qutrits (must be even).
    """

    def __init__(self, num_qutrits: int) -> None:
        if num_qutrits < 2:
            raise ValueError("PoolingLayer needs ≥ 2 qutrits.")
        super().__init__(num_qutrits, name="PoolingLayer")
        self._output_qutrits = (num_qutrits + 1) // 2
        # Trainable rotations on the kept qutrits after pooling
        rng = np.random.default_rng()
        self._params = rng.uniform(0, 2 * np.pi, size=self._output_qutrits * 3).tolist()

    @property
    def output_qutrits(self) -> int:
        return self._output_qutrits

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        idx = 0
        n = self._num_qutrits
        for i in range(0, n - 1, 2):
            circuit.csum(i, i + 1)
        # Apply parameterised rotations on the kept qutrits
        kept = list(range(1, n, 2))
        if n % 2 == 1:
            kept.append(n - 1)
        for q_idx, q in enumerate(kept[: self._output_qutrits]):
            for gm_k in [1, 2, 3]:
                if idx < len(self._params):
                    circuit.r_gm(gm_k, self._params[idx], q)
                    idx += 1
        return circuit
