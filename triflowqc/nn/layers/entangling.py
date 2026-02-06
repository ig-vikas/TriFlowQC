"""
Entangling layer using CSUM gates between qutrits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class EntanglingLayer(BaseQutritLayer):
    """
    Entangling layer applying CSUM gates in a chosen pattern.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits.
    pattern : str
        ``'linear'`` — adjacent pairs (0,1), (1,2), …
        ``'ring'``   — linear + (n-1, 0)
        ``'full'``   — all pairs
        ``'random'`` — random selection
    seed : int or None
        Seed for ``'random'`` pattern.
    """

    def __init__(
        self,
        num_qutrits: int,
        pattern: str = "ring",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_qutrits, name=f"Entangling({pattern})")
        self._pattern = pattern
        self._pairs = self._build_pairs(seed)
        self._params = []  # no trainable parameters

    def _build_pairs(self, seed: Optional[int]) -> List[Tuple[int, int]]:
        n = self._num_qutrits
        if n < 2:
            return []
        if self._pattern == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        elif self._pattern == "ring":
            pairs = [(i, i + 1) for i in range(n - 1)]
            pairs.append((n - 1, 0))
            return pairs
        elif self._pattern == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif self._pattern == "random":
            rng = np.random.default_rng(seed)
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            k = min(n, len(all_pairs))
            chosen = rng.choice(len(all_pairs), size=k, replace=False)
            return [all_pairs[c] for c in chosen]
        else:
            raise ValueError(f"Unknown pattern: {self._pattern}")

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        for ctrl, tgt in self._pairs:
            circuit.csum(ctrl, tgt)
        return circuit
