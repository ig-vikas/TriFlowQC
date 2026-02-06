"""
Dense variational qutrit layer — the core trainable layer.

For each qutrit, applies a sequence of Gell-Mann rotations with
trainable parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class DenseQutrit(BaseQutritLayer):
    """
    Dense (variational) qutrit layer.

    For each qutrit and each sub-layer, applies rotations around
    selected Gell-Mann axes.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits.
    num_layers : int
        Number of repetition layers.
    rotations : str
        ``'basic'`` — 3 rotations (R1, R2, R3) per qutrit per layer.
        ``'full'``  — 8 rotations (all Gell-Mann) per qutrit per layer.
    seed : int or None
        Random seed for parameter initialisation.

    Examples
    --------
    >>> layer = DenseQutrit(4, num_layers=2, rotations='basic')
    >>> layer.num_parameters
    24
    """

    def __init__(
        self,
        num_qutrits: int,
        num_layers: int = 1,
        rotations: str = "basic",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(num_qutrits, name="DenseQutrit")
        self._num_layers = num_layers
        self._rotations = rotations

        if rotations == "basic":
            self._gm_indices = [1, 2, 3]
        elif rotations == "full":
            self._gm_indices = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            raise ValueError(f"rotations must be 'basic' or 'full', got {rotations!r}")

        rng = np.random.default_rng(seed)
        n_params = num_qutrits * len(self._gm_indices) * num_layers
        self._params = rng.uniform(0, 2 * np.pi, size=n_params).tolist()

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        """Append dense rotation gates to the circuit."""
        idx = 0
        for _layer in range(self._num_layers):
            for q in range(self._num_qutrits):
                for gm_k in self._gm_indices:
                    circuit.r_gm(gm_k, self._params[idx], q)
                    idx += 1
        return circuit
