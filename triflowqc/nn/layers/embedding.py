"""
Embedding layers that encode classical data into qutrit quantum states.

- AngleEmbedding:     feature → rotation angle on qutrit
- AmplitudeEmbedding: feature vector → qutrit amplitudes
- BasisEmbedding:     integer {0,1,2} → basis state
- TernaryEmbedding:   convert to base-3, encode each trit
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class AngleEmbedding(BaseQutritLayer):
    """
    Encode each feature as a rotation angle on one qutrit.

    Feature x_i is encoded by applying R_GM(k, x_i) to qutrit i.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits (and expected input features).
    gm_index : int
        Which Gell-Mann rotation to use (1–8). Default 2 (RY3).
    """

    def __init__(self, num_qutrits: int, gm_index: int = 2) -> None:
        super().__init__(num_qutrits, name="AngleEmbedding")
        self._gm_index = gm_index
        # No trainable parameters (data-driven)
        self._params = []

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        if x is None:
            raise ValueError("AngleEmbedding requires input features x.")
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.shape[0] < self._num_qutrits:
            raise ValueError(
                f"Expected ≥ {self._num_qutrits} features, got {x.shape[0]}"
            )
        for i in range(self._num_qutrits):
            circuit.r_gm(self._gm_index, float(x[i]), i)
        return circuit


class AmplitudeEmbedding(BaseQutritLayer):
    """
    Encode a normalised feature vector as qutrit amplitudes.

    The feature vector length must equal 3^n (where n = num_qutrits) or
    be padded/truncated to that size.
    """

    def __init__(self, num_qutrits: int) -> None:
        super().__init__(num_qutrits, name="AmplitudeEmbedding")
        self._params = []

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        """
        For AmplitudeEmbedding the state is prepared directly (not via gates).
        The circuit's initial state should be set externally.
        This is a marker layer that records the intent.
        """
        # AmplitudeEmbedding needs special handling at the simulator level:
        # We store the data in the circuit metadata for the simulator to read.
        if x is None:
            raise ValueError("AmplitudeEmbedding requires input features x.")
        x = np.asarray(x, dtype=np.complex128).ravel()
        dim = 3 ** self._num_qutrits
        if x.shape[0] > dim:
            x = x[:dim]
        elif x.shape[0] < dim:
            x = np.pad(x, (0, dim - x.shape[0]))
        norm = np.linalg.norm(x)
        if norm < 1e-15:
            raise ValueError("Feature vector is zero.")
        x = x / norm
        # Store in circuit for later use
        if not hasattr(circuit, '_amplitude_state'):
            circuit._amplitude_state = None
        circuit._amplitude_state = x
        return circuit


class BasisEmbedding(BaseQutritLayer):
    """
    Encode integer labels {0, 1, 2} directly as basis states.

    Input: array of integers of length num_qutrits.
    """

    def __init__(self, num_qutrits: int) -> None:
        super().__init__(num_qutrits, name="BasisEmbedding")
        self._params = []

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        if x is None:
            raise ValueError("BasisEmbedding requires input x.")
        x = np.asarray(x, dtype=np.int64).ravel()
        if x.shape[0] < self._num_qutrits:
            raise ValueError(
                f"Expected ≥ {self._num_qutrits} values, got {x.shape[0]}"
            )
        for i in range(self._num_qutrits):
            val = int(x[i])
            if val == 1:
                circuit.x3_01(i)  # |0⟩ → |1⟩
            elif val == 2:
                circuit.x3_02(i)  # |0⟩ → |2⟩
            # val == 0: no gate needed
        return circuit


class TernaryEmbedding(BaseQutritLayer):
    """
    Convert input integer to base-3 and encode each trit as a basis state.
    """

    def __init__(self, num_qutrits: int) -> None:
        super().__init__(num_qutrits, name="TernaryEmbedding")
        self._params = []

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        if x is None:
            raise ValueError("TernaryEmbedding requires input x.")
        x_val = int(np.asarray(x).ravel()[0])
        trits = []
        for _ in range(self._num_qutrits):
            trits.append(x_val % 3)
            x_val //= 3
        trits = list(reversed(trits))
        for i, val in enumerate(trits):
            if val == 1:
                circuit.x3_01(i)
            elif val == 2:
                circuit.x3_02(i)
        return circuit
