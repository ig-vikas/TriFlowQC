"""
Measurement output layer — reads out expectation values from qutrits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.nn.layers.base_layer import BaseQutritLayer

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class MeasurementLayer(BaseQutritLayer):
    """
    Measurement layer that extracts class probabilities from qutrits.

    For classification with *num_classes* classes, measures the first
    *num_classes* qutrits and uses their marginal probabilities for
    each outcome as the output logits.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits in the circuit.
    num_classes : int
        Number of output classes (≤ num_qutrits * 3).
        If ≤ 3, measure qurit 0 and use P(|0⟩), P(|1⟩), P(|2⟩).
        If > 3, measure multiple qutrits.
    """

    def __init__(self, num_qutrits: int, num_classes: int = 3) -> None:
        super().__init__(num_qutrits, name="MeasurementLayer")
        self._num_classes = num_classes
        self._params = []

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        """Mark qutrits for measurement."""
        circuit.measure_all()
        return circuit

    def extract_probabilities(
        self, probs: NDArray[np.floating], num_qutrits: int
    ) -> NDArray[np.floating]:
        """
        Extract class probabilities from the full probability vector.

        For num_classes ≤ 3: marginal of qutrit 0.
        For num_classes > 3: marginals of first ceil(num_classes/3) qutrits.

        Parameters
        ----------
        probs : ndarray, shape (3^n,)
            Full probability vector.
        num_qutrits : int
            Number of qutrits in the register.

        Returns
        -------
        ndarray, shape (num_classes,)
            Output class probabilities (normalised).
        """
        dim = 3 ** num_qutrits
        probs = np.asarray(probs, dtype=np.float64)[:dim]

        if self._num_classes <= 3:
            # Marginal of qutrit 0
            tensor = probs.reshape([3] * num_qutrits)
            axes_to_sum = tuple(range(1, num_qutrits))
            if axes_to_sum:
                marginal = np.sum(tensor, axis=axes_to_sum)
            else:
                marginal = tensor
            output = marginal[: self._num_classes]
        else:
            # Measure multiple qutrits
            nq_needed = int(np.ceil(self._num_classes / 3))
            nq_needed = min(nq_needed, num_qutrits)
            tensor = probs.reshape([3] * num_qutrits)
            axes_to_sum = tuple(range(nq_needed, num_qutrits))
            if axes_to_sum:
                reduced = np.sum(tensor, axis=axes_to_sum)
            else:
                reduced = tensor
            output = reduced.ravel()[: self._num_classes]

        total = output.sum()
        if total > 1e-15:
            output = output / total
        else:
            output = np.ones(self._num_classes) / self._num_classes
        return output
