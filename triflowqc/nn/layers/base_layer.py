"""
Abstract base class for all qutrit neural-network layers.
"""

from __future__ import annotations

import abc
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class BaseQutritLayer(abc.ABC):
    """
    Base class for qutrit layers.

    Every layer must implement ``build_circuit`` which appends its gates
    to an existing :class:`QutritCircuit`, and ``num_parameters``.
    """

    def __init__(self, num_qutrits: int, name: str = "layer") -> None:
        self._num_qutrits = num_qutrits
        self._name = name
        self._params: List[float] = []

    @property
    def num_qutrits(self) -> int:
        return self._num_qutrits

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_parameters(self) -> int:
        return len(self._params)

    @property
    def parameters(self) -> List[float]:
        return list(self._params)

    @parameters.setter
    def parameters(self, values: List[float]) -> None:
        if len(values) != len(self._params):
            raise ValueError(
                f"Expected {len(self._params)} parameters, got {len(values)}"
            )
        self._params = [float(v) for v in values]

    @abc.abstractmethod
    def build_circuit(
        self,
        circuit: "QutritCircuit",
        x: Optional[NDArray[np.floating]] = None,
    ) -> "QutritCircuit":
        """
        Append this layer's gates to *circuit*.

        Parameters
        ----------
        circuit : QutritCircuit
            Circuit to append to.
        x : ndarray or None
            Input features (for embedding layers).

        Returns
        -------
        QutritCircuit
        """
        ...

    def summary_line(self) -> str:
        """One-line summary for model.summary()."""
        return f"{self._name:<30s} qutrits={self._num_qutrits}  params={self.num_parameters}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_qutrits={self._num_qutrits}, params={self.num_parameters})"
