"""
Functional model — supports non-sequential topologies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Sequence, Tuple
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.layers.base_layer import BaseQutritLayer


class FunctionalModel(Sequential):
    """
    Functional model — currently wraps Sequential with the same interface.

    For most qutrit QML tasks a sequential pipeline suffices. This class
    is provided for API compatibility and future extension to DAG-based
    architectures.
    """

    def __init__(self, layers: Sequence[BaseQutritLayer]) -> None:
        super().__init__(layers)

    def __repr__(self) -> str:
        return f"FunctionalModel(layers={len(self._layers)}, params={self.num_parameters})"
