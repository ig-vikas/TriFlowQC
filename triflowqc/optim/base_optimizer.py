"""
Abstract base class for optimisers.
"""

from __future__ import annotations

import abc
import numpy as np
from typing import List


class BaseOptimizer(abc.ABC):
    """
    Base class for parameter optimisers.

    Subclasses must implement ``step``.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    @abc.abstractmethod
    def step(self, parameters: List[float], gradients: np.ndarray) -> List[float]:
        """
        Update parameters given gradients.

        Parameters
        ----------
        parameters : list of float
            Current parameter values.
        gradients : ndarray
            Gradient vector (same length as parameters).

        Returns
        -------
        list of float
            Updated parameter values.
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state (moments, etc.)."""
        ...
