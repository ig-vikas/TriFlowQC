"""
Trainable parameter variable for quantum circuit optimisation.

Wraps a float value with tracking metadata used by optimisers.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class Variable:
    """
    A single trainable parameter.

    Parameters
    ----------
    value : float
        Initial value.
    name : str or None
        Optional human-readable name.
    trainable : bool
        If *False* the variable is frozen (gradients ignored).

    Examples
    --------
    >>> v = Variable(0.5, name="theta_0")
    >>> v.value
    0.5
    >>> v.value = 1.2
    >>> v.grad = -0.03
    """

    def __init__(
        self, value: float = 0.0, name: Optional[str] = None, trainable: bool = True
    ) -> None:
        self._value: float = float(value)
        self.name: Optional[str] = name
        self.trainable: bool = trainable
        self.grad: float = 0.0
        # Bookkeeping for adaptive optimisers
        self._m: float = 0.0  # first moment
        self._v: float = 0.0  # second moment
        self._step: int = 0

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, new_val: float) -> None:
        self._value = float(new_val)

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        self.grad = 0.0

    def numpy(self) -> float:
        """Return value as plain float (compatibility alias)."""
        return self._value

    @staticmethod
    def create_array(
        n: int, init: str = "random", name_prefix: str = "theta", trainable: bool = True
    ) -> list["Variable"]:
        """
        Create a list of *n* variables.

        Parameters
        ----------
        n : int
            Number of variables.
        init : str
            ``'random'`` — uniform [0, 2π); ``'zeros'`` — all 0; ``'ones'`` — all 1.
        name_prefix : str
            Prefix for auto-generated names.
        trainable : bool
            Whether the variables are trainable.
        """
        if init == "random":
            vals = np.random.uniform(0, 2 * np.pi, size=n)
        elif init == "zeros":
            vals = np.zeros(n)
        elif init == "ones":
            vals = np.ones(n)
        else:
            raise ValueError(f"Unknown init strategy: {init}")
        return [
            Variable(float(vals[i]), name=f"{name_prefix}_{i}", trainable=trainable)
            for i in range(n)
        ]

    def __repr__(self) -> str:
        return f"Variable({self._value:.6f}, name={self.name!r}, trainable={self.trainable})"

    def __float__(self) -> float:
        return self._value
