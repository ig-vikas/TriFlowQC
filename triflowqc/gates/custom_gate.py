"""
User-defined gate builder.

Allows users to construct custom qutrit gates from:
- Direct matrix specification
- Product of existing gates
- Parameterised expressions
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from scipy.linalg import expm


class CustomGate:
    """
    Build a custom qutrit gate.

    Parameters
    ----------
    matrix : ndarray or None
        If given, the 3×3 (or 9×9, 27×27, …) unitary matrix.
    name : str
        Human-readable name.

    Examples
    --------
    >>> g = CustomGate(my_unitary, name="MyGate")
    >>> g.matrix
    >>> g.is_unitary()
    """

    def __init__(
        self,
        matrix: Optional[NDArray[np.complexfloating]] = None,
        name: str = "Custom",
    ) -> None:
        self.name = name
        self._matrix: Optional[NDArray[np.complexfloating]] = None
        if matrix is not None:
            self._matrix = np.asarray(matrix, dtype=np.complex128)
            if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
                raise ValueError("Gate matrix must be square.")
            if not self.is_unitary():
                raise ValueError(f"Gate '{name}' is not unitary.")

    @property
    def matrix(self) -> NDArray[np.complexfloating]:
        if self._matrix is None:
            raise RuntimeError("Gate matrix has not been set.")
        return self._matrix.copy()

    @property
    def dim(self) -> int:
        if self._matrix is None:
            raise RuntimeError("Gate matrix has not been set.")
        return self._matrix.shape[0]

    def is_unitary(self, atol: float = 1e-8) -> bool:
        """Check if the gate is unitary."""
        if self._matrix is None:
            return False
        d = self._matrix.shape[0]
        return bool(np.allclose(self._matrix @ self._matrix.conj().T, np.eye(d), atol=atol))

    @classmethod
    def from_product(cls, gates: List[NDArray[np.complexfloating]], name: str = "Product") -> "CustomGate":
        """Construct a gate as the product of a list of gates (applied left to right)."""
        if not gates:
            raise ValueError("Gate list must be non-empty.")
        result = np.asarray(gates[0], dtype=np.complex128).copy()
        for g in gates[1:]:
            result = np.asarray(g, dtype=np.complex128) @ result
        return cls(result, name=name)

    @classmethod
    def from_hermitian(cls, H: NDArray[np.complexfloating], theta: float = 1.0, name: str = "Exp") -> "CustomGate":
        """
        Construct gate as exp(-i θ H) from a Hermitian matrix H.

        Parameters
        ----------
        H : ndarray
            Hermitian generator.
        theta : float
            Rotation angle.
        name : str
            Gate name.
        """
        H = np.asarray(H, dtype=np.complex128)
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise ValueError("Generator H must be Hermitian.")
        U = expm(-1j * theta * H)
        return cls(U, name=name)

    def __repr__(self) -> str:
        d = self._matrix.shape[0] if self._matrix is not None else "?"
        return f"CustomGate(name={self.name!r}, dim={d})"
