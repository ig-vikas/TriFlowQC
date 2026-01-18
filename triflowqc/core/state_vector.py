"""
State-vector representation utilities for qutrit systems.

Provides helper functions for creating, manipulating, and inspecting
state vectors in :math:`3^n`-dimensional Hilbert space.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Sequence


def basis_state(label: Sequence[int], num_qutrits: int) -> NDArray[np.complexfloating]:
    """
    Create a computational basis state vector.

    Parameters
    ----------
    label : sequence of int
        Trit values for each qutrit, e.g. [0, 2, 1].
    num_qutrits : int
        Total number of qutrits.

    Returns
    -------
    ndarray, shape (3**num_qutrits,)
        Normalised basis state vector.
    """
    label = list(label)
    if len(label) != num_qutrits:
        raise ValueError(f"label length {len(label)} != num_qutrits {num_qutrits}")
    for v in label:
        if v not in (0, 1, 2):
            raise ValueError(f"Each trit must be 0, 1, or 2; got {v}")
    dim = 3 ** num_qutrits
    index = 0
    for v in label:
        index = index * 3 + v
    sv = np.zeros(dim, dtype=np.complex128)
    sv[index] = 1.0 + 0.0j
    return sv


def tensor_product(*states: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Compute the tensor (Kronecker) product of state vectors.

    Parameters
    ----------
    *states : ndarray
        Any number of 1-D state vectors.

    Returns
    -------
    ndarray
        Kronecker product.
    """
    if len(states) == 0:
        raise ValueError("At least one state required.")
    result = states[0].copy()
    for s in states[1:]:
        result = np.kron(result, s)
    return result


def is_normalised(sv: NDArray[np.complexfloating], atol: float = 1e-10) -> bool:
    """Check whether a state vector is normalised."""
    return bool(np.isclose(np.linalg.norm(sv), 1.0, atol=atol))


def index_to_trits(index: int, num_qutrits: int) -> List[int]:
    """Convert a flat basis index to a list of trit values."""
    digits: List[int] = []
    for _ in range(num_qutrits):
        digits.append(index % 3)
        index //= 3
    return list(reversed(digits))


def trits_to_index(trits: Sequence[int]) -> int:
    """Convert a list of trit values to a flat basis index."""
    idx = 0
    for t in trits:
        idx = idx * 3 + int(t)
    return idx


class StateVector:
    """
    Wrapper around a raw :math:`3^n` complex vector with convenience methods.

    Parameters
    ----------
    data : ndarray
        Complex vector of length :math:`3^n`.
    num_qutrits : int
        Number of qutrits.
    """

    def __init__(self, data: NDArray[np.complexfloating], num_qutrits: int) -> None:
        data = np.asarray(data, dtype=np.complex128).ravel()
        expected = 3 ** num_qutrits
        if data.shape[0] != expected:
            raise ValueError(f"Expected length {expected}, got {data.shape[0]}")
        norm = np.linalg.norm(data)
        if norm < 1e-15:
            raise ValueError("State vector cannot be zero.")
        self._data = data / norm
        self._n = num_qutrits

    @property
    def data(self) -> NDArray[np.complexfloating]:
        return self._data.copy()

    @property
    def num_qutrits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return 3 ** self._n

    def probabilities(self) -> NDArray[np.floating]:
        return np.abs(self._data) ** 2

    def density_matrix(self) -> NDArray[np.complexfloating]:
        return np.outer(self._data, np.conj(self._data))

    def expectation(self, observable: NDArray[np.complexfloating]) -> complex:
        """⟨ψ|O|ψ⟩"""
        return complex(np.conj(self._data) @ observable @ self._data)

    def fidelity(self, other: "StateVector") -> float:
        return float(np.abs(np.vdot(self._data, other._data)) ** 2)

    def __repr__(self) -> str:
        return f"StateVector(num_qutrits={self._n}, dim={self.dim})"
