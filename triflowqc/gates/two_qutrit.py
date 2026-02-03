"""
Two-qutrit gates (9×9 unitary matrices).

These operate on the tensor-product space C³ ⊗ C³ with computational basis
|00⟩, |01⟩, |02⟩, |10⟩, |11⟩, |12⟩, |20⟩, |21⟩, |22⟩.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_unitary_9(U: NDArray[np.complexfloating], name: str) -> None:
    if not np.allclose(U @ U.conj().T, np.eye(9), atol=1e-10):
        raise RuntimeError(f"Two-qutrit gate {name} failed unitarity check.")


def _build_csum() -> NDArray[np.complexfloating]:
    """
    CSUM (controlled-SUM) gate: |a, b⟩ → |a, (a + b) mod 3⟩.

    This is the qutrit analogue of CNOT.
    """
    U = np.zeros((9, 9), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            row = a * 3 + (a + b) % 3
            col = a * 3 + b
            U[row, col] = 1.0
    return U


def _build_cphase() -> NDArray[np.complexfloating]:
    """
    Controlled-phase gate: |a, b⟩ → ω^{ab} |a, b⟩  where ω = e^{2πi/3}.
    """
    omega = np.exp(2j * np.pi / 3)
    diag = np.zeros(9, dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            diag[a * 3 + b] = omega ** (a * b)
    return np.diag(diag)


def _build_swap3() -> NDArray[np.complexfloating]:
    """SWAP gate: |a, b⟩ → |b, a⟩."""
    U = np.zeros((9, 9), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            row = b * 3 + a
            col = a * 3 + b
            U[row, col] = 1.0
    return U


def _build_cx3() -> NDArray[np.complexfloating]:
    """
    Controlled-X3 gate: |a, b⟩ → |a, X3^a |b⟩⟩.

    X3 is the cyclic shift 0→1→2→0, so X3^a shifts by *a* positions.
    This is identical to CSUM.
    """
    return _build_csum()


# Build and validate at import time
CSUM: NDArray[np.complexfloating] = _build_csum()
"""CSUM gate: |a,b⟩ → |a, (a+b) mod 3⟩. Qutrit analogue of CNOT."""
_validate_unitary_9(CSUM, "CSUM")

CPHASE: NDArray[np.complexfloating] = _build_cphase()
"""Controlled-phase gate: |a,b⟩ → ω^{ab} |a,b⟩."""
_validate_unitary_9(CPHASE, "CPHASE")

SWAP3: NDArray[np.complexfloating] = _build_swap3()
"""SWAP gate: |a,b⟩ → |b,a⟩."""
_validate_unitary_9(SWAP3, "SWAP3")

CX3: NDArray[np.complexfloating] = _build_cx3()
"""Controlled-X3 gate: |a,b⟩ → |a, (a+b) mod 3⟩. Same as CSUM."""
_validate_unitary_9(CX3, "CX3")
