"""
Three-qutrit gates (27×27 unitary matrices).

Operates on C³ ⊗ C³ ⊗ C³.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _build_toffoli3() -> NDArray[np.complexfloating]:
    """
    Qutrit Toffoli gate: |a, b, c⟩ → |a, b, (c + a·b) mod 3⟩.

    The target qutrit is shifted by (a * b) mod 3, controlled on two qutrits.
    When both controls are |1⟩, the target undergoes a +1 cyclic shift.
    This is a 27×27 permutation matrix.
    """
    dim = 27
    U = np.zeros((dim, dim), dtype=np.complex128)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                col = a * 9 + b * 3 + c
                c_new = (c + a * b) % 3
                row = a * 9 + b * 3 + c_new
                U[row, col] = 1.0
    return U


Toffoli3: NDArray[np.complexfloating] = _build_toffoli3()
"""Qutrit Toffoli: |a,b,c⟩ → |a, b, (c + a·b) mod 3⟩."""

# Validate
assert np.allclose(Toffoli3 @ Toffoli3.conj().T, np.eye(27), atol=1e-10), \
    "Toffoli3 unitarity check failed"
