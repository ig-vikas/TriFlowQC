"""
Single-qutrit gates (3×3 unitary matrices).

Every gate is a 3×3 complex128 numpy array verified to be unitary
at module load time.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_omega: complex = np.exp(2j * np.pi / 3)  # primitive cube root of unity


def _validate_unitary(U: NDArray[np.complexfloating], name: str) -> None:
    """Raise if *U* is not unitary."""
    if not np.allclose(U @ U.conj().T, np.eye(3), atol=1e-10):
        raise RuntimeError(f"Gate {name} failed unitarity check.")


# ------------------------------------------------------------------
# Cyclic shift: |0⟩→|1⟩, |1⟩→|2⟩, |2⟩→|0⟩
# ------------------------------------------------------------------
X3: NDArray[np.complexfloating] = np.array(
    [[0, 0, 1],
     [1, 0, 0],
     [0, 1, 0]], dtype=np.complex128
)
"""Cyclic shift (generalised X): |0⟩→|1⟩, |1⟩→|2⟩, |2⟩→|0⟩.  X3³ = I."""
_validate_unitary(X3, "X3")

# ------------------------------------------------------------------
# Partial swaps
# ------------------------------------------------------------------
X3_01: NDArray[np.complexfloating] = np.array(
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 1]], dtype=np.complex128
)
"""Swap |0⟩↔|1⟩, leave |2⟩ unchanged."""
_validate_unitary(X3_01, "X3_01")

X3_12: NDArray[np.complexfloating] = np.array(
    [[1, 0, 0],
     [0, 0, 1],
     [0, 1, 0]], dtype=np.complex128
)
"""Swap |1⟩↔|2⟩, leave |0⟩ unchanged."""
_validate_unitary(X3_12, "X3_12")

X3_02: NDArray[np.complexfloating] = np.array(
    [[0, 0, 1],
     [0, 1, 0],
     [1, 0, 0]], dtype=np.complex128
)
"""Swap |0⟩↔|2⟩, leave |1⟩ unchanged."""
_validate_unitary(X3_02, "X3_02")

# ------------------------------------------------------------------
# Phase gate: diag(1, ω, ω²)
# ------------------------------------------------------------------
Z3: NDArray[np.complexfloating] = np.diag([1.0 + 0j, _omega, _omega ** 2]).astype(np.complex128)
"""Phase gate: diag(1, ω, ω²) where ω = exp(2πi/3).  Z3³ = I."""
_validate_unitary(Z3, "Z3")

# ------------------------------------------------------------------
# Qutrit Hadamard (3×3 QFT matrix / √3)
# ------------------------------------------------------------------
H3: NDArray[np.complexfloating] = (1.0 / np.sqrt(3.0)) * np.array(
    [[1,      1,          1],
     [1,      _omega,     _omega ** 2],
     [1,      _omega ** 2, _omega]], dtype=np.complex128
)
"""Qutrit Hadamard (3-point QFT / √3)."""
_validate_unitary(H3, "H3")

# ------------------------------------------------------------------
# T3 gate: diag(1, ω^{1/3}, ω^{2/3}) — finer phase rotation
# ------------------------------------------------------------------
_t3_phase = np.exp(2j * np.pi / 9)
T3: NDArray[np.complexfloating] = np.diag(
    [1.0 + 0j, _t3_phase, _t3_phase ** 2]
).astype(np.complex128)
"""Qutrit T gate: diag(1, e^{2πi/9}, e^{4πi/9}).  T3⁹ = I (up to global phase)."""
_validate_unitary(T3, "T3")

# ------------------------------------------------------------------
# S3 gate: diag(1, ω^{2/3}, ω^{4/3}) — intermediate phase
# ------------------------------------------------------------------
_s3_phase = np.exp(2j * np.pi / 3)
S3: NDArray[np.complexfloating] = np.diag(
    [1.0 + 0j, _s3_phase, _s3_phase ** 2]
).astype(np.complex128)
"""Qutrit S gate: diag(1, e^{2πi/3}, e^{4πi/3}).  S3 = Z3."""
_validate_unitary(S3, "S3")

# ------------------------------------------------------------------
# General phase gate
# ------------------------------------------------------------------
def Phase3(phi1: float, phi2: float) -> NDArray[np.complexfloating]:
    """
    Parameterised phase gate: diag(1, e^{iφ₁}, e^{iφ₂}).

    Parameters
    ----------
    phi1, phi2 : float
        Phase angles in radians.

    Returns
    -------
    ndarray, shape (3, 3)
    """
    U = np.diag([1.0 + 0j, np.exp(1j * phi1), np.exp(1j * phi2)]).astype(np.complex128)
    return U

# ------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------
Identity3: NDArray[np.complexfloating] = np.eye(3, dtype=np.complex128)
"""3×3 identity gate."""
