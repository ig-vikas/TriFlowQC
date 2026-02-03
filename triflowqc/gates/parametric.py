"""
Parameterised qutrit rotation gates using Gell-Mann matrices.

R_k(θ) = expm(-i θ λ_k / 2)   for k = 1, 2, …, 8

where λ_k is the k-th Gell-Mann matrix and expm is the matrix exponential.

Convenience aliases:
    RX3(θ) = R_1(θ)  — rotation around Gell-Mann λ₁ axis
    RY3(θ) = R_2(θ)  — rotation around Gell-Mann λ₂ axis
    RZ3(θ) = R_3(θ)  — rotation around Gell-Mann λ₃ axis
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from triflowqc.gates.gell_mann import gell_mann


def R_GM(k: int, theta: float) -> NDArray[np.complexfloating]:
    """
    Rotation gate around the k-th Gell-Mann axis.

    R_k(θ) = exp(-i θ λ_k / 2)

    Parameters
    ----------
    k : int
        Gell-Mann index (1–8).
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray, shape (3, 3)
        Unitary rotation matrix.
    """
    lam = gell_mann(k)
    U = expm(-1j * theta * lam / 2.0)
    return U.astype(np.complex128)


def RX3(theta: float) -> NDArray[np.complexfloating]:
    """
    Rotation around the Gell-Mann λ₁ axis: R_1(θ) = exp(-i θ λ₁/2).

    Operates in the |0⟩–|1⟩ subspace (analogous to qubit Rx).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    """
    return R_GM(1, theta)


def RY3(theta: float) -> NDArray[np.complexfloating]:
    """
    Rotation around the Gell-Mann λ₂ axis: R_2(θ) = exp(-i θ λ₂/2).

    Operates in the |0⟩–|1⟩ subspace (analogous to qubit Ry).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    """
    return R_GM(2, theta)


def RZ3(theta: float) -> NDArray[np.complexfloating]:
    """
    Rotation around the Gell-Mann λ₃ axis: R_3(θ) = exp(-i θ λ₃/2).

    Operates in the |0⟩–|1⟩ subspace (analogous to qubit Rz).

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    """
    return R_GM(3, theta)


def general_su3_rotation(thetas: NDArray[np.floating]) -> NDArray[np.complexfloating]:
    """
    General SU(3) rotation parameterised by 8 angles.

    U = R_1(θ₁) R_2(θ₂) R_3(θ₃) R_4(θ₄) R_5(θ₅) R_6(θ₆) R_7(θ₇) R_8(θ₈)

    Parameters
    ----------
    thetas : array-like, shape (8,)
        Eight rotation angles.

    Returns
    -------
    ndarray, shape (3, 3)
        Unitary SU(3) matrix.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.shape != (8,):
        raise ValueError(f"Expected 8 angles, got shape {thetas.shape}")
    U = np.eye(3, dtype=np.complex128)
    for k in range(1, 9):
        U = R_GM(k, float(thetas[k - 1])) @ U
    return U
