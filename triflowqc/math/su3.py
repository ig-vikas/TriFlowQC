"""
SU(3) Lie group utilities for qutrit quantum computing.

Provides generators, structure constants, Haar-random elements, and
basis decomposition of 3×3 matrices in the Gell-Mann basis.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from typing import Tuple


def su3_generators() -> list:
    """
    Return the 8 Gell-Mann matrices (generators of SU(3)).

    Returns
    -------
    list of ndarray
        λ₁ through λ₈, each 3×3 complex.
    """
    from triflowqc.gates.gell_mann import (
        LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
        LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8,
    )
    return [LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
            LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8]


def su3_structure_constants() -> NDArray[np.float64]:
    """
    Compute the structure constants f_{ijk} of SU(3).

    Defined by [λ_i, λ_j] = 2i Σ_k f_{ijk} λ_k.

    Returns
    -------
    ndarray of shape (8, 8, 8)
    """
    from triflowqc.gates.gell_mann import structure_constants
    return structure_constants()


def su3_basis_decompose(M: NDArray[np.complexfloating]) -> NDArray[np.float64]:
    """
    Decompose a 3×3 traceless Hermitian matrix in the Gell-Mann basis.

    M = Σ_k c_k λ_k where c_k = Tr(M λ_k) / 2.

    Parameters
    ----------
    M : ndarray (3×3)
        Traceless Hermitian matrix.

    Returns
    -------
    ndarray of shape (8,)
        Coefficients c_1 through c_8.
    """
    M = np.asarray(M, dtype=np.complex128)
    gens = su3_generators()
    coeffs = np.zeros(8, dtype=np.float64)
    for k in range(8):
        coeffs[k] = np.real(np.trace(M @ gens[k])) / 2.0
    return coeffs


def su3_reconstruct(coeffs: NDArray[np.float64]) -> NDArray[np.complexfloating]:
    """
    Reconstruct a traceless Hermitian matrix from Gell-Mann coefficients.

    Parameters
    ----------
    coeffs : ndarray of shape (8,)

    Returns
    -------
    ndarray (3×3)
    """
    gens = su3_generators()
    M = np.zeros((3, 3), dtype=np.complex128)
    for k in range(8):
        M += coeffs[k] * gens[k]
    return M


def su3_random_element(rng: np.random.Generator = None) -> NDArray[np.complexfloating]:
    """
    Generate a Haar-random SU(3) matrix.

    Uses the QR decomposition of a random Gaussian matrix with phase
    correction to ensure uniform distribution on SU(3).

    Parameters
    ----------
    rng : numpy Generator or None

    Returns
    -------
    ndarray (3×3) — a unitary matrix with det = 1.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random complex Gaussian matrix
    Z = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    Q, R = np.linalg.qr(Z)

    # Fix phase: ensure diagonal of R is positive
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q @ np.diag(ph)

    # Normalise determinant to 1
    det = np.linalg.det(Q)
    Q = Q / (det ** (1.0 / 3.0))

    return Q


def is_su3(U: NDArray[np.complexfloating], atol: float = 1e-8) -> bool:
    """
    Check if a 3×3 matrix is in SU(3).

    Verifies U†U = I and det(U) = 1.

    Parameters
    ----------
    U : ndarray (3×3)
    atol : float

    Returns
    -------
    bool
    """
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (3, 3):
        return False
    # Unitarity
    eye = U.conj().T @ U
    if not np.allclose(eye, np.eye(3), atol=atol):
        return False
    # Determinant = 1
    det = np.linalg.det(U)
    if not np.isclose(det, 1.0, atol=atol):
        return False
    return True


def su3_exponential(
    coefficients: NDArray[np.float64],
) -> NDArray[np.complexfloating]:
    """
    Compute exp(-i Σ_k c_k λ_k / 2) — a general SU(3) element from
    Lie algebra coefficients.

    Parameters
    ----------
    coefficients : ndarray of shape (8,)

    Returns
    -------
    ndarray (3×3)
    """
    H = su3_reconstruct(coefficients)
    return expm(-1j * H / 2.0)


def su3_adjoint_representation(
    U: NDArray[np.complexfloating],
) -> NDArray[np.float64]:
    """
    Compute the 8×8 adjoint representation of an SU(3) element.

    Ad(U)_{ij} = Tr(U λ_i U† λ_j) / 2.

    Parameters
    ----------
    U : ndarray (3×3) — SU(3) element.

    Returns
    -------
    ndarray (8×8)
    """
    gens = su3_generators()
    ad = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        rotated = U @ gens[i] @ U.conj().T
        for j in range(8):
            ad[i, j] = np.real(np.trace(rotated @ gens[j])) / 2.0
    return ad


def casimir_operator() -> NDArray[np.complexfloating]:
    """
    Compute the quadratic Casimir operator C₂ = Σ_k λ_k².

    For the fundamental representation of SU(3), C₂ = (16/3) I.

    Returns
    -------
    ndarray (3×3)
    """
    gens = su3_generators()
    C = np.zeros((3, 3), dtype=np.complex128)
    for lam in gens:
        C += lam @ lam
    return C
