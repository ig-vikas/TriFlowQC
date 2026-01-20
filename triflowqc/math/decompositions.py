"""
Matrix decompositions for qutrit quantum computing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm, polar
from typing import Tuple, List


def qr_decomposition(
    M: NDArray[np.complexfloating],
) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    QR decomposition with phase correction to ensure R has positive diagonal.

    Parameters
    ----------
    M : ndarray

    Returns
    -------
    Q : ndarray — unitary
    R : ndarray — upper triangular with positive diagonal
    """
    Q, R = np.linalg.qr(M)
    d = np.diag(R)
    ph = np.where(np.abs(d) > 1e-15, d / np.abs(d), 1.0)
    Q = Q @ np.diag(ph)
    R = np.diag(np.conj(ph)) @ R
    return Q, R


def polar_decomposition(
    M: NDArray[np.complexfloating],
) -> Tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    Polar decomposition M = U P where U is unitary and P is positive semi-definite.

    Parameters
    ----------
    M : ndarray

    Returns
    -------
    U : ndarray — unitary
    P : ndarray — positive semi-definite Hermitian
    """
    U, P = polar(M)
    return U, P


def spectral_decomposition(
    H: NDArray[np.complexfloating],
) -> Tuple[NDArray[np.float64], NDArray[np.complexfloating]]:
    """
    Spectral decomposition of a Hermitian matrix.

    H = Σ_k λ_k |v_k⟩⟨v_k|

    Parameters
    ----------
    H : ndarray — Hermitian matrix

    Returns
    -------
    eigenvalues : ndarray (d,)
    eigenvectors : ndarray (d, d)
        Column k is the eigenvector for eigenvalue k. Sorted ascending.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def schmidt_decomposition(
    psi: NDArray[np.complexfloating],
    dims: Tuple[int, int],
) -> Tuple[NDArray[np.float64], NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    """
    Schmidt decomposition of a bipartite pure state.

    |ψ⟩ = Σ_k λ_k |a_k⟩ ⊗ |b_k⟩

    Parameters
    ----------
    psi : ndarray (d_A * d_B,)
        State vector.
    dims : tuple of (d_A, d_B)
        Subsystem dimensions.

    Returns
    -------
    schmidt_coefficients : ndarray (r,)
        Non-negative Schmidt coefficients (sorted descending).
    U : ndarray (d_A, r)
        Left Schmidt vectors (columns).
    V : ndarray (d_B, r)
        Right Schmidt vectors (columns).
    """
    d_a, d_b = dims
    M = psi.reshape(d_a, d_b)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # Keep only non-negligible coefficients
    mask = S > 1e-15
    return S[mask], U[:, mask], Vh[mask, :].conj().T


def schmidt_rank(
    psi: NDArray[np.complexfloating],
    dims: Tuple[int, int],
    tol: float = 1e-10,
) -> int:
    """
    Compute the Schmidt rank of a bipartite state.

    Returns 1 for product states, >1 for entangled states.
    """
    coeffs, _, _ = schmidt_decomposition(psi, dims)
    return int(np.sum(coeffs > tol))


def bloch_vector_qutrit(rho: NDArray[np.complexfloating]) -> NDArray[np.float64]:
    """
    Compute the generalised Bloch vector (8-dimensional) for a single-qutrit state.

    ρ = (1/3)(I + √3 Σ_k n_k λ_k)

    Parameters
    ----------
    rho : ndarray (3×3)

    Returns
    -------
    ndarray (8,)
    """
    from triflowqc.math.su3 import su3_generators

    gens = su3_generators()
    n = np.zeros(8, dtype=np.float64)
    for k in range(8):
        n[k] = np.real(np.trace(rho @ gens[k])) * np.sqrt(3) / 2.0
    return n


def reconstruct_from_bloch(n: NDArray[np.float64]) -> NDArray[np.complexfloating]:
    """
    Reconstruct a qutrit density matrix from its Bloch vector.

    Parameters
    ----------
    n : ndarray (8,)

    Returns
    -------
    ndarray (3×3)
    """
    from triflowqc.math.su3 import su3_generators

    gens = su3_generators()
    rho = np.eye(3, dtype=np.complex128) / 3.0
    for k in range(8):
        rho += (n[k] / (2.0 * np.sqrt(3))) * gens[k]
    return rho


def cosine_sine_decomposition(
    U: NDArray[np.complexfloating],
) -> dict:
    """
    Cosine-Sine decomposition of a unitary matrix.

    Decomposes into block-diagonal unitaries and a diagonal of cosines/sines.

    Parameters
    ----------
    U : ndarray (2n×2n)

    Returns
    -------
    dict with keys: 'U1', 'U2', 'V1', 'V2', 'theta'
    """
    from scipy.linalg import cossin
    n = U.shape[0] // 2
    u, cs, vdh = cossin(U, p=n, q=n)

    # Extract theta from cs matrix
    theta = np.arccos(np.clip(np.diag(cs[:n, :n]), -1, 1))

    return {
        "U1": u[:n, :n],
        "U2": u[n:, n:],
        "V1": vdh[:n, :n],
        "V2": vdh[n:, n:],
        "theta": theta,
    }
