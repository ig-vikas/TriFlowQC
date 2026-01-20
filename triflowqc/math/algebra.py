"""
Linear algebra utilities for qutrit quantum computing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm, sqrtm
from typing import Tuple, List


def tensor_product(*matrices: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Compute the Kronecker (tensor) product of multiple matrices.

    Parameters
    ----------
    *matrices : ndarray

    Returns
    -------
    ndarray
    """
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def partial_trace_pure(
    state: NDArray[np.complexfloating],
    keep: List[int],
    dims: List[int],
) -> NDArray[np.complexfloating]:
    """
    Partial trace of a pure state's density matrix ρ = |ψ⟩⟨ψ|.

    Parameters
    ----------
    state : ndarray
        State vector.
    keep : list of int
        Indices of subsystems to keep.
    dims : list of int
        Dimensions of each subsystem (e.g. [3, 3] for 2 qutrits).

    Returns
    -------
    ndarray
        Reduced density matrix.
    """
    rho = np.outer(state, np.conj(state))
    return partial_trace_mixed(rho, keep, dims)


def partial_trace_mixed(
    rho: NDArray[np.complexfloating],
    keep: List[int],
    dims: List[int],
) -> NDArray[np.complexfloating]:
    """
    Partial trace of a density matrix over specified subsystems.

    Parameters
    ----------
    rho : ndarray
        Density matrix of the full system.
    keep : list of int
        Subsystem indices to keep.
    dims : list of int
        Dimensions of each subsystem.

    Returns
    -------
    ndarray
        Reduced density matrix.
    """
    n = len(dims)
    rho_tensor = rho.reshape(dims + dims)

    # Determine which subsystems to trace out
    trace_out = sorted(set(range(n)) - set(keep))

    # Trace out subsystems in reverse order (to keep index stability)
    for idx in reversed(trace_out):
        # Trace: contract axis idx with axis idx + remaining_n
        remaining_n = rho_tensor.ndim // 2
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + remaining_n)

    # Reshape to matrix
    keep_dim = int(np.prod([dims[k] for k in keep]))
    return rho_tensor.reshape(keep_dim, keep_dim)


def commutator(
    A: NDArray[np.complexfloating],
    B: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(
    A: NDArray[np.complexfloating],
    B: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Compute {A, B} = AB + BA."""
    return A @ B + B @ A


def matrix_exponential(
    M: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Matrix exponential using scipy."""
    return expm(M)


def matrix_logarithm(
    U: NDArray[np.complexfloating],
) -> NDArray[np.complexfloating]:
    """Matrix logarithm."""
    from scipy.linalg import logm
    return logm(U)


def fidelity_pure(
    psi: NDArray[np.complexfloating],
    phi: NDArray[np.complexfloating],
) -> float:
    """
    Fidelity between two pure states: |⟨ψ|φ⟩|².

    Parameters
    ----------
    psi, phi : ndarray
        State vectors.

    Returns
    -------
    float
    """
    return float(np.abs(np.vdot(psi, phi)) ** 2)


def fidelity_mixed(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """
    Fidelity between two density matrices:
    F(ρ, σ) = (Tr √(√ρ σ √ρ))².

    Parameters
    ----------
    rho, sigma : ndarray
        Density matrices.

    Returns
    -------
    float
    """
    sqrt_rho = sqrtm(rho)
    M = sqrt_rho @ sigma @ sqrt_rho
    sqrt_M = sqrtm(M)
    tr = np.real(np.trace(sqrt_M))
    return float(tr ** 2)


def trace_distance(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """
    Trace distance: D(ρ, σ) = ½ Tr|ρ - σ|.

    Parameters
    ----------
    rho, sigma : ndarray

    Returns
    -------
    float
    """
    diff = rho - sigma
    eigenvalues = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigenvalues))


def is_unitary(U: NDArray[np.complexfloating], atol: float = 1e-8) -> bool:
    """Check if U†U = I."""
    n = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(n), atol=atol)


def is_hermitian(M: NDArray[np.complexfloating], atol: float = 1e-8) -> bool:
    """Check if M = M†."""
    return np.allclose(M, M.conj().T, atol=atol)


def is_positive_semidefinite(
    M: NDArray[np.complexfloating], atol: float = 1e-8
) -> bool:
    """Check if all eigenvalues ≥ -atol."""
    eigenvalues = np.linalg.eigvalsh(M)
    return bool(np.all(eigenvalues >= -atol))


def gram_schmidt_qutrit(
    vectors: List[NDArray[np.complexfloating]],
) -> List[NDArray[np.complexfloating]]:
    """
    Gram-Schmidt orthonormalisation of complex vectors.

    Parameters
    ----------
    vectors : list of ndarray

    Returns
    -------
    list of ndarray
        Orthonormal vectors.
    """
    ortho = []
    for v in vectors:
        v = np.asarray(v, dtype=np.complex128).copy()
        for u in ortho:
            v -= np.vdot(u, v) * u
        norm = np.linalg.norm(v)
        if norm > 1e-15:
            ortho.append(v / norm)
    return ortho


def project_to_unitary(M: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Project a matrix to the nearest unitary via polar decomposition.

    Parameters
    ----------
    M : ndarray

    Returns
    -------
    ndarray — unitary matrix
    """
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh
