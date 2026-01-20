"""
Quantum information theory measures for qutrit systems.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from typing import List


def von_neumann_entropy(rho: NDArray[np.complexfloating]) -> float:
    """
    Von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ).

    Parameters
    ----------
    rho : ndarray
        Density matrix.

    Returns
    -------
    float
        Entropy in bits.
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    # Filter zero eigenvalues
    nonzero = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def linear_entropy(rho: NDArray[np.complexfloating]) -> float:
    """
    Linear entropy: S_lin(ρ) = 1 - Tr(ρ²).

    Ranges from 0 (pure) to (d-1)/d (maximally mixed).

    Parameters
    ----------
    rho : ndarray

    Returns
    -------
    float
    """
    return float(1.0 - np.real(np.trace(rho @ rho)))


def purity(rho: NDArray[np.complexfloating]) -> float:
    """Purity: Tr(ρ²)."""
    return float(np.real(np.trace(rho @ rho)))


def mutual_information(
    rho_ab: NDArray[np.complexfloating],
    dims: List[int],
) -> float:
    """
    Quantum mutual information I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB).

    Parameters
    ----------
    rho_ab : ndarray
        Joint density matrix of bipartite system.
    dims : list of int
        Dimensions of subsystems A and B (e.g. [3, 3]).

    Returns
    -------
    float
    """
    from triflowqc.math.algebra import partial_trace_mixed

    rho_a = partial_trace_mixed(rho_ab, keep=[0], dims=dims)
    rho_b = partial_trace_mixed(rho_ab, keep=[1], dims=dims)

    S_a = von_neumann_entropy(rho_a)
    S_b = von_neumann_entropy(rho_b)
    S_ab = von_neumann_entropy(rho_ab)

    return S_a + S_b - S_ab


def concurrence_qutrit(rho: NDArray[np.complexfloating]) -> float:
    """
    Generalised concurrence for a two-qutrit system.

    C(ρ) = √(d/(d-1) · (1 - Tr(ρ_A²)))

    where d = 3 and ρ_A is the reduced density matrix.

    Parameters
    ----------
    rho : ndarray (9×9)
        Two-qutrit density matrix.

    Returns
    -------
    float
    """
    from triflowqc.math.algebra import partial_trace_mixed

    if rho.shape != (9, 9):
        raise ValueError("Expected 9×9 density matrix for two qutrits")

    rho_a = partial_trace_mixed(rho, keep=[0], dims=[3, 3])
    purity_a = np.real(np.trace(rho_a @ rho_a))

    d = 3
    val = (d / (d - 1)) * (1.0 - purity_a)
    return float(np.sqrt(max(val, 0.0)))


def negativity(
    rho: NDArray[np.complexfloating], dims: List[int]
) -> float:
    """
    Negativity as entanglement measure via partial transpose.

    N(ρ) = (||ρ^{T_B}||₁ - 1) / 2

    Parameters
    ----------
    rho : ndarray
        Bipartite density matrix.
    dims : list of int
        Subsystem dimensions.

    Returns
    -------
    float
    """
    rho_pt = partial_transpose(rho, dims, subsystem=1)
    eigenvalues = np.linalg.eigvalsh(rho_pt)
    return float((np.sum(np.abs(eigenvalues)) - 1.0) / 2.0)


def partial_transpose(
    rho: NDArray[np.complexfloating],
    dims: List[int],
    subsystem: int = 1,
) -> NDArray[np.complexfloating]:
    """
    Partial transpose of a bipartite density matrix.

    Parameters
    ----------
    rho : ndarray
    dims : list of int
        [d_A, d_B]
    subsystem : int
        Which subsystem to transpose (0 or 1).

    Returns
    -------
    ndarray
    """
    d_a, d_b = dims
    rho_reshaped = rho.reshape(d_a, d_b, d_a, d_b)

    if subsystem == 1:
        # Transpose subsystem B: swap axes 1 and 3
        rho_pt = rho_reshaped.transpose(0, 3, 2, 1)
    else:
        # Transpose subsystem A: swap axes 0 and 2
        rho_pt = rho_reshaped.transpose(2, 1, 0, 3)

    return rho_pt.reshape(d_a * d_b, d_a * d_b)


def relative_entropy(
    rho: NDArray[np.complexfloating],
    sigma: NDArray[np.complexfloating],
) -> float:
    """
    Quantum relative entropy: S(ρ || σ) = Tr(ρ (log₂ρ - log₂σ)).

    Parameters
    ----------
    rho, sigma : ndarray

    Returns
    -------
    float
    """
    from scipy.linalg import logm

    log_rho = logm(rho) / np.log(2)
    log_sigma = logm(sigma) / np.log(2)
    return float(np.real(np.trace(rho @ (log_rho - log_sigma))))


def max_entangled_state(d: int = 3) -> NDArray[np.complexfloating]:
    """
    Maximally entangled state |Φ⁺⟩ = (1/√d) Σ_i |ii⟩ for two d-dimensional systems.

    Parameters
    ----------
    d : int

    Returns
    -------
    ndarray of shape (d²,)
    """
    state = np.zeros(d * d, dtype=np.complex128)
    for i in range(d):
        state[i * d + i] = 1.0 / np.sqrt(d)
    return state


def entropy_of_formation(rho: NDArray[np.complexfloating], dims: List[int]) -> float:
    """
    Upper bound on entanglement of formation via negativity.

    For qutrits, this provides a useful bound.

    Parameters
    ----------
    rho : ndarray
    dims : list of int

    Returns
    -------
    float
    """
    neg = negativity(rho, dims)
    if neg <= 0:
        return 0.0
    # Upper bound using log-negativity
    return float(np.log2(2 * neg + 1))
