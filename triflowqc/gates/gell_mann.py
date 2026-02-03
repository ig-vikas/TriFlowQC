"""
All 8 Gell-Mann matrices — the SU(3) generators.

These are to qutrits what Pauli matrices are to qubits.
They form a basis for the Lie algebra su(3) and satisfy:

- Hermitian: λ† = λ
- Traceless: Tr(λ) = 0
- Orthogonality: Tr(λ_i λ_j) = 2 δ_{ij}

The structure constants f_{ijk} are defined by
    [λ_i, λ_j] = 2i Σ_k f_{ijk} λ_k
and are totally antisymmetric.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Gell-Mann matrices (1-indexed λ₁ … λ₈)
# ---------------------------------------------------------------------------

LAMBDA_1: NDArray[np.complexfloating] = np.array(
    [[0, 1, 0],
     [1, 0, 0],
     [0, 0, 0]], dtype=np.complex128
)
"""λ₁: Off-diagonal (0↔1) real. Analogous to Pauli σ_x in the 0-1 subspace."""

LAMBDA_2: NDArray[np.complexfloating] = np.array(
    [[0, -1j, 0],
     [1j, 0,  0],
     [0,  0,  0]], dtype=np.complex128
)
"""λ₂: Off-diagonal (0↔1) imaginary. Analogous to Pauli σ_y in the 0-1 subspace."""

LAMBDA_3: NDArray[np.complexfloating] = np.array(
    [[1,  0, 0],
     [0, -1, 0],
     [0,  0, 0]], dtype=np.complex128
)
"""λ₃: Diagonal (0-1). Analogous to Pauli σ_z in the 0-1 subspace."""

LAMBDA_4: NDArray[np.complexfloating] = np.array(
    [[0, 0, 1],
     [0, 0, 0],
     [1, 0, 0]], dtype=np.complex128
)
"""λ₄: Off-diagonal (0↔2) real."""

LAMBDA_5: NDArray[np.complexfloating] = np.array(
    [[0, 0, -1j],
     [0, 0,  0],
     [1j, 0, 0]], dtype=np.complex128
)
"""λ₅: Off-diagonal (0↔2) imaginary."""

LAMBDA_6: NDArray[np.complexfloating] = np.array(
    [[0, 0, 0],
     [0, 0, 1],
     [0, 1, 0]], dtype=np.complex128
)
"""λ₆: Off-diagonal (1↔2) real."""

LAMBDA_7: NDArray[np.complexfloating] = np.array(
    [[0, 0,  0],
     [0, 0, -1j],
     [0, 1j, 0]], dtype=np.complex128
)
"""λ₇: Off-diagonal (1↔2) imaginary."""

LAMBDA_8: NDArray[np.complexfloating] = (1.0 / np.sqrt(3.0)) * np.array(
    [[1, 0,  0],
     [0, 1,  0],
     [0, 0, -2]], dtype=np.complex128
)
"""λ₈: Diagonal. (1/√3) diag(1, 1, -2)."""

_ALL_LAMBDAS = [LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
                LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8]


def gell_mann_matrices() -> list:
    """
    Return all 8 Gell-Mann matrices as a list.

    Returns
    -------
    list of ndarray
        [λ₁, λ₂, …, λ₈], each 3×3 complex.
    """
    return [lam.copy() for lam in _ALL_LAMBDAS]


def gell_mann(k: int) -> NDArray[np.complexfloating]:
    """
    Return the *k*-th Gell-Mann matrix (1 ≤ k ≤ 8).

    Parameters
    ----------
    k : int
        Index in {1, 2, …, 8}.

    Returns
    -------
    ndarray, shape (3, 3)
        The Gell-Mann matrix λ_k.
    """
    if k < 1 or k > 8:
        raise ValueError(f"k must be 1–8, got {k}")
    return _ALL_LAMBDAS[k - 1].copy()


def is_gell_mann_valid(matrix: NDArray[np.complexfloating], atol: float = 1e-10) -> bool:
    """
    Check whether a matrix satisfies Gell-Mann properties:
    Hermitian, traceless, and Tr(M²) = 2.
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.shape != (3, 3):
        return False
    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        return False
    if not np.isclose(np.trace(matrix), 0.0, atol=atol):
        return False
    if not np.isclose(np.real(np.trace(matrix @ matrix)), 2.0, atol=atol):
        return False
    return True


def commutator(A: NDArray[np.complexfloating], B: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Commutator [A, B] = AB - BA."""
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    return A @ B - B @ A


def anticommutator(A: NDArray[np.complexfloating], B: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Anticommutator {A, B} = AB + BA."""
    A = np.asarray(A, dtype=np.complex128)
    B = np.asarray(B, dtype=np.complex128)
    return A @ B + B @ A


def structure_constants() -> NDArray[np.floating]:
    """
    Compute the totally antisymmetric structure constants f_{ijk}.

    Defined by [λ_i, λ_j] = 2i Σ_k f_{ijk} λ_k.

    Returns
    -------
    ndarray, shape (8, 8, 8)
        f_{ijk} tensor.
    """
    f = np.zeros((8, 8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            comm = commutator(_ALL_LAMBDAS[i], _ALL_LAMBDAS[j])
            for k in range(8):
                # f_{ijk} = (1/4i) Tr([λ_i, λ_j] λ_k)  [using Tr(λ_a λ_b) = 2δ_{ab}]
                # [λ_i, λ_j] = 2i Σ_m f_{ijm} λ_m
                # Tr([λ_i, λ_j] λ_k) = 2i Σ_m f_{ijm} Tr(λ_m λ_k) = 4i f_{ijk}
                val = np.trace(comm @ _ALL_LAMBDAS[k])
                f[i, j, k] = np.real(val / (4.0j))
    return f
