"""
Density matrix representation for qutrit systems.

A density matrix ρ is a positive semidefinite, Hermitian, trace-1 operator.
For *n* qutrits it has shape :math:`(3^n, 3^n)`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence


class DensityMatrix:
    """
    Density matrix for an *n*-qutrit system.

    Parameters
    ----------
    data : ndarray, shape (d, d)
        Density matrix. Must be Hermitian, positive-semidefinite, trace ≈ 1.
    num_qutrits : int or None
        Number of qutrits. Inferred from shape if *None*.
    validate : bool
        Whether to validate properties at construction.
    """

    def __init__(
        self,
        data: NDArray[np.complexfloating],
        num_qutrits: Optional[int] = None,
        validate: bool = True,
    ) -> None:
        self._rho = np.asarray(data, dtype=np.complex128)
        d = self._rho.shape[0]
        if self._rho.shape != (d, d):
            raise ValueError(f"Density matrix must be square, got {self._rho.shape}")

        if num_qutrits is not None:
            self._n = num_qutrits
            if 3 ** self._n != d:
                raise ValueError(f"3^{self._n} = {3**self._n} ≠ {d}")
        else:
            n = 0
            tmp = d
            while tmp > 1:
                if tmp % 3 != 0:
                    raise ValueError(f"Dimension {d} is not a power of 3.")
                tmp //= 3
                n += 1
            self._n = n

        if validate:
            self._validate()

    def _validate(self) -> None:
        """Check Hermitian, positive-semidefinite, trace = 1."""
        if not np.allclose(self._rho, self._rho.conj().T, atol=1e-9):
            raise ValueError("Density matrix is not Hermitian.")
        eigvals = np.linalg.eigvalsh(self._rho)
        if np.any(eigvals < -1e-9):
            raise ValueError("Density matrix is not positive semidefinite.")
        tr = np.real(np.trace(self._rho))
        if not np.isclose(tr, 1.0, atol=1e-8):
            raise ValueError(f"Trace = {tr:.8f}, expected 1.0.")

    @classmethod
    def from_state_vector(
        cls, sv: NDArray[np.complexfloating], num_qutrits: Optional[int] = None
    ) -> "DensityMatrix":
        """Create a pure-state density matrix |ψ⟩⟨ψ|."""
        sv = np.asarray(sv, dtype=np.complex128).ravel()
        rho = np.outer(sv, np.conj(sv))
        return cls(rho, num_qutrits=num_qutrits)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def matrix(self) -> NDArray[np.complexfloating]:
        return self._rho.copy()

    @property
    def num_qutrits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._rho.shape[0]

    @property
    def purity(self) -> float:
        """Tr(ρ²). Equals 1 for pure states, < 1 for mixed."""
        return float(np.real(np.trace(self._rho @ self._rho)))

    def is_pure(self, atol: float = 1e-8) -> bool:
        """Check if the state is pure (purity ≈ 1)."""
        return bool(np.isclose(self.purity, 1.0, atol=atol))

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def evolve(self, U: NDArray[np.complexfloating]) -> "DensityMatrix":
        """Unitary evolution: ρ → UρU†."""
        U = np.asarray(U, dtype=np.complex128)
        new_rho = U @ self._rho @ U.conj().T
        return DensityMatrix(new_rho, num_qutrits=self._n, validate=False)

    def apply_channel(
        self, kraus_ops: Sequence[NDArray[np.complexfloating]]
    ) -> "DensityMatrix":
        """Apply a quantum channel ρ → Σ_k K_k ρ K_k†."""
        new_rho = np.zeros_like(self._rho)
        for K in kraus_ops:
            K = np.asarray(K, dtype=np.complex128)
            new_rho += K @ self._rho @ K.conj().T
        return DensityMatrix(new_rho, num_qutrits=self._n, validate=False)

    def probabilities(self) -> NDArray[np.floating]:
        """Diagonal elements (measurement probabilities)."""
        return np.real(np.diag(self._rho))

    def expectation(self, observable: NDArray[np.complexfloating]) -> complex:
        """Tr(ρ O)."""
        return complex(np.trace(self._rho @ np.asarray(observable, dtype=np.complex128)))

    def von_neumann_entropy(self) -> float:
        """S(ρ) = -Tr(ρ log ρ). Uses natural logarithm."""
        eigvals = np.linalg.eigvalsh(self._rho)
        eigvals = eigvals[eigvals > 1e-15]
        return float(-np.sum(eigvals * np.log(eigvals)))

    def fidelity(self, other: "DensityMatrix") -> float:
        """
        Fidelity F(ρ, σ) = (Tr √(√ρ σ √ρ))².

        For a pure state ρ = |ψ⟩⟨ψ| this reduces to ⟨ψ|σ|ψ⟩.
        """
        sqrt_rho = _matrix_sqrt(self._rho)
        product = sqrt_rho @ other._rho @ sqrt_rho
        sqrt_product = _matrix_sqrt(product)
        return float(np.real(np.trace(sqrt_product)) ** 2)

    def partial_trace(self, keep: Sequence[int]) -> "DensityMatrix":
        """Trace out all qutrits not in *keep*."""
        keep = sorted(keep)
        trace_out = sorted(set(range(self._n)) - set(keep))
        k = len(keep)
        dim_keep = 3 ** k
        rho_reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        n = self._n

        for r in range(dim_keep):
            trits_r = _index_to_trits(r, k)
            for c in range(dim_keep):
                trits_c = _index_to_trits(c, k)
                val = 0.0 + 0.0j
                dim_traced = 3 ** len(trace_out)
                for t in range(dim_traced):
                    trits_t = _index_to_trits(t, len(trace_out))
                    full_row = [0] * n
                    full_col = [0] * n
                    for idx_k, qk in enumerate(keep):
                        full_row[qk] = trits_r[idx_k]
                        full_col[qk] = trits_c[idx_k]
                    for idx_t, qt in enumerate(trace_out):
                        full_row[qt] = trits_t[idx_t]
                        full_col[qt] = trits_t[idx_t]
                    fr = _trits_to_index(full_row)
                    fc = _trits_to_index(full_col)
                    val += self._rho[fr, fc]
                rho_reduced[r, c] = val
        return DensityMatrix(rho_reduced, num_qutrits=k, validate=False)

    def __repr__(self) -> str:
        return f"DensityMatrix(n={self._n}, dim={self.dim}, purity={self.purity:.4f})"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _matrix_sqrt(M: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """Compute matrix square root of a positive-semidefinite matrix."""
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 0.0)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T


def _index_to_trits(index: int, n: int) -> list:
    digits = []
    for _ in range(n):
        digits.append(index % 3)
        index //= 3
    return list(reversed(digits))


def _trits_to_index(trits: Sequence[int]) -> int:
    idx = 0
    for t in trits:
        idx = idx * 3 + t
    return idx
