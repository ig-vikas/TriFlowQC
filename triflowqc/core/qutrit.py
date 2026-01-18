"""
Single qutrit state representation and manipulation.

A qutrit lives in 3-dimensional Hilbert space:
    |ψ⟩ = α|0⟩ + β|1⟩ + γ|2⟩
where α, β, γ ∈ ℂ and |α|² + |β|² + |γ|² = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, List


class Qutrit:
    """
    A single qutrit (3-level quantum system).

    Parameters
    ----------
    state : int or array-like, optional
        If int (0, 1, or 2), initialises to the corresponding basis state.
        If array-like of length 3, used as the amplitude vector (will be
        normalised). Default is 0 (the |0⟩ state).

    Examples
    --------
    >>> q = Qutrit()                                  # |0⟩
    >>> q = Qutrit(1)                                 # |1⟩
    >>> q = Qutrit([1, 1, 1])                         # equal superposition (normalised)
    >>> q.probabilities()
    array([0.33333333, 0.33333333, 0.33333333])
    """

    def __init__(self, state: Union[int, List[complex], NDArray[np.complexfloating]] = 0) -> None:
        if isinstance(state, (int, np.integer)):
            if state not in (0, 1, 2):
                raise ValueError(f"Basis state must be 0, 1, or 2, got {state}")
            self._amplitudes = np.zeros(3, dtype=np.complex128)
            self._amplitudes[int(state)] = 1.0 + 0.0j
        else:
            arr = np.asarray(state, dtype=np.complex128)
            if arr.shape != (3,):
                raise ValueError(f"State vector must have length 3, got shape {arr.shape}")
            norm = np.linalg.norm(arr)
            if norm < 1e-15:
                raise ValueError("State vector cannot be zero.")
            self._amplitudes = arr / norm
        self._validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Check normalisation."""
        norm = np.linalg.norm(self._amplitudes)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"State is not normalised (norm={norm:.12f}).")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def amplitudes(self) -> NDArray[np.complexfloating]:
        """Complex amplitude vector of shape (3,)."""
        return self._amplitudes.copy()

    @property
    def state_vector(self) -> NDArray[np.complexfloating]:
        """Alias for amplitudes."""
        return self.amplitudes

    @property
    def dim(self) -> int:
        """Hilbert space dimension (always 3)."""
        return 3

    # ------------------------------------------------------------------
    # Probabilities & measurement
    # ------------------------------------------------------------------
    def probabilities(self) -> NDArray[np.floating]:
        """Return measurement probabilities ``[|α|², |β|², |γ|²]``."""
        return np.abs(self._amplitudes) ** 2

    def measure(self, rng: Optional[np.random.Generator] = None) -> int:
        """
        Perform a projective measurement in the computational basis.

        Returns
        -------
        int
            Outcome 0, 1, or 2 sampled according to Born-rule probabilities.
            The state is **collapsed** to the measured basis state.
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        outcome: int = int(rng.choice(3, p=probs))
        self._amplitudes = np.zeros(3, dtype=np.complex128)
        self._amplitudes[outcome] = 1.0 + 0.0j
        return outcome

    def measure_no_collapse(self, rng: Optional[np.random.Generator] = None) -> int:
        """Sample an outcome without collapsing the state."""
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        return int(rng.choice(3, p=probs))

    # ------------------------------------------------------------------
    # Density matrix
    # ------------------------------------------------------------------
    def density_matrix(self) -> NDArray[np.complexfloating]:
        """Return the 3×3 density matrix |ψ⟩⟨ψ|."""
        return np.outer(self._amplitudes, np.conj(self._amplitudes))

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------
    def apply_gate(self, U: NDArray[np.complexfloating]) -> "Qutrit":
        """
        Apply a 3×3 unitary gate to this qutrit (in-place).

        Parameters
        ----------
        U : ndarray, shape (3, 3)
            Unitary matrix.

        Returns
        -------
        Qutrit
            Self, for chaining.
        """
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (3, 3):
            raise ValueError(f"Gate must be 3×3, got {U.shape}")
        # Quick unitarity check
        if not np.allclose(U @ U.conj().T, np.eye(3), atol=1e-8):
            raise ValueError("Gate is not unitary.")
        self._amplitudes = U @ self._amplitudes
        # Re-normalise to guard against floating-point drift
        self._amplitudes /= np.linalg.norm(self._amplitudes)
        return self

    # ------------------------------------------------------------------
    # Fidelity
    # ------------------------------------------------------------------
    def fidelity(self, other: "Qutrit") -> float:
        """
        State fidelity |⟨ψ|φ⟩|² between this qutrit and *other*.

        Returns
        -------
        float
            Value in [0, 1].
        """
        return float(np.abs(np.vdot(self._amplitudes, other._amplitudes)) ** 2)

    # ------------------------------------------------------------------
    # Bloch vector (generalised)
    # ------------------------------------------------------------------
    def bloch_vector(self) -> NDArray[np.floating]:
        """
        Return the 8-dimensional generalised Bloch vector.

        For a qutrit state ρ = (I + √3 Σ_k r_k λ_k) / 3,
        the components are r_k = (√3/2) Tr(ρ λ_k).
        """
        from triflowqc.gates.gell_mann import gell_mann

        rho = self.density_matrix()
        vec = np.zeros(8)
        for k in range(1, 9):
            lam = gell_mann(k)
            vec[k - 1] = np.real(np.sqrt(3.0) / 2.0 * np.trace(rho @ lam))
        return vec

    # ------------------------------------------------------------------
    # Copy & equality
    # ------------------------------------------------------------------
    def copy(self) -> "Qutrit":
        """Return a deep copy."""
        q = Qutrit.__new__(Qutrit)
        q._amplitudes = self._amplitudes.copy()
        return q

    # ------------------------------------------------------------------
    # String representations
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts: list[str] = []
        labels = ["|0⟩", "|1⟩", "|2⟩"]
        for amp, label in zip(self._amplitudes, labels):
            re, im = amp.real, amp.imag
            if np.abs(amp) < 1e-12:
                continue
            if np.abs(im) < 1e-12:
                parts.append(f"{re:+.4f}{label}")
            elif np.abs(re) < 1e-12:
                parts.append(f"{im:+.4f}i{label}")
            else:
                parts.append(f"({re:+.4f}{im:+.4f}i){label}")
        s = " ".join(parts) if parts else "0"
        # Strip leading '+'
        if s.startswith("+"):
            s = s[1:]
        return f"Qutrit({s.strip()})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Qutrit):
            return NotImplemented
        return bool(np.allclose(self._amplitudes, other._amplitudes, atol=1e-10))
