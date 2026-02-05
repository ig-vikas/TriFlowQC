"""
Noise models for qutrit quantum systems.

Provides Kraus-operator based noise channels including depolarising,
dephasing, amplitude damping, and custom channels — all generalised to
dimension 3 (qutrits).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Dict
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Kraus channel helpers
# ---------------------------------------------------------------------------

def _validate_kraus(ops: List[NDArray[np.complexfloating]], rtol: float = 1e-6) -> None:
    """Check completeness: Σ K†K = I."""
    d = ops[0].shape[0]
    total = sum(K.conj().T @ K for K in ops)
    if not np.allclose(total, np.eye(d, dtype=np.complex128), atol=rtol):
        raise ValueError("Kraus operators do not satisfy completeness relation.")


# ---------------------------------------------------------------------------
# Single-qutrit noise channels
# ---------------------------------------------------------------------------

def depolarising_channel(p: float) -> List[NDArray[np.complexfloating]]:
    r"""
    Qutrit depolarising channel.

    .. math::
        \mathcal{E}(\rho) = (1 - p)\rho + \frac{p}{8}\sum_{k=1}^{8}\lambda_k\rho\lambda_k

    Parameters
    ----------
    p : float
        Depolarising probability, 0 ≤ p ≤ 1.

    Returns
    -------
    list of ndarray
        9 Kraus operators.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    from triflowqc.gates.gell_mann import LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4
    from triflowqc.gates.gell_mann import LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8

    lambdas = [LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
               LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8]

    K0 = np.sqrt(1.0 - p) * np.eye(3, dtype=np.complex128)
    ops = [K0]
    for lam in lambdas:
        ops.append(np.sqrt(p / 8.0) * lam)
    return ops


def dephasing_channel(p: float) -> List[NDArray[np.complexfloating]]:
    r"""
    Qutrit dephasing channel.

    Damps off-diagonal elements. Uses the diagonal Gell-Mann matrices
    (λ₃ and λ₈) and the identity.

    .. math::
        \mathcal{E}(\rho) = (1-p)\rho + \frac{p}{2}(\lambda_3\rho\lambda_3
                            + \lambda_8\rho\lambda_8)

    Parameters
    ----------
    p : float
        Dephasing probability, 0 ≤ p ≤ 1.

    Returns
    -------
    list of ndarray
        3 Kraus operators.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    from triflowqc.gates.gell_mann import LAMBDA_3, LAMBDA_8

    K0 = np.sqrt(1.0 - p) * np.eye(3, dtype=np.complex128)
    K1 = np.sqrt(p / 2.0) * LAMBDA_3
    K2 = np.sqrt(p / 2.0) * LAMBDA_8
    return [K0, K1, K2]


def amplitude_damping_channel(
    gamma_10: float,
    gamma_20: float,
    gamma_21: float = 0.0,
) -> List[NDArray[np.complexfloating]]:
    r"""
    Qutrit amplitude damping channel.

    Generalises the qubit amplitude damping to three decay pathways:
    |1⟩→|0⟩ with rate γ₁₀, |2⟩→|0⟩ with rate γ₂₀, |2⟩→|1⟩ with rate γ₂₁.

    Parameters
    ----------
    gamma_10 : float
        Decay rate from |1⟩ to |0⟩.
    gamma_20 : float
        Decay rate from |2⟩ to |0⟩.
    gamma_21 : float
        Decay rate from |2⟩ to |1⟩.

    Returns
    -------
    list of ndarray
    """
    for name, val in [("gamma_10", gamma_10), ("gamma_20", gamma_20), ("gamma_21", gamma_21)]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    # K0: no-jump operator
    K0 = np.diag([
        1.0,
        np.sqrt(1.0 - gamma_10),
        np.sqrt(max(1.0 - gamma_20 - gamma_21, 0.0)),
    ]).astype(np.complex128)

    # K1: |1⟩ → |0⟩
    K1 = np.zeros((3, 3), dtype=np.complex128)
    K1[0, 1] = np.sqrt(gamma_10)

    # K2: |2⟩ → |0⟩
    K2 = np.zeros((3, 3), dtype=np.complex128)
    K2[0, 2] = np.sqrt(gamma_20)

    # K3: |2⟩ → |1⟩
    K3 = np.zeros((3, 3), dtype=np.complex128)
    K3[1, 2] = np.sqrt(gamma_21)

    ops = [K0, K1, K2, K3]
    # Remove zero operators
    ops = [K for K in ops if np.linalg.norm(K) > 1e-15]
    return ops


def trit_flip_channel(p: float) -> List[NDArray[np.complexfloating]]:
    r"""
    Qutrit trit-flip channel (analogous to qubit bit-flip).

    With probability p/2, applies X₃ or X₃², cyclically permuting
    computational basis states.

    Parameters
    ----------
    p : float
        Flip probability, 0 ≤ p ≤ 1.

    Returns
    -------
    list of ndarray
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    from triflowqc.gates.single_qutrit import X3

    X3_sq = X3 @ X3
    K0 = np.sqrt(1.0 - p) * np.eye(3, dtype=np.complex128)
    K1 = np.sqrt(p / 2.0) * X3
    K2 = np.sqrt(p / 2.0) * X3_sq
    return [K0, K1, K2]


def custom_kraus_channel(
    ops: List[NDArray[np.complexfloating]],
    validate: bool = True,
) -> List[NDArray[np.complexfloating]]:
    """Wrap arbitrary Kraus operators as a channel, optionally validating."""
    ops = [np.asarray(K, dtype=np.complex128) for K in ops]
    if validate:
        _validate_kraus(ops)
    return ops


# ---------------------------------------------------------------------------
# Noise Model
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    """
    Noise model that maps gate names / qutrits to noise channels.

    Usage
    -----
    >>> nm = NoiseModel()
    >>> nm.add_all_qutrit_depolarising(0.01)
    >>> nm.add_readout_error([[0.95, 0.03, 0.02], [0.02, 0.95, 0.03], [0.01, 0.04, 0.95]])
    """

    # Gate errors: (gate_name | "all") -> list of Kraus ops applied after gate
    _gate_errors: Dict[str, List[NDArray[np.complexfloating]]] = field(
        default_factory=dict, repr=False
    )

    # Qutrit-specific gate errors: (qutrit_index, gate_name) -> Kraus
    _local_errors: Dict[tuple, List[NDArray[np.complexfloating]]] = field(
        default_factory=dict, repr=False
    )

    # Readout error: 3×3 stochastic matrix per qutrit, or single global
    _readout_errors: Dict[int, NDArray[np.float64]] = field(
        default_factory=dict, repr=False
    )

    _global_readout: Optional[NDArray[np.float64]] = field(default=None, repr=False)

    def add_all_qutrit_depolarising(self, p: float) -> None:
        """Add depolarising noise after every gate."""
        self._gate_errors["all"] = depolarising_channel(p)

    def add_all_qutrit_dephasing(self, p: float) -> None:
        """Add dephasing noise after every gate."""
        self._gate_errors["all_dephasing"] = dephasing_channel(p)

    def add_gate_error(
        self,
        gate_name: str,
        kraus_ops: List[NDArray[np.complexfloating]],
    ) -> None:
        """Add noise after a specific gate type."""
        self._gate_errors[gate_name] = kraus_ops

    def add_local_error(
        self,
        qutrit: int,
        gate_name: str,
        kraus_ops: List[NDArray[np.complexfloating]],
    ) -> None:
        """Add noise after a gate on a specific qutrit."""
        self._local_errors[(qutrit, gate_name)] = kraus_ops

    def add_readout_error(
        self,
        matrix: NDArray[np.float64],
        qutrit: Optional[int] = None,
    ) -> None:
        """
        Add readout (measurement) error.

        Parameters
        ----------
        matrix : ndarray (3×3)
            Row-stochastic matrix: ``matrix[i, j]`` = Pr(measure j | true state i).
        qutrit : int or None
            If None, apply to all qutrits.
        """
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("Readout matrix must be 3×3")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Readout matrix rows must sum to 1")
        if qutrit is None:
            self._global_readout = matrix
        else:
            self._readout_errors[qutrit] = matrix

    def get_gate_noise(
        self, gate_name: str, targets: list
    ) -> Optional[List[NDArray[np.complexfloating]]]:
        """Get noise channel for a gate application."""
        # Check local errors first
        for t in targets:
            key = (t, gate_name)
            if key in self._local_errors:
                return self._local_errors[key]

        # Named gate error
        if gate_name in self._gate_errors:
            return self._gate_errors[gate_name]

        # Global "all" error
        if "all" in self._gate_errors:
            return self._gate_errors["all"]

        return None

    def get_readout_matrix(self, qutrit: int) -> Optional[NDArray[np.float64]]:
        """Get readout error matrix for a specific qutrit."""
        if qutrit in self._readout_errors:
            return self._readout_errors[qutrit]
        return self._global_readout

    def has_noise(self) -> bool:
        return bool(
            self._gate_errors
            or self._local_errors
            or self._readout_errors
            or self._global_readout is not None
        )

    def __repr__(self) -> str:
        parts = []
        if self._gate_errors:
            parts.append(f"gate_errors={list(self._gate_errors.keys())}")
        if self._local_errors:
            parts.append(f"local_errors={len(self._local_errors)}")
        if self._readout_errors or self._global_readout is not None:
            parts.append("readout_error=True")
        return f"NoiseModel({', '.join(parts)})"
