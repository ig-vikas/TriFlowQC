"""
Noisy density-matrix simulator.

Combines the DensityMatrixSimulator with a NoiseModel to apply Kraus
channels after each gate and readout errors before measurement sampling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.circuit.measurement import MeasurementResult, sample_from_probabilities
from triflowqc.simulator.noise_model import NoiseModel

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class NoisySimulator:
    """
    Density-matrix simulator that injects noise after each gate.

    Parameters
    ----------
    noise_model : NoiseModel
        The noise model describing errors.
    """

    def __init__(self, noise_model: NoiseModel):
        self.noise_model = noise_model

    def run(
        self,
        circuit: "QutritCircuit",
        shots: int = 1024,
        initial_dm: Optional[NDArray[np.complexfloating]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementResult:
        """
        Execute the circuit with noise injection.

        Parameters
        ----------
        circuit : QutritCircuit
        shots : int
            Number of measurement shots (must be >0 for noisy sim).
        initial_dm : ndarray or None
        rng : numpy Generator

        Returns
        -------
        MeasurementResult
        """
        n = circuit.num_qutrits
        dim = 3 ** n

        if initial_dm is not None:
            rho = np.asarray(initial_dm, dtype=np.complex128)
        else:
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0
            rho = np.outer(state, np.conj(state))

        for instruction in circuit.instructions:
            name, _, targets, _ = instruction
            matrix = circuit.resolve_gate(instruction)

            # Apply ideal gate: ρ → U ρ U†
            U_full = _expand_gate(matrix, targets, n)
            rho = U_full @ rho @ U_full.conj().T

            # Apply noise channel if any
            noise_ops = self.noise_model.get_gate_noise(name, targets)
            if noise_ops is not None:
                rho = _apply_noise_channel(rho, noise_ops, targets, n)

        # Extract ideal probabilities
        probs = np.real(np.diag(rho))
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        # Apply readout error
        probs = self._apply_readout_error(probs, n)

        if shots > 0:
            counts = sample_from_probabilities(probs, n, shots, rng)
        else:
            counts = {}

        return MeasurementResult(
            counts=counts,
            probabilities=probs,
            statevector=None,
            shots=shots,
            num_qutrits=n,
        )

    def _apply_readout_error(
        self, probs: NDArray[np.float64], n: int
    ) -> NDArray[np.float64]:
        """
        Apply per-qutrit readout error.

        For each qutrit, the readout matrix maps the ideal marginal distribution
        to a noisy one. We construct the full readout confusion matrix as the
        Kronecker product of per-qutrit confusion matrices.
        """
        has_any = False
        matrices = []
        for q in range(n):
            M = self.noise_model.get_readout_matrix(q)
            if M is not None:
                has_any = True
                matrices.append(M)
            else:
                matrices.append(np.eye(3, dtype=np.float64))

        if not has_any:
            return probs

        # Build full confusion matrix via Kronecker product
        full_cm = matrices[0]
        for M in matrices[1:]:
            full_cm = np.kron(full_cm, M)

        # Apply: p_noisy[j] = Σ_i full_cm[i, j] * p_ideal[i]
        noisy = full_cm.T @ probs
        noisy = np.maximum(noisy, 0.0)
        total = noisy.sum()
        if total > 0:
            noisy /= total
        return noisy

    def get_density_matrix(
        self,
        circuit: "QutritCircuit",
        initial_dm: Optional[NDArray[np.complexfloating]] = None,
    ) -> NDArray[np.complexfloating]:
        """
        Execute the circuit with noise and return the final density matrix.

        Parameters
        ----------
        circuit : QutritCircuit
        initial_dm : ndarray or None

        Returns
        -------
        ndarray
            The density matrix after applying all gates and noise channels.
        """
        n = circuit.num_qutrits
        dim = 3 ** n

        if initial_dm is not None:
            rho = np.asarray(initial_dm, dtype=np.complex128)
        else:
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0
            rho = np.outer(state, np.conj(state))

        for instruction in circuit.instructions:
            name, _, targets, _ = instruction
            matrix = circuit.resolve_gate(instruction)
            U_full = _expand_gate(matrix, targets, n)
            rho = U_full @ rho @ U_full.conj().T

            noise_ops = self.noise_model.get_gate_noise(name, targets)
            if noise_ops is not None:
                rho = _apply_noise_channel(rho, noise_ops, targets, n)

        return rho


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def _apply_noise_channel(
    rho: NDArray[np.complexfloating],
    kraus_ops: list,
    targets: list,
    n: int,
) -> NDArray[np.complexfloating]:
    """
    Apply a single-qutrit Kraus channel to the density matrix.

    For multi-qutrit noise, the Kraus operators are expanded and applied
    element-wise.
    """
    k_dim = kraus_ops[0].shape[0]
    dim = 3 ** n

    if k_dim == 3 and len(targets) >= 1:
        # Single-qutrit noise → apply to each target qutrit independently
        for t in targets:
            new_rho = np.zeros_like(rho)
            for K in kraus_ops:
                K_full = _expand_single(K, t, n)
                new_rho += K_full @ rho @ K_full.conj().T
            rho = new_rho
    elif k_dim == dim:
        # Full-space Kraus operators
        new_rho = np.zeros_like(rho)
        for K in kraus_ops:
            new_rho += K @ rho @ K.conj().T
        rho = new_rho
    else:
        # Multi-qutrit local noise
        new_rho = np.zeros_like(rho)
        for K in kraus_ops:
            K_full = _expand_gate(K, targets, n)
            new_rho += K_full @ rho @ K_full.conj().T
        rho = new_rho

    return rho


def _expand_gate(
    gate: NDArray[np.complexfloating],
    targets: list,
    n_qutrits: int,
) -> NDArray[np.complexfloating]:
    """Expand a local gate to the full Hilbert space."""
    dim = 3 ** n_qutrits
    k = len(targets)

    if k == 1 and gate.shape == (3, 3):
        return _expand_single(gate, targets[0], n_qutrits)

    full = np.zeros((dim, dim), dtype=np.complex128)
    for row in range(dim):
        dr = _index_to_trits(row, n_qutrits)
        for col in range(dim):
            dc = _index_to_trits(col, n_qutrits)
            match = all(
                dr[i] == dc[i] for i in range(n_qutrits) if i not in targets
            )
            if not match:
                continue
            sub_r = sum(dr[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
            sub_c = sum(dc[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
            full[row, col] = gate[sub_r, sub_c]
    return full


def _expand_single(
    gate: NDArray[np.complexfloating],
    target: int,
    n: int,
) -> NDArray[np.complexfloating]:
    """Expand single-qutrit gate using Kronecker products."""
    ops = []
    for i in range(n):
        if i == target:
            ops.append(gate)
        else:
            ops.append(np.eye(3, dtype=np.complex128))
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _index_to_trits(index: int, n: int) -> list:
    digits = []
    for _ in range(n):
        digits.append(index % 3)
        index //= 3
    return list(reversed(digits))
