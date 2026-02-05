"""
Density-matrix simulator for qutrit circuits.

Maintains the full :math:`3^n × 3^n` density matrix and applies gates
as ρ → U ρ U†.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.circuit.measurement import MeasurementResult, sample_from_probabilities
from triflowqc.core.density_matrix import DensityMatrix

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class DensityMatrixSimulator:
    """
    Density-matrix simulator for qutrit circuits (supports mixed states).
    """

    def run(
        self,
        circuit: "QutritCircuit",
        shots: int = 0,
        initial_dm: Optional[NDArray[np.complexfloating]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementResult:
        """
        Execute the circuit using density-matrix simulation.

        Parameters
        ----------
        circuit : QutritCircuit
        shots : int
        initial_dm : ndarray or None
            Initial density matrix. Defaults to |0…0⟩⟨0…0|.

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

        # Build full unitary for each gate and apply ρ → U ρ U†
        for instruction in circuit.instructions:
            matrix = circuit.resolve_gate(instruction)
            _, _, targets, _ = instruction
            U_full = self._expand_gate(matrix, targets, n)
            rho = U_full @ rho @ U_full.conj().T

        probs = np.real(np.diag(rho))
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        if shots > 0:
            counts = sample_from_probabilities(probs, n, shots, rng)
        else:
            counts = {}

        # No pure statevector in density matrix sim
        return MeasurementResult(
            counts=counts,
            probabilities=probs,
            statevector=None,
            shots=shots,
            num_qutrits=n,
        )

    def get_density_matrix(
        self,
        circuit: "QutritCircuit",
        initial_dm: Optional[NDArray[np.complexfloating]] = None,
    ) -> NDArray[np.complexfloating]:
        """
        Execute the circuit and return the final density matrix.

        Parameters
        ----------
        circuit : QutritCircuit
        initial_dm : ndarray or None

        Returns
        -------
        ndarray
            The density matrix after applying all gates.
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
            matrix = circuit.resolve_gate(instruction)
            _, _, targets, _ = instruction
            U_full = self._expand_gate(matrix, targets, n)
            rho = U_full @ rho @ U_full.conj().T

        return rho

    def _expand_gate(
        self,
        gate: NDArray[np.complexfloating],
        targets: list,
        n_qutrits: int,
    ) -> NDArray[np.complexfloating]:
        """Expand a local gate to the full Hilbert space."""
        dim = 3 ** n_qutrits
        k = len(targets)
        size = 3 ** k

        if k == 1 and gate.shape == (3, 3):
            return self._expand_single(gate, targets[0], n_qutrits)

        full = np.zeros((dim, dim), dtype=np.complex128)
        for row in range(dim):
            dr = _index_to_trits(row, n_qutrits)
            for col in range(dim):
                dc = _index_to_trits(col, n_qutrits)
                match = all(dr[i] == dc[i] for i in range(n_qutrits) if i not in targets)
                if not match:
                    continue
                sub_r = sum(dr[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
                sub_c = sum(dc[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
                full[row, col] = gate[sub_r, sub_c]
        return full

    def _expand_single(
        self,
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
