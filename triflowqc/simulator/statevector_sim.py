"""
State-vector simulator for qutrit circuits.

Maintains the full :math:`3^n` complex state vector and applies gates
as matrix–vector multiplications via efficient tensor contraction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, TYPE_CHECKING

from triflowqc.circuit.measurement import MeasurementResult, sample_from_probabilities

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class StatevectorSimulator:
    """
    Exact state-vector simulator for qutrit circuits.

    Examples
    --------
    >>> sim = StatevectorSimulator()
    >>> result = sim.run(circuit, shots=1024)
    >>> result.counts
    {'00': 512, '01': 256, '10': 256}
    """

    def run(
        self,
        circuit: "QutritCircuit",
        shots: int = 0,
        initial_state: Optional[NDArray[np.complexfloating]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> MeasurementResult:
        """
        Execute the circuit.

        Parameters
        ----------
        circuit : QutritCircuit
        shots : int
            Number of measurement shots. If 0, return exact probabilities.
        initial_state : ndarray or None
            Custom initial state vector. If None, start from |00…0⟩.
        rng : numpy random generator

        Returns
        -------
        MeasurementResult
        """
        n = circuit.num_qutrits
        dim = 3 ** n

        if initial_state is not None:
            state = np.asarray(initial_state, dtype=np.complex128).ravel()
            if state.shape[0] != dim:
                raise ValueError(f"Initial state length must be {dim}")
            state = state / np.linalg.norm(state)
        else:
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1.0 + 0.0j

        # Apply each gate
        for instruction in circuit.instructions:
            matrix = circuit.resolve_gate(instruction)
            _, _, targets, _ = instruction

            if len(targets) == 1:
                state = self._apply_single(state, matrix, targets[0], n)
            elif len(targets) == 2:
                state = self._apply_two(state, matrix, targets, n)
            else:
                state = self._apply_multi(state, matrix, targets, n)

        # Probabilities
        probs = np.abs(state) ** 2
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total

        # Sampling
        if shots > 0:
            counts = sample_from_probabilities(probs, n, shots, rng)
        else:
            counts = {}

        return MeasurementResult(
            counts=counts,
            probabilities=probs,
            statevector=state,
            shots=shots,
            num_qutrits=n,
        )

    def _apply_single(
        self,
        state: NDArray[np.complexfloating],
        gate: NDArray[np.complexfloating],
        target: int,
        n: int,
    ) -> NDArray[np.complexfloating]:
        """Apply 3×3 gate to target qutrit via tensor contraction."""
        tensor = state.reshape([3] * n)
        tensor = np.tensordot(gate, tensor, axes=([1], [target]))
        tensor = np.moveaxis(tensor, 0, target)
        return tensor.reshape(3 ** n)

    def _apply_two(
        self,
        state: NDArray[np.complexfloating],
        gate: NDArray[np.complexfloating],
        targets: list,
        n: int,
    ) -> NDArray[np.complexfloating]:
        """Apply 9×9 gate to two target qutrits."""
        q0, q1 = targets
        gate_tensor = gate.reshape(3, 3, 3, 3)
        tensor = state.reshape([3] * n)

        # Move target axes to the end
        axes_order = [i for i in range(n) if i not in (q0, q1)] + [q0, q1]
        inv_order = [0] * n
        for new_pos, old_pos in enumerate(axes_order):
            inv_order[old_pos] = new_pos

        tensor = np.transpose(tensor, axes_order)
        old_shape = tensor.shape
        batch = int(np.prod(old_shape[:-2])) if n > 2 else 1
        tensor = tensor.reshape(batch, 3, 3)
        result = np.einsum("abcd,xcd->xab", gate_tensor, tensor)
        result = result.reshape(old_shape)
        tensor = np.transpose(result, [inv_order[i] for i in range(n)])
        return tensor.reshape(3 ** n)

    def _apply_multi(
        self,
        state: NDArray[np.complexfloating],
        gate: NDArray[np.complexfloating],
        targets: list,
        n: int,
    ) -> NDArray[np.complexfloating]:
        """Apply multi-qutrit gate via full matrix multiplication."""
        dim = 3 ** n
        k = len(targets)
        size = 3 ** k
        full = np.zeros((dim, dim), dtype=np.complex128)

        for row in range(dim):
            digits_row = _index_to_trits(row, n)
            for col in range(dim):
                digits_col = _index_to_trits(col, n)
                match = all(
                    digits_row[i] == digits_col[i]
                    for i in range(n)
                    if i not in targets
                )
                if not match:
                    continue
                sub_row = sum(
                    digits_row[targets[j]] * (3 ** (k - 1 - j)) for j in range(k)
                )
                sub_col = sum(
                    digits_col[targets[j]] * (3 ** (k - 1 - j)) for j in range(k)
                )
                full[row, col] = gate[sub_row, sub_col]
        return full @ state


def _index_to_trits(index: int, n: int) -> list:
    digits = []
    for _ in range(n):
        digits.append(index % 3)
        index //= 3
    return list(reversed(digits))
