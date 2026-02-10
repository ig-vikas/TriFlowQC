"""
Gate decomposition utilities.

Decomposes embedded qutrit unitaries into sequences of one- and two-qubit
gates that are native to IBM Quantum hardware.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Optional

try:
    from qiskit.circuit import QuantumCircuit as QiskitCircuit
    from qiskit.quantum_info import Operator, OneQubitEulerDecomposer
    from qiskit import transpile
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from triflowqc.compiler.qutrit_to_qubit import (
    embed_single_qutrit_gate,
    embed_two_qutrit_gate,
)


class GateDecomposer:
    """
    Decomposes embedded qutrit-to-qubit unitaries into native gate sequences.

    Supports:
    - Single-qutrit → 2-qubit decomposition (KAK / Euler)
    - Two-qutrit → 4-qubit decomposition (via Qiskit transpiler)
    - Custom basis gate targeting

    Parameters
    ----------
    basis_gates : list of str
        Target basis gates for decomposition.
    optimization_level : int
        Qiskit optimization level (0–3).
    """

    DEFAULT_BASIS = ["cx", "u3", "u2", "u1", "id"]

    def __init__(
        self,
        basis_gates: Optional[List[str]] = None,
        optimization_level: int = 2,
    ):
        if not HAS_QISKIT:
            raise ImportError("Qiskit required for gate decomposition")
        self.basis_gates = basis_gates or self.DEFAULT_BASIS
        self.optimization_level = optimization_level

    def decompose_single_qutrit(
        self,
        gate_3x3: NDArray[np.complexfloating],
        qubits: Tuple[int, int] = (0, 1),
    ) -> "QiskitCircuit":
        """
        Decompose a single-qutrit gate (3×3) into a 2-qubit circuit.

        Parameters
        ----------
        gate_3x3 : ndarray (3×3)
        qubits : tuple of 2 ints

        Returns
        -------
        QiskitCircuit
        """
        V = embed_single_qutrit_gate(gate_3x3)
        qc = QiskitCircuit(max(qubits) + 1)
        qc.unitary(Operator(V), list(qubits))
        return transpile(
            qc,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
        )

    def decompose_two_qutrit(
        self,
        gate_9x9: NDArray[np.complexfloating],
        qubits: Tuple[int, int, int, int] = (0, 1, 2, 3),
    ) -> "QiskitCircuit":
        """
        Decompose a two-qutrit gate (9×9) into a 4-qubit circuit.

        Parameters
        ----------
        gate_9x9 : ndarray (9×9)
        qubits : tuple of 4 ints

        Returns
        -------
        QiskitCircuit
        """
        V = embed_two_qutrit_gate(gate_9x9)
        qc = QiskitCircuit(max(qubits) + 1)
        qc.unitary(Operator(V), list(qubits))
        return transpile(
            qc,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
        )

    def decompose_arbitrary(
        self,
        unitary: NDArray[np.complexfloating],
        qubits: List[int],
    ) -> "QiskitCircuit":
        """
        Decompose an arbitrary unitary via Qiskit transpiler.

        Parameters
        ----------
        unitary : ndarray
            Square unitary matrix.
        qubits : list of int
            Target qubit indices.

        Returns
        -------
        QiskitCircuit
        """
        n = max(qubits) + 1
        qc = QiskitCircuit(n)
        qc.unitary(Operator(unitary), qubits)
        return transpile(
            qc,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
        )

    def gate_count(self, qc: "QiskitCircuit") -> Dict[str, int]:
        """
        Count gates by type in a compiled circuit.

        Returns
        -------
        dict
            Gate name → count.
        """
        counts: Dict[str, int] = {}
        for instruction in qc.data:
            name = instruction.operation.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    def cx_count(self, qc: "QiskitCircuit") -> int:
        """Return the number of CNOT (cx) gates."""
        return self.gate_count(qc).get("cx", 0)

    def depth(self, qc: "QiskitCircuit") -> int:
        """Return the circuit depth."""
        return qc.depth()

    def decompose_full_circuit(
        self,
        qutrit_circuit: "QiskitCircuit",
    ) -> "QiskitCircuit":
        """
        Take an already-compiled (embedded) Qiskit circuit and decompose
        all unitaries into the basis gate set.
        """
        return transpile(
            qutrit_circuit,
            basis_gates=self.basis_gates,
            optimization_level=self.optimization_level,
        )
