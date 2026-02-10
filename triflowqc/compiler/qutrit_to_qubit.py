"""
Qutrit-to-qubit compiler.

Encoding scheme (per qutrit → 2 physical qubits):
    |0⟩_t  →  |00⟩_q
    |1⟩_t  →  |01⟩_q
    |2⟩_t  →  |10⟩_q
    |11⟩_q  →  FORBIDDEN (leakage)

A qutrit gate U (3×3) is embedded into the 4×4 space by placing U in the
upper-left 3×3 block and setting the (3,3) entry to 1 (identity on the
forbidden |11⟩ state).  Two-qutrit gates (9×9) are similarly embedded into
16×16 = (4⊗4).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit

try:
    from qiskit.circuit import QuantumCircuit as QiskitCircuit
    from qiskit.quantum_info import Operator
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

# Qutrit basis → qubit pair mapping
ENCODING: Dict[int, Tuple[int, int]] = {0: (0, 0), 1: (0, 1), 2: (1, 0)}
FORBIDDEN_STATE = (1, 1)  # |11⟩


def embed_single_qutrit_gate(U: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Embed a 3×3 qutrit unitary into a 4×4 two-qubit unitary.

    The forbidden |11⟩ state maps to itself (identity).
    """
    assert U.shape == (3, 3), f"Expected 3×3, got {U.shape}"
    V = np.eye(4, dtype=np.complex128)
    # Map qutrit states 0,1,2 to qubit states 00,01,10 (indices 0,1,2 in 4-dim)
    for i in range(3):
        for j in range(3):
            V[i, j] = U[i, j]
    return V


def embed_two_qutrit_gate(U: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Embed a 9×9 two-qutrit unitary into a 16×16 four-qubit unitary.

    The encoding maps each qutrit to 2 qubits, so 2 qutrits = 4 qubits.
    States that involve any |11⟩ qubit pair are mapped to themselves (identity).
    """
    assert U.shape == (9, 9), f"Expected 9×9, got {U.shape}"
    V = np.eye(16, dtype=np.complex128)

    # Build the mapping: 2-qutrit state (t0, t1) → 4-qubit state index
    # Qutrit state (t0, t1) → qubit state (q0a, q0b, q1a, q1b)
    for t0_in in range(3):
        for t1_in in range(3):
            q0a_in, q0b_in = ENCODING[t0_in]
            q1a_in, q1b_in = ENCODING[t1_in]
            qubit_in = (q0a_in << 3) | (q0b_in << 2) | (q1a_in << 1) | q1b_in

            for t0_out in range(3):
                for t1_out in range(3):
                    q0a_out, q0b_out = ENCODING[t0_out]
                    q1a_out, q1b_out = ENCODING[t1_out]
                    qubit_out = (q0a_out << 3) | (q0b_out << 2) | (q1a_out << 1) | q1b_out

                    trit_in = t0_in * 3 + t1_in
                    trit_out = t0_out * 3 + t1_out
                    V[qubit_out, qubit_in] = U[trit_out, trit_in]
    return V


def embed_three_qutrit_gate(U: NDArray[np.complexfloating]) -> NDArray[np.complexfloating]:
    """
    Embed a 27×27 three-qutrit unitary into a 64×64 six-qubit unitary.
    """
    assert U.shape == (27, 27), f"Expected 27×27, got {U.shape}"
    V = np.eye(64, dtype=np.complex128)

    for trits_in_val in range(27):
        t0_in = trits_in_val // 9
        t1_in = (trits_in_val // 3) % 3
        t2_in = trits_in_val % 3
        q0a_in, q0b_in = ENCODING[t0_in]
        q1a_in, q1b_in = ENCODING[t1_in]
        q2a_in, q2b_in = ENCODING[t2_in]
        qubit_in = (
            (q0a_in << 5) | (q0b_in << 4)
            | (q1a_in << 3) | (q1b_in << 2)
            | (q2a_in << 1) | q2b_in
        )

        for trits_out_val in range(27):
            t0_out = trits_out_val // 9
            t1_out = (trits_out_val // 3) % 3
            t2_out = trits_out_val % 3
            q0a_out, q0b_out = ENCODING[t0_out]
            q1a_out, q1b_out = ENCODING[t1_out]
            q2a_out, q2b_out = ENCODING[t2_out]
            qubit_out = (
                (q0a_out << 5) | (q0b_out << 4)
                | (q1a_out << 3) | (q1b_out << 2)
                | (q2a_out << 1) | q2b_out
            )
            V[qubit_out, qubit_in] = U[trits_out_val, trits_in_val]

    return V


# ---------------------------------------------------------------------------
# Main compiler class
# ---------------------------------------------------------------------------

class QutritToQubitCompiler:
    """
    Compiles a QutritCircuit into a Qiskit QuantumCircuit.

    Each qutrit is encoded as 2 physical qubits.
    Single-qutrit gates (3×3) → 2-qubit unitaries (4×4).
    Two-qutrit gates (9×9) → 4-qubit unitaries (16×16).
    Three-qutrit gates (27×27) → 6-qubit unitaries (64×64).

    Parameters
    ----------
    leakage_post_select : bool
        If True, add classical post-selection instructions to flag leakage.
    optimization_level : int
        Qiskit transpiler optimization level (0–3) for final decomposition.
    """

    def __init__(
        self,
        leakage_post_select: bool = True,
        optimization_level: int = 1,
    ):
        if not HAS_QISKIT:
            raise ImportError(
                "Qiskit is required for compilation. Install with: pip install qiskit"
            )
        self.leakage_post_select = leakage_post_select
        self.optimization_level = optimization_level

    # ------------------------------------------------------------------
    # Convenience wrappers around module-level embedding functions
    # ------------------------------------------------------------------

    @staticmethod
    def encode_state(
        trit_state: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """
        Encode a single-qutrit state vector into 2-qubit space.

        Maps |0⟩→|00⟩, |1⟩→|01⟩, |2⟩→|10⟩.

        Parameters
        ----------
        trit_state : ndarray, shape (3,)

        Returns
        -------
        ndarray, shape (4,)
        """
        trit_state = np.asarray(trit_state, dtype=np.complex128)
        qubit_state = np.zeros(4, dtype=np.complex128)
        for k in range(3):
            qubit_state[k] = trit_state[k]
        return qubit_state

    @staticmethod
    def embed_single_qutrit_gate(
        U: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """Embed a 3×3 qutrit unitary into 4×4 two-qubit space."""
        return embed_single_qutrit_gate(U)

    @staticmethod
    def embed_two_qutrit_gate(
        U: NDArray[np.complexfloating],
    ) -> NDArray[np.complexfloating]:
        """Embed a 9×9 two-qutrit unitary into 16×16 four-qubit space."""
        return embed_two_qutrit_gate(U)

    def compile(
        self,
        circuit: "QutritCircuit",
        decompose: bool = True,
    ) -> "QiskitCircuit":
        """
        Compile a QutritCircuit to a Qiskit QuantumCircuit.

        Parameters
        ----------
        circuit : QutritCircuit
        decompose : bool
            If True, decompose embedded unitaries into native gates via
            Qiskit's transpiler.

        Returns
        -------
        qiskit.circuit.QuantumCircuit
        """
        n_qutrits = circuit.num_qutrits
        n_qubits = 2 * n_qutrits
        qc = QiskitCircuit(n_qubits, n_qubits)

        for instruction in circuit.instructions:
            name, _, targets, _ = instruction
            matrix = circuit.resolve_gate(instruction)
            n_targets = len(targets)

            # Compute qubit indices
            qubit_indices: List[int] = []
            for t in targets:
                qubit_indices.extend([2 * t, 2 * t + 1])

            if n_targets == 1:
                V = embed_single_qutrit_gate(matrix)
            elif n_targets == 2:
                V = embed_two_qutrit_gate(matrix)
            elif n_targets == 3:
                V = embed_three_qutrit_gate(matrix)
            else:
                raise ValueError(
                    f"Gate on {n_targets} qutrits not supported for compilation"
                )

            qc.unitary(Operator(V), qubit_indices, label=name)

        # Add leakage-detection measurements if requested
        if self.leakage_post_select:
            _add_leakage_checks(qc, n_qutrits)

        # Optionally decompose into native gate set
        if decompose:
            from qiskit import transpile
            qc = transpile(
                qc,
                optimization_level=self.optimization_level,
                basis_gates=["cx", "u3", "u2", "u1", "id"],
            )

        return qc

    def compile_and_bind(
        self,
        circuit: "QutritCircuit",
        parameters: Optional[NDArray[np.floating]] = None,
    ) -> "QiskitCircuit":
        """
        Compile with specific parameter values bound.
        """
        if parameters is not None:
            circuit.set_parameters(parameters)
        return self.compile(circuit, decompose=True)

    def resource_estimate(
        self, circuit: "QutritCircuit"
    ) -> Dict[str, int]:
        """Estimate qubit and gate resources for the compiled circuit."""
        n_qutrits = circuit.num_qutrits
        n_qubits = 2 * n_qutrits
        n_single = sum(1 for _, _, t, _ in circuit.instructions if len(t) == 1)
        n_two = sum(1 for _, _, t, _ in circuit.instructions if len(t) == 2)
        n_three = sum(1 for _, _, t, _ in circuit.instructions if len(t) == 3)
        return {
            "num_qutrits": n_qutrits,
            "num_physical_qubits": n_qubits,
            "num_single_qutrit_gates": n_single,
            "num_two_qutrit_gates": n_two,
            "num_three_qutrit_gates": n_three,
            "estimated_cnots_upper_bound": n_single * 3 + n_two * 15 + n_three * 60,
        }


def _add_leakage_checks(qc: "QiskitCircuit", n_qutrits: int) -> None:
    """
    Add measurements to all qubits. Post-selection for leakage is done
    classically by discarding shots where any qubit pair is |11⟩.
    """
    n_qubits = 2 * n_qutrits
    qc.barrier()
    for i in range(n_qubits):
        qc.measure(i, i)
