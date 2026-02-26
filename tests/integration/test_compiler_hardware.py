"""
Integration tests: Circuit → Compiler → (mock) Hardware pipeline.
"""

import numpy as np
import pytest

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.compiler.qutrit_to_qubit import (
    QutritToQubitCompiler,
    embed_single_qutrit_gate,
    embed_two_qutrit_gate,
)
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper
from triflowqc.gates.single_qutrit import X3, Z3, H3
from triflowqc.gates.two_qutrit import CSUM, SWAP3


class TestCompilerPipeline:
    """End-to-end compiler pipeline tests."""

    def test_single_gate_compile_and_map(self):
        """Compile single X3 gate and verify mapping consistency."""
        qc = QutritCircuit(1)
        qc.x3(0)
        compiler = QutritToQubitCompiler()
        qubit_circuit = compiler.compile(qc)
        assert qubit_circuit is not None
        # Compiled circuit should use 2 qubits for 1 qutrit
        assert qubit_circuit.num_qubits == 2

    def test_two_qutrit_compile(self):
        """Compile a 2-qutrit circuit with CSUM."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        compiler = QutritToQubitCompiler()
        qubit_circuit = compiler.compile(qc)
        assert qubit_circuit is not None
        assert qubit_circuit.num_qubits == 4  # 2 qutrits × 2 qubits

    def test_parametric_gate_compile(self):
        """Compile parametric rotation gate."""
        qc = QutritCircuit(1)
        qc.rx3(np.pi / 4, 0)
        qc.ry3(np.pi / 3, 0)
        compiler = QutritToQubitCompiler()
        qubit_circuit = compiler.compile(qc)
        assert qubit_circuit is not None
        assert qubit_circuit.num_qubits == 2

    def test_result_mapper_round_trip(self):
        """Compile, simulate trit probs, map back to trit outcomes."""
        # Simulate at qutrit level
        qc = QutritCircuit(1)
        qc.x3(0)  # |1⟩
        sim = StatevectorSimulator()
        result = sim.run(qc, shots=1000)

        # Create mock qubit counts from trit result
        # |1⟩ in qutrit = |01⟩ in qubit encoding
        qubit_counts = {'01': 1000}

        mapper = ResultMapper(n_qutrits=1)
        trit_counts = mapper.map_counts(qubit_counts)
        assert '1' in trit_counts
        assert trit_counts['1'] == 1000

    def test_result_mapper_multi_qutrit(self):
        """Map 2-qutrit qubit results back to trit basis."""
        # |1⟩|2⟩ = |01⟩|10⟩ in qubit encoding = '0110'
        qubit_counts = {'0110': 500, '0001': 300, '0000': 200}
        mapper = ResultMapper(n_qutrits=2)
        trit_counts = mapper.map_counts(qubit_counts)
        # |01⟩|10⟩ → '12', |00⟩|01⟩ → '01', |00⟩|00⟩ → '00'
        assert trit_counts.get('12', 0) == 500
        assert trit_counts.get('01', 0) == 300
        assert trit_counts.get('00', 0) == 200

    def test_leakage_handler_clean_counts(self):
        """LeakageHandler removes |11⟩ states from counts."""
        # Include a forbidden '11' state for one qutrit
        qubit_counts = {'01': 800, '11': 100, '00': 100}
        handler = LeakageHandler(n_qutrits=1, post_select=True)
        valid, leaked = handler.filter_counts(qubit_counts)
        assert '11' not in valid
        # Total should be reduced
        assert sum(valid.values()) == 900

    def test_leakage_handler_nearest_mode(self):
        """LeakageHandler nearest mode maps |11⟩ → closest valid."""
        qubit_counts = {'11': 100, '00': 900}
        handler = LeakageHandler(n_qutrits=1, post_select=False, leakage_fill='nearest')
        valid, leaked = handler.filter_counts(qubit_counts)
        # '11' should be mapped to some valid state (trit '2' via nearest)
        total = sum(valid.values())
        assert total == 1000

    def test_leakage_rate_calculation(self):
        """LeakageHandler computes correct leakage rate."""
        qubit_counts = {'00': 700, '01': 200, '10': 50, '11': 50}
        handler = LeakageHandler(n_qutrits=1, post_select=True)
        rate = handler.leakage_rate(qubit_counts)
        assert rate == pytest.approx(50 / 1000, abs=1e-10)


class TestCompilerEmbeddingConsistency:
    """Verify that compiled gates preserve qutrit-subspace unitarity."""

    def test_x3_embedding_is_unitary(self):
        """Embedded X3 gate is unitary in full qubit space."""
        embedded = embed_single_qutrit_gate(X3)
        assert embedded.shape == (4, 4)
        identity = embedded @ embedded.conj().T
        np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=10)

    def test_h3_embedding_is_unitary(self):
        """Embedded H3 gate is unitary."""
        embedded = embed_single_qutrit_gate(H3)
        identity = embedded @ embedded.conj().T
        np.testing.assert_array_almost_equal(identity, np.eye(4), decimal=10)

    def test_csum_embedding_is_unitary(self):
        """Embedded CSUM is unitary in 16-dimensional space."""
        embedded = embed_two_qutrit_gate(CSUM)
        assert embedded.shape == (16, 16)
        identity = embedded @ embedded.conj().T
        np.testing.assert_array_almost_equal(identity, np.eye(16), decimal=10)

    def test_embedding_preserves_qutrit_action(self):
        """Embedded X3 acting on |0⟩=|00⟩ gives |1⟩=|01⟩."""
        embedded = embed_single_qutrit_gate(X3)
        state_00 = np.array([1, 0, 0, 0], dtype=complex)
        result = embedded @ state_00
        expected = np.array([0, 1, 0, 0], dtype=complex)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_embedding_blocks_forbidden_state(self):
        """Embedded gate maps |11⟩→|11⟩ (identity on forbidden subspace)."""
        embedded = embed_single_qutrit_gate(X3)
        state_11 = np.array([0, 0, 0, 1], dtype=complex)
        result = embedded @ state_11
        # |11⟩ should map to |11⟩ (identity on forbidden)
        np.testing.assert_array_almost_equal(result, state_11, decimal=10)


class TestFullPipeline:
    """Full pipeline: circuit → simulate → compile → map results."""

    def test_simulate_then_compile_consistency(self):
        """Qutrit simulation and compiled simulation give same probabilities."""
        # Qutrit level
        qc = QutritCircuit(1)
        qc.x3(0)
        sim = StatevectorSimulator()
        trit_result = sim.run(qc, shots=10000)

        # The trit result: statevector index 1 should dominate
        sv = trit_result.statevector
        assert np.abs(sv[1]) ** 2 == pytest.approx(1.0, abs=1e-10)

        # Compiled qubit level (simulate the embedding)
        embedded_x3 = embed_single_qutrit_gate(X3)
        qubit_state = embedded_x3 @ np.array([1, 0, 0, 0], dtype=complex)
        qubit_probs = np.abs(qubit_state) ** 2

        # Map back: |01⟩ should have probability ~1
        assert qubit_probs[1] == pytest.approx(1.0, abs=1e-10)

        # Result mapper
        mapper = ResultMapper(n_qutrits=1)
        # Simulate qubit measurement
        qubit_counts = {'01': 10000}
        trit_counts = mapper.map_counts(qubit_counts)
        assert trit_counts.get('1', 0) == 10000

    def test_bell_state_pipeline(self):
        """Qutrit Bell state: H → CSUM → measure, full pipeline."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        sim = StatevectorSimulator()
        result = sim.run(qc, shots=30000)

        # Check statevector for |00⟩, |11⟩, |22⟩ each ~1/3
        sv = result.statevector
        probs_sv = np.abs(sv) ** 2
        assert probs_sv[0] == pytest.approx(1.0 / 3, abs=0.05)  # |00⟩
        assert probs_sv[4] == pytest.approx(1.0 / 3, abs=0.05)  # |11⟩
        assert probs_sv[8] == pytest.approx(1.0 / 3, abs=0.05)  # |22⟩
