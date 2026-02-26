"""Unit tests for simulator module."""

import numpy as np
import pytest
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator


class TestStatevectorSimulator:
    def test_empty_circuit(self):
        qc = QutritCircuit(1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        # Should be |0⟩
        assert np.isclose(result.probabilities[0], 1.0)

    def test_x3_gate(self):
        qc = QutritCircuit(1)
        qc.x3(0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        # |0⟩ → |1⟩
        assert np.isclose(result.probabilities[1], 1.0)

    def test_h3_gate(self):
        qc = QutritCircuit(1)
        qc.h3(0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        # Equal superposition
        assert np.allclose(result.probabilities, [1 / 3] * 3, atol=1e-10)

    def test_two_qutrits(self):
        qc = QutritCircuit(2)
        qc.x3(0)
        qc.csum(0, 1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        # |0⟩|0⟩ → |1⟩|0⟩ → CSUM → |1⟩|(1+0)mod3⟩ = |1⟩|1⟩ = index 4
        assert np.isclose(result.probabilities[4], 1.0)

    def test_with_shots(self):
        qc = QutritCircuit(1)
        sim = StatevectorSimulator()
        result = sim.run(qc, shots=1000)
        assert result.shots == 1000
        assert sum(result.counts.values()) == 1000

    def test_statevector_returned(self):
        qc = QutritCircuit(1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        assert result.statevector is not None
        assert len(result.statevector) == 3


class TestDensityMatrixSimulator:
    def test_empty_circuit(self):
        qc = QutritCircuit(1)
        sim = DensityMatrixSimulator()
        result = sim.run(qc)
        assert np.isclose(result.probabilities[0], 1.0)

    def test_x3_gate(self):
        qc = QutritCircuit(1)
        qc.x3(0)
        sim = DensityMatrixSimulator()
        result = sim.run(qc)
        assert np.isclose(result.probabilities[1], 1.0)

    def test_agreement_with_statevector(self):
        """Both simulators should give same probabilities."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)

        sv_sim = StatevectorSimulator()
        dm_sim = DensityMatrixSimulator()

        sv_result = sv_sim.run(qc)
        dm_result = dm_sim.run(qc)

        assert np.allclose(
            sv_result.probabilities,
            dm_result.probabilities,
            atol=1e-10,
        )
