"""
Integration tests: Circuit creation → Simulator execution → Result analysis.
"""

import numpy as np
import pytest

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.circuit.measurement import MeasurementResult
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator
from triflowqc.simulator.noise_model import NoiseModel, depolarising_channel
from triflowqc.simulator.noisy_simulator import NoisySimulator


class TestCircuitToSimulator:
    """End-to-end circuit → statevector simulator."""

    def test_identity_circuit(self):
        """Empty circuit returns |000...0⟩."""
        qc = QutritCircuit(3)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert sv is not None
        assert abs(sv[0]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_x3_gate_cycles(self):
        """X3 applied 3 times returns to |0⟩."""
        qc = QutritCircuit(1)
        qc.x3(0)
        qc.x3(0)
        qc.x3(0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert abs(sv[0]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_x3_single_application(self):
        """|0⟩ → |1⟩ after one X3."""
        qc = QutritCircuit(1)
        qc.x3(0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert abs(sv[1]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_hadamard_superposition(self):
        """H3 on |0⟩ creates uniform superposition."""
        qc = QutritCircuit(1)
        qc.h3(0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        probs = np.abs(sv) ** 2
        expected = 1.0 / 3.0
        for p in probs:
            assert p == pytest.approx(expected, abs=1e-10)

    def test_csum_entanglement(self):
        """CSUM on |1⟩|0⟩ → |1⟩|1⟩."""
        qc = QutritCircuit(2)
        qc.x3(0)  # |1⟩|0⟩
        qc.csum(0, 1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        # |1⟩|1⟩ = index 1*3 + 1 = 4
        assert abs(sv[4]) ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_csum_with_h_creates_entanglement(self):
        """H3|0⟩ then CSUM creates entangled state."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        probs = np.abs(sv) ** 2
        # Expect |00⟩, |11⟩, |22⟩ each with prob 1/3
        assert probs[0] == pytest.approx(1.0 / 3, abs=1e-6)  # |00⟩
        assert probs[4] == pytest.approx(1.0 / 3, abs=1e-6)  # |11⟩
        assert probs[8] == pytest.approx(1.0 / 3, abs=1e-6)  # |22⟩

    def test_parametric_gate_in_circuit(self):
        """Parametric rotation preserves norm."""
        qc = QutritCircuit(1)
        qc.rx3(np.pi / 4, 0)
        qc.ry3(np.pi / 3, 0)
        qc.rz3(np.pi / 6, 0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        norm = np.sum(np.abs(sv) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_multi_qutrit_parametric(self):
        """Multi-qutrit circuit with parametric gates."""
        qc = QutritCircuit(3)
        for i in range(3):
            qc.rx3(np.pi / (i + 2), i)
            qc.ry3(np.pi / (i + 3), i)
        qc.csum(0, 1)
        qc.csum(1, 2)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert np.sum(np.abs(sv) ** 2) == pytest.approx(1.0, abs=1e-10)

    def test_measurement_result_properties(self):
        """MeasurementResult has correct probabilities."""
        qc = QutritCircuit(2)
        qc.x3(0)  # |10⟩ in qutrit basis
        sim = StatevectorSimulator()
        result = sim.run(qc, shots=10000)
        probs = result.histogram()
        # Should be nearly all in state |10⟩
        assert probs.get('10', 0) > 0.95

    def test_two_simulators_agree(self):
        """Statevector and density matrix simulators produce same probabilities."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.rx3(np.pi / 5, 1)
        qc.csum(0, 1)
        sv_sim = StatevectorSimulator()
        dm_sim = DensityMatrixSimulator()
        sv_result = sv_sim.run(qc, shots=50000)
        dm_result = dm_sim.run(qc, shots=50000)
        sv_probs = sv_result.histogram()
        dm_probs = dm_result.histogram()
        all_keys = set(list(sv_probs.keys()) + list(dm_probs.keys()))
        for k in all_keys:
            assert sv_probs.get(k, 0) == pytest.approx(dm_probs.get(k, 0), abs=0.05)


class TestCircuitToNoisySimulator:
    """Circuit → Noisy simulator integration."""

    def test_noiseless_noisy_sim_matches_statevector(self):
        """NoisySimulator with no noise matches DensityMatrixSimulator."""
        qc = QutritCircuit(1)
        qc.h3(0)
        noise = NoiseModel()
        noisy_sim = NoisySimulator(noise)
        dm_sim = DensityMatrixSimulator()
        noisy_result = noisy_sim.run(qc, shots=20000)
        dm_result = dm_sim.run(qc, shots=20000)
        noisy_probs = noisy_result.histogram()
        dm_probs = dm_result.histogram()
        for k in ['0', '1', '2']:
            np_val = noisy_probs.get(k, 0)
            dm_val = dm_probs.get(k, 0)
            assert np_val == pytest.approx(dm_val, abs=0.05)

    def test_heavy_noise_degrades_signal(self):
        """Heavy depolarising noise moves probabilities toward uniform."""
        qc = QutritCircuit(1)
        qc.x3(0)  # |1⟩ deterministically
        noise = NoiseModel()
        noise.add_gate_error('X3', depolarising_channel(0.9))
        noisy_sim = NoisySimulator(noise)
        result = noisy_sim.run(qc, shots=50000)
        probs = result.histogram()
        # With 90% depolarising, signal is heavily degraded
        # P(1) should be much less than 1.0
        p1 = probs.get('1', 0)
        assert p1 < 0.8  # Not nearly deterministic anymore

    def test_multi_qutrit_noisy_circuit(self):
        """Multi-qutrit noisy circuit still produces valid distribution."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        noise = NoiseModel()
        noise.add_gate_error('H3', depolarising_channel(0.01))
        noise.add_gate_error('CSUM', depolarising_channel(0.02))
        noisy_sim = NoisySimulator(noise)
        result = noisy_sim.run(qc, shots=10000)
        probs = result.histogram()
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=0.01)


class TestCircuitDepthAndParameters:
    """Test circuit analysis integrates with simulation."""

    def test_circuit_depth(self):
        """Circuit reports correct depth."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.h3(1)
        qc.csum(0, 1)
        qc.rx3(0.5, 0)
        depth = qc.depth
        assert depth >= 2  # At least the parallel H + CSUM

    def test_parameter_count(self):
        """Circuit reports correct parameter count."""
        qc = QutritCircuit(2)
        qc.rx3(0.1, 0)
        qc.ry3(0.2, 1)
        qc.rz3(0.3, 0)
        params = qc.parameters
        assert len(params) >= 3

    def test_parameterized_circuit_simulation(self):
        """Run a parameterized circuit, update parameters, re-run."""
        qc = QutritCircuit(1)
        qc.rx3(0.0, 0)
        sim = StatevectorSimulator()
        r1 = sim.run(qc)
        sv1 = r1.statevector
        # |0⟩ should be unchanged with theta=0
        assert abs(sv1[0]) ** 2 == pytest.approx(1.0, abs=1e-10)

        # Now create circuit with non-zero angle
        qc2 = QutritCircuit(1)
        qc2.rx3(np.pi / 2, 0)
        r2 = sim.run(qc2)
        sv2 = r2.statevector
        # Should no longer be |0⟩
        assert abs(sv2[0]) ** 2 < 0.99


class TestLargeCircuit:
    """Test with larger circuits for performance/correctness."""

    def test_five_qutrit_circuit(self):
        """5-qutrit circuit with various gates."""
        qc = QutritCircuit(5)
        for i in range(5):
            qc.h3(i)
        for i in range(4):
            qc.csum(i, i + 1)
        for i in range(5):
            qc.rx3(np.pi / (i + 2), i)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert len(sv) == 3 ** 5  # 243
        assert np.sum(np.abs(sv) ** 2) == pytest.approx(1.0, abs=1e-10)

    def test_repeated_layers(self):
        """Simulate a circuit with repeated gate layers."""
        n_qutrits = 3
        n_layers = 5
        qc = QutritCircuit(n_qutrits)
        for layer in range(n_layers):
            for i in range(n_qutrits):
                qc.rx3(0.1 * layer, i)
                qc.ry3(0.2 * layer, i)
            for i in range(n_qutrits - 1):
                qc.csum(i, i + 1)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        assert np.sum(np.abs(sv) ** 2) == pytest.approx(1.0, abs=1e-10)
