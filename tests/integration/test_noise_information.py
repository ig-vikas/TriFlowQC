"""
Integration tests: Noise model → noisy simulation → information theory analysis.
"""

import numpy as np
import pytest

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator
from triflowqc.simulator.noise_model import (
    NoiseModel,
    depolarising_channel,
    dephasing_channel,
    amplitude_damping_channel,
    trit_flip_channel,
)
from triflowqc.simulator.noisy_simulator import NoisySimulator
from triflowqc.math.information import (
    von_neumann_entropy,
    linear_entropy,
    purity,
)


class TestNoiseAndInformation:
    """Test that noise model effects are correctly captured by information measures."""

    def test_pure_state_entropy_is_zero(self):
        """A pure state (no noise) should have zero entropy."""
        qc = QutritCircuit(1)
        qc.h3(0)
        dm_sim = DensityMatrixSimulator()
        result = dm_sim.run(qc)
        # Get density matrix from simulator
        rho = dm_sim.get_density_matrix(qc)
        entropy = von_neumann_entropy(rho)
        assert entropy == pytest.approx(0.0, abs=1e-10)

    def test_noise_increases_entropy(self):
        """Depolarising noise should increase entropy."""
        qc = QutritCircuit(1)
        qc.x3(0)

        # Pure state entropy
        dm_sim = DensityMatrixSimulator()
        rho_pure = dm_sim.get_density_matrix(qc)
        entropy_pure = von_neumann_entropy(rho_pure)

        # Noisy state entropy
        noise = NoiseModel()
        noise.add_gate_error('X3', depolarising_channel(0.3))
        noisy_sim = NoisySimulator(noise)
        rho_noisy = noisy_sim.get_density_matrix(qc)
        entropy_noisy = von_neumann_entropy(rho_noisy)

        assert entropy_noisy > entropy_pure

    def test_full_depolarisation_gives_max_entropy(self):
        """Full depolarisation (p=1) gives maximally mixed state."""
        qc = QutritCircuit(1)
        qc.x3(0)
        noise = NoiseModel()
        noise.add_gate_error('X3', depolarising_channel(1.0))
        noisy_sim = NoisySimulator(noise)
        rho = noisy_sim.get_density_matrix(qc)

        # Maximally mixed state has entropy log(3)
        max_entropy = np.log(3)
        actual_entropy = von_neumann_entropy(rho)
        assert actual_entropy == pytest.approx(max_entropy, abs=0.1)

    def test_purity_decreases_with_noise(self):
        """Purity should decrease with depolarising noise."""
        qc = QutritCircuit(1)
        qc.h3(0)

        dm_sim = DensityMatrixSimulator()
        rho_pure = dm_sim.get_density_matrix(qc)
        purity_pure = purity(rho_pure)

        noise = NoiseModel()
        noise.add_gate_error('H3', depolarising_channel(0.2))
        noisy_sim = NoisySimulator(noise)
        rho_noisy = noisy_sim.get_density_matrix(qc)
        purity_noisy = purity(rho_noisy)

        assert purity_pure == pytest.approx(1.0, abs=1e-10)
        assert purity_noisy < purity_pure

    def test_dephasing_preserves_populations(self):
        """Dephasing should not change diagonal elements of density matrix."""
        qc = QutritCircuit(1)
        qc.h3(0)  # Creates superposition

        dm_sim = DensityMatrixSimulator()
        rho_pure = dm_sim.get_density_matrix(qc)
        diag_pure = np.diag(rho_pure).real

        noise = NoiseModel()
        noise.add_gate_error('H3', dephasing_channel(0.5))
        noisy_sim = NoisySimulator(noise)
        rho_noisy = noisy_sim.get_density_matrix(qc)
        diag_noisy = np.diag(rho_noisy).real

        # Populations (diagonal) should be preserved
        np.testing.assert_array_almost_equal(diag_pure, diag_noisy, decimal=5)


class TestNoiseChannelComposition:
    """Test composition of multiple noise channels."""

    def test_double_noise_more_mixed(self):
        """Two noisy gates produce more mixed state than one."""
        # Single noisy gate
        qc1 = QutritCircuit(1)
        qc1.x3(0)

        noise = NoiseModel()
        noise.add_gate_error('X3', depolarising_channel(0.1))
        noisy_sim = NoisySimulator(noise)
        rho1 = noisy_sim.get_density_matrix(qc1)

        # Two noisy gates
        qc2 = QutritCircuit(1)
        qc2.x3(0)
        qc2.x3(0)
        rho2 = noisy_sim.get_density_matrix(qc2)

        p1 = purity(rho1)
        p2 = purity(rho2)
        # More noisy gates → less purity
        assert p2 <= p1 + 1e-10

    def test_trit_flip_channel(self):
        """Trit-flip channel mixes states."""
        qc = QutritCircuit(1)
        # Start in |0⟩ — trit flip should mix in |1⟩ and |2⟩
        noise = NoiseModel()
        noise.add_local_error(0, 'identity', trit_flip_channel(0.5))
        noisy_sim = NoisySimulator(noise)
        result = noisy_sim.run(qc, shots=10000)
        probs = result.histogram()
        # Should see mixture of states
        assert probs.get('0', 0) > 0.1

    def test_amplitude_damping_to_ground(self):
        """Amplitude damping should push state toward |0⟩."""
        qc = QutritCircuit(1)
        qc.x3(0)  # Start in |1⟩

        noise = NoiseModel()
        noise.add_gate_error('X3', amplitude_damping_channel(0.8, 0.5))
        noisy_sim = NoisySimulator(noise)
        result = noisy_sim.run(qc, shots=10000)
        probs = result.histogram()
        # |0⟩ should have significant probability due to damping
        assert probs.get('0', 0) > 0.3


class TestMultiQutritNoise:
    """Multi-qutrit noisy simulation."""

    def test_two_qutrit_noisy_entanglement(self):
        """Noisy Bell state has reduced correlations."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)

        # Noiseless
        dm_sim = DensityMatrixSimulator()
        rho_pure = dm_sim.get_density_matrix(qc)

        # Noisy
        noise = NoiseModel()
        noise.add_gate_error('H3', depolarising_channel(0.05))
        noise.add_gate_error('CSUM', depolarising_channel(0.05))
        noisy_sim = NoisySimulator(noise)
        rho_noisy = noisy_sim.get_density_matrix(qc)

        # Noisy state should have less purity
        assert purity(rho_noisy) < purity(rho_pure) + 1e-10

    def test_readout_error(self):
        """Readout error changes measurement statistics."""
        qc = QutritCircuit(1)
        qc.x3(0)  # Deterministic |1⟩

        # No readout error
        noise_clean = NoiseModel()
        sim_clean = NoisySimulator(noise_clean)
        result_clean = sim_clean.run(qc, shots=10000)

        # With readout error
        noise_readout = NoiseModel()
        # Confusion matrix: 10% chance of misread
        confusion = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        noise_readout.add_readout_error(confusion)
        sim_readout = NoisySimulator(noise_readout)
        result_readout = sim_readout.run(qc, shots=10000)

        # Clean should be nearly all '1'
        clean_probs = result_clean.histogram()
        readout_probs = result_readout.histogram()
        assert clean_probs.get('1', 0) > 0.95
        # Readout error should spread probability
        assert readout_probs.get('1', 0) < 0.95
        assert readout_probs.get('0', 0) > 0.01
