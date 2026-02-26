"""Unit tests for noise model and noisy simulator."""

import numpy as np
import pytest
from triflowqc.simulator.noise_model import (
    NoiseModel,
    depolarising_channel,
    dephasing_channel,
    amplitude_damping_channel,
    trit_flip_channel,
    _validate_kraus,
)
from triflowqc.simulator.noisy_simulator import NoisySimulator
from triflowqc.circuit.circuit import QutritCircuit


class TestNoiseChannels:
    def test_depolarising_completeness(self):
        """Kraus operators sum to identity: Σ K†K = I."""
        ops = depolarising_channel(0.1)
        total = sum(K.conj().T @ K for K in ops)
        assert np.allclose(total, np.eye(3), atol=1e-10)

    def test_dephasing_completeness(self):
        ops = dephasing_channel(0.2)
        total = sum(K.conj().T @ K for K in ops)
        assert np.allclose(total, np.eye(3), atol=1e-10)

    def test_amplitude_damping_completeness(self):
        ops = amplitude_damping_channel(0.1, 0.05, 0.02)
        total = sum(K.conj().T @ K for K in ops)
        assert np.allclose(total, np.eye(3), atol=1e-10)

    def test_trit_flip_completeness(self):
        ops = trit_flip_channel(0.3)
        total = sum(K.conj().T @ K for K in ops)
        assert np.allclose(total, np.eye(3), atol=1e-10)

    def test_zero_noise_preserves_state(self):
        ops = depolarising_channel(0.0)
        rho = np.diag([1.0, 0.0, 0.0]).astype(complex)
        result = sum(K @ rho @ K.conj().T for K in ops)
        assert np.allclose(result, rho)

    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            depolarising_channel(1.5)


class TestNoiseModel:
    def test_has_noise(self):
        nm = NoiseModel()
        assert not nm.has_noise()
        nm.add_all_qutrit_depolarising(0.01)
        assert nm.has_noise()

    def test_readout_error(self):
        nm = NoiseModel()
        M = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        nm.add_readout_error(M)
        assert nm.get_readout_matrix(0) is not None


class TestNoisySimulator:
    def test_noisy_run(self):
        qc = QutritCircuit(1)
        qc.x3(0)

        nm = NoiseModel()
        nm.add_all_qutrit_depolarising(0.01)

        sim = NoisySimulator(nm)
        result = sim.run(qc, shots=1000)
        assert sum(result.counts.values()) == 1000
        # Mostly |1⟩ but with some noise
        assert result.probabilities[1] > 0.5
