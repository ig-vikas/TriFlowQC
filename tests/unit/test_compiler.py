"""Unit tests for compiler module."""

import numpy as np
import pytest
from triflowqc.compiler.qutrit_to_qubit import (
    embed_single_qutrit_gate,
    embed_two_qutrit_gate,
    embed_three_qutrit_gate,
    ENCODING,
)
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper


class TestEmbedding:
    def test_single_identity(self):
        V = embed_single_qutrit_gate(np.eye(3, dtype=complex))
        assert np.allclose(V, np.eye(4))

    def test_single_unitary(self):
        """Embedded gate should be unitary in 4×4 space."""
        from triflowqc.gates.single_qutrit import X3
        V = embed_single_qutrit_gate(X3)
        assert np.allclose(V @ V.conj().T, np.eye(4), atol=1e-10)

    def test_single_preserves_encoding(self):
        """Check that |0⟩→|00⟩, |1⟩→|01⟩, |2⟩→|10⟩ mapping works."""
        from triflowqc.gates.single_qutrit import X3
        V = embed_single_qutrit_gate(X3)
        # X3|0⟩ = |1⟩ → in qubit space: |00⟩ → |01⟩
        state_00 = np.array([1, 0, 0, 0], dtype=complex)
        result = V @ state_00
        # Should be |01⟩ = [0, 1, 0, 0]
        assert np.isclose(np.abs(result[1]), 1.0)

    def test_two_qutrit_unitary(self):
        from triflowqc.gates.two_qutrit import CSUM
        V = embed_two_qutrit_gate(CSUM)
        assert V.shape == (16, 16)
        assert np.allclose(V @ V.conj().T, np.eye(16), atol=1e-10)

    def test_three_qutrit_unitary(self):
        from triflowqc.gates.three_qutrit import Toffoli3
        V = embed_three_qutrit_gate(Toffoli3)
        assert V.shape == (64, 64)
        assert np.allclose(V @ V.conj().T, np.eye(64), atol=1e-10)


class TestLeakageHandler:
    def test_no_leakage(self):
        lh = LeakageHandler(n_qutrits=1)
        assert not lh.is_leaked("00")
        assert not lh.is_leaked("01")
        assert not lh.is_leaked("10")

    def test_leakage_detected(self):
        lh = LeakageHandler(n_qutrits=1)
        assert lh.is_leaked("11")

    def test_bitstring_to_trits_valid(self):
        lh = LeakageHandler(n_qutrits=2)
        assert lh.bitstring_to_trits("0001") == "01"
        assert lh.bitstring_to_trits("0010") == "02"
        assert lh.bitstring_to_trits("1000") == "20"

    def test_bitstring_to_trits_leaked(self):
        lh = LeakageHandler(n_qutrits=1)
        assert lh.bitstring_to_trits("11") is None

    def test_filter_counts(self):
        lh = LeakageHandler(n_qutrits=1)
        counts = {"00": 40, "01": 30, "10": 20, "11": 10}
        valid, leaked = lh.filter_counts(counts)
        assert "11" not in str(valid)
        assert sum(valid.values()) == 90
        assert sum(leaked.values()) == 10

    def test_leakage_rate(self):
        lh = LeakageHandler(n_qutrits=1)
        counts = {"00": 90, "11": 10}
        rate = lh.leakage_rate(counts)
        assert np.isclose(rate, 0.1)


class TestResultMapper:
    def test_map_counts(self):
        mapper = ResultMapper(n_qutrits=1)
        qubit_counts = {"00": 50, "01": 30, "10": 20}
        trit_counts = mapper.map_counts(qubit_counts)
        assert trit_counts == {"0": 50, "1": 30, "2": 20}

    def test_map_to_measurement_result(self):
        mapper = ResultMapper(n_qutrits=1)
        qubit_counts = {"00": 100}
        result = mapper.map_to_measurement_result(qubit_counts)
        assert np.isclose(result.probabilities[0], 1.0)
