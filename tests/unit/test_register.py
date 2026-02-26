"""Unit tests for core.register module."""

import numpy as np
import pytest
from triflowqc.core.register import QutritRegister


class TestRegisterInit:
    def test_default_state(self):
        reg = QutritRegister(2)
        assert reg.num_qutrits == 2
        assert len(reg.state_vector) == 9
        assert np.isclose(reg.state_vector[0], 1.0)

    def test_custom_state(self):
        state = np.zeros(9, dtype=complex)
        state[4] = 1.0  # |11⟩
        reg = QutritRegister(2, state)
        assert np.isclose(reg.state_vector[4], 1.0)

    def test_normalisation(self):
        reg = QutritRegister(3)
        assert np.isclose(np.linalg.norm(reg.state_vector), 1.0)


class TestRegisterGates:
    def test_apply_single_gate(self):
        reg = QutritRegister(2)
        X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
        reg.apply_single_gate(X3, 0)
        # |00⟩ → |10⟩ — qutrit 0 flipped from 0→1
        probs = np.abs(reg.state_vector) ** 2
        # |10⟩ in base-3 = 1*3 + 0 = index 3
        assert np.isclose(probs[3], 1.0)

    def test_apply_two_gate(self):
        reg = QutritRegister(2)
        SWAP = np.eye(9, dtype=complex)
        # Build swap: |ab⟩ → |ba⟩
        for a in range(3):
            for b in range(3):
                SWAP[a * 3 + b, :] = 0
        for a in range(3):
            for b in range(3):
                SWAP[a * 3 + b, b * 3 + a] = 1.0

        reg.apply_single_gate(
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex), 0
        )
        # Now in state |10⟩
        reg.apply_two_gate(SWAP, 0, 1)
        # Should be in |01⟩ = index 1
        probs = np.abs(reg.state_vector) ** 2
        assert np.isclose(probs[1], 1.0)

    def test_measure_all(self):
        reg = QutritRegister(2)
        result = reg.measure_all()
        assert len(result) == 2
        assert all(r in [0, 1, 2] for r in result)

    def test_measure_qutrit(self):
        reg = QutritRegister(2)
        result, prob = reg.measure_qutrit(0)
        assert result in [0, 1, 2]
        assert 0.0 <= prob <= 1.0

    def test_partial_trace(self):
        reg = QutritRegister(2)
        dm = reg.partial_trace([1])  # trace out qutrit 1
        assert dm.shape == (3, 3)
        assert np.isclose(np.trace(dm), 1.0)
