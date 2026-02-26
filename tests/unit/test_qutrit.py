"""Unit tests for core.qutrit module."""

import numpy as np
import pytest
from triflowqc.core.qutrit import Qutrit


class TestQutritInit:
    """Test Qutrit initialization."""

    def test_init_from_int_0(self):
        q = Qutrit(0)
        assert np.allclose(q.amplitudes, [1, 0, 0])

    def test_init_from_int_1(self):
        q = Qutrit(1)
        assert np.allclose(q.amplitudes, [0, 1, 0])

    def test_init_from_int_2(self):
        q = Qutrit(2)
        assert np.allclose(q.amplitudes, [0, 0, 1])

    def test_init_from_array(self):
        state = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
        q = Qutrit(state)
        assert np.allclose(q.amplitudes, state)

    def test_init_normalises(self):
        q = Qutrit(np.array([1, 1, 0], dtype=complex))
        assert np.isclose(np.linalg.norm(q.amplitudes), 1.0)

    def test_invalid_int(self):
        with pytest.raises((ValueError, IndexError)):
            Qutrit(3)


class TestQutritOperations:
    """Test Qutrit methods."""

    def test_probabilities_basis(self):
        q = Qutrit(0)
        probs = q.probabilities()
        assert np.allclose(probs, [1, 0, 0])

    def test_probabilities_superposition(self):
        q = Qutrit(np.array([1, 1, 1], dtype=complex))
        probs = q.probabilities()
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-10)

    def test_measure_returns_valid(self):
        q = Qutrit(0)
        result = q.measure()
        assert result in [0, 1, 2]

    def test_measure_basis_0(self):
        q = Qutrit(0)
        # Should always measure 0
        results = [q.measure() for _ in range(100)]
        # Reset each time
        for _ in range(100):
            q2 = Qutrit(0)
            r = q2.measure()
            assert r == 0

    def test_density_matrix(self):
        q = Qutrit(0)
        dm = q.density_matrix()
        expected = np.zeros((3, 3), dtype=complex)
        expected[0, 0] = 1.0
        assert np.allclose(dm, expected)

    def test_apply_gate(self):
        q = Qutrit(0)
        # Apply X3 (cyclic permutation)
        X3 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=complex)
        q.apply_gate(X3)
        assert np.allclose(q.probabilities(), [0, 1, 0])

    def test_fidelity_same_state(self):
        q1 = Qutrit(0)
        q2 = Qutrit(0)
        assert np.isclose(q1.fidelity(q2), 1.0)

    def test_fidelity_orthogonal(self):
        q1 = Qutrit(0)
        q2 = Qutrit(1)
        assert np.isclose(q1.fidelity(q2), 0.0)

    def test_copy(self):
        q = Qutrit(np.array([1, 1j, 0], dtype=complex))
        q_copy = q.copy()
        assert np.allclose(q.amplitudes, q_copy.amplitudes)

    def test_bloch_vector_dimension(self):
        q = Qutrit(0)
        bv = q.bloch_vector()
        assert bv.shape == (8,)

    def test_repr(self):
        q = Qutrit(0)
        r = repr(q)
        assert "Qutrit" in r
