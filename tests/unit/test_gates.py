"""Unit tests for gates module."""

import numpy as np
import pytest
from triflowqc.gates.gell_mann import (
    LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
    LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8,
    gell_mann, structure_constants,
)
from triflowqc.gates.single_qutrit import (
    X3, Z3, H3, T3, S3, X3_01, X3_12, X3_02, Phase3, Identity3,
)
from triflowqc.gates.parametric import R_GM, RX3, RY3, RZ3, general_su3_rotation
from triflowqc.gates.two_qutrit import CSUM, CPHASE, SWAP3, CX3
from triflowqc.gates.three_qutrit import Toffoli3
from triflowqc.gates.custom_gate import CustomGate


class TestGellMann:
    def test_count(self):
        for k in range(1, 9):
            lam = gell_mann(k)
            assert lam.shape == (3, 3)

    def test_hermitian(self):
        for k in range(1, 9):
            lam = gell_mann(k)
            assert np.allclose(lam, lam.conj().T)

    def test_traceless(self):
        for k in range(1, 9):
            assert np.isclose(np.trace(gell_mann(k)), 0.0)

    def test_normalisation(self):
        """Tr(λ_i λ_j) = 2 δ_ij."""
        for i in range(1, 9):
            for j in range(1, 9):
                tr = np.trace(gell_mann(i) @ gell_mann(j))
                expected = 2.0 if i == j else 0.0
                assert np.isclose(tr, expected, atol=1e-10), f"Failed for ({i},{j})"

    def test_structure_constants_antisymmetric(self):
        f = structure_constants()
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    assert np.isclose(f[i, j, k], -f[j, i, k], atol=1e-10)


class TestSingleQutritGates:
    def test_X3_unitary(self):
        assert np.allclose(X3 @ X3.conj().T, np.eye(3))

    def test_X3_cyclic(self):
        """X3³ = I."""
        assert np.allclose(X3 @ X3 @ X3, np.eye(3))

    def test_Z3_unitary(self):
        assert np.allclose(Z3 @ Z3.conj().T, np.eye(3))

    def test_H3_unitary(self):
        assert np.allclose(H3 @ H3.conj().T, np.eye(3))

    def test_H3_is_QFT(self):
        """H3 should be the 3-point QFT / √3."""
        omega = np.exp(2j * np.pi / 3)
        expected = np.array([
            [1, 1, 1],
            [1, omega, omega ** 2],
            [1, omega ** 2, omega ** 4],
        ]) / np.sqrt(3)
        assert np.allclose(H3, expected)

    def test_identity(self):
        assert np.allclose(Identity3, np.eye(3))

    def test_phase3(self):
        P = Phase3(np.pi / 4, np.pi / 3)
        assert np.allclose(P @ P.conj().T, np.eye(3))

    def test_partial_swaps_unitary(self):
        for gate in [X3_01, X3_12, X3_02]:
            assert np.allclose(gate @ gate.conj().T, np.eye(3))


class TestParametricGates:
    def test_identity_at_zero(self):
        """R_GM(k, 0) = I for all k."""
        for k in range(1, 9):
            assert np.allclose(R_GM(k, 0.0), np.eye(3), atol=1e-10)

    def test_unitary(self):
        for k in range(1, 9):
            U = R_GM(k, 1.23)
            assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-10)

    def test_general_su3(self):
        angles = np.random.uniform(0, 2 * np.pi, 8)
        U = general_su3_rotation(*angles)
        assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-10)


class TestTwoQutritGates:
    def test_CSUM_unitary(self):
        assert np.allclose(CSUM @ CSUM.conj().T, np.eye(9))

    def test_CPHASE_unitary(self):
        assert np.allclose(CPHASE @ CPHASE.conj().T, np.eye(9))

    def test_SWAP3_unitary(self):
        assert np.allclose(SWAP3 @ SWAP3.conj().T, np.eye(9))

    def test_SWAP3_involution(self):
        """SWAP² = I."""
        assert np.allclose(SWAP3 @ SWAP3, np.eye(9))

    def test_CSUM_action(self):
        """CSUM |a,b⟩ = |a, (a+b) mod 3⟩."""
        for a in range(3):
            for b in range(3):
                state = np.zeros(9, dtype=complex)
                state[a * 3 + b] = 1.0
                result = CSUM @ state
                expected_b = (a + b) % 3
                expected_idx = a * 3 + expected_b
                assert np.isclose(np.abs(result[expected_idx]), 1.0)


class TestThreeQutritGates:
    def test_Toffoli3_unitary(self):
        assert np.allclose(Toffoli3 @ Toffoli3.conj().T, np.eye(27))

    def test_Toffoli3_shape(self):
        assert Toffoli3.shape == (27, 27)


class TestCustomGate:
    def test_from_array(self):
        U = np.eye(3, dtype=complex)
        gate = CustomGate(U)
        assert np.allclose(gate.matrix, U)

    def test_validates_unitary(self):
        with pytest.raises(ValueError):
            CustomGate(np.zeros((3, 3), dtype=complex))
