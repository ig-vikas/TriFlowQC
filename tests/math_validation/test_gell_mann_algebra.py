"""
Mathematical validation tests — verify Gell-Mann matrices, SU(3) properties,
and qutrit gate algebra against known analytical results.
"""

import numpy as np
import pytest
from triflowqc.gates.gell_mann import (
    LAMBDA_1, LAMBDA_2, LAMBDA_3, LAMBDA_4,
    LAMBDA_5, LAMBDA_6, LAMBDA_7, LAMBDA_8,
    gell_mann, structure_constants, commutator as gm_comm,
)
from triflowqc.gates.parametric import R_GM
from triflowqc.math.su3 import (
    su3_generators,
    su3_random_element,
    is_su3,
    casimir_operator,
)


class TestGellMannAlgebra:
    """Verify the Gell-Mann matrices satisfy SU(3) Lie algebra relations."""

    def test_tr_lambda_i_lambda_j(self):
        """Tr(λ_i λ_j) = 2δ_{ij}."""
        gens = su3_generators()
        for i in range(8):
            for j in range(8):
                tr = np.trace(gens[i] @ gens[j])
                expected = 2.0 if i == j else 0.0
                assert np.isclose(tr, expected, atol=1e-12), \
                    f"Tr(λ_{i+1}λ_{j+1}) = {tr}, expected {expected}"

    def test_commutation_relations(self):
        """[λ_i, λ_j] = 2i Σ_k f_{ijk} λ_k."""
        gens = su3_generators()
        f = structure_constants()
        for i in range(8):
            for j in range(8):
                comm = gens[i] @ gens[j] - gens[j] @ gens[i]
                expected = 2j * sum(f[i, j, k] * gens[k] for k in range(8))
                assert np.allclose(comm, expected, atol=1e-10), \
                    f"Commutation failed for [λ_{i+1}, λ_{j+1}]"

    def test_known_structure_constants(self):
        """Check specific known f_{ijk} values."""
        f = structure_constants()
        # f_{123} = 1
        assert np.isclose(f[0, 1, 2], 1.0, atol=1e-10)
        # f_{458} = f_{678} = √3/2
        assert np.isclose(f[3, 4, 7], np.sqrt(3) / 2, atol=1e-10)
        assert np.isclose(f[5, 6, 7], np.sqrt(3) / 2, atol=1e-10)
        # f_{147} = f_{246} = f_{257} = f_{345} = 1/2
        assert np.isclose(f[0, 3, 6], 0.5, atol=1e-10)

    def test_structure_constants_totally_antisymmetric(self):
        """f_{ijk} is totally antisymmetric."""
        f = structure_constants()
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    assert np.isclose(f[i, j, k], -f[j, i, k], atol=1e-10)
                    assert np.isclose(f[i, j, k], -f[i, k, j], atol=1e-10)

    def test_casimir_operator_value(self):
        """C₂ = Σ_k λ_k² = (16/3) I for fundamental rep."""
        C = casimir_operator()
        assert np.allclose(C, (16.0 / 3.0) * np.eye(3), atol=1e-10)


class TestParametricGateAlgebra:
    """Verify parametric gate mathematical properties."""

    def test_rotation_at_zero_is_identity(self):
        """R_k(0) = I for all k."""
        for k in range(1, 9):
            assert np.allclose(R_GM(k, 0.0), np.eye(3), atol=1e-12)

    def test_rotation_determinant(self):
        """det(R_k(θ)) = 1 for all k, θ (SU(3) element)."""
        rng = np.random.default_rng(42)
        for k in range(1, 9):
            for _ in range(10):
                theta = rng.uniform(-2 * np.pi, 2 * np.pi)
                U = R_GM(k, theta)
                assert np.isclose(np.abs(np.linalg.det(U)), 1.0, atol=1e-10)

    def test_rotation_unitarity(self):
        """R_k(θ)† R_k(θ) = I."""
        rng = np.random.default_rng(42)
        for k in range(1, 9):
            for _ in range(10):
                theta = rng.uniform(-2 * np.pi, 2 * np.pi)
                U = R_GM(k, theta)
                assert np.allclose(U @ U.conj().T, np.eye(3), atol=1e-10)

    def test_rotation_composition(self):
        """R_k(α) R_k(β) = R_k(α+β) for diagonal generators (k=3,8)."""
        for k in [3, 8]:
            alpha = 1.2
            beta = 0.7
            U1 = R_GM(k, alpha)
            U2 = R_GM(k, beta)
            U_combined = R_GM(k, alpha + beta)
            assert np.allclose(U1 @ U2, U_combined, atol=1e-10)

    def test_rotation_period(self):
        """
        For Gell-Mann generator λ_k with eigenvalues ±1, 0:
        R_k(4π) should be close to identity (period depends on eigenvalues).
        """
        # λ₃ has eigenvalues 1, -1, 0 → period is 4π
        U = R_GM(3, 4 * np.pi)
        assert np.allclose(U, np.eye(3), atol=1e-10)


class TestQutritGateProperties:
    """Test algebraic properties of qutrit gates."""

    def test_x3_cubed_is_identity(self):
        """X₃³ = I (cyclic group of order 3)."""
        from triflowqc.gates.single_qutrit import X3
        assert np.allclose(X3 @ X3 @ X3, np.eye(3), atol=1e-12)

    def test_z3_cubed_is_identity(self):
        """Z₃³ = I."""
        from triflowqc.gates.single_qutrit import Z3
        assert np.allclose(Z3 @ Z3 @ Z3, np.eye(3), atol=1e-12)

    def test_h3_creates_equal_superposition(self):
        """H₃|0⟩ has equal probabilities."""
        from triflowqc.gates.single_qutrit import H3
        state = np.array([1, 0, 0], dtype=complex)
        result = H3 @ state
        probs = np.abs(result) ** 2
        assert np.allclose(probs, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    def test_csum_associativity(self):
        """CSUM|a,b⟩ = |a, (a+b) mod 3⟩ for all basis states."""
        from triflowqc.gates.two_qutrit import CSUM
        for a in range(3):
            for b in range(3):
                state = np.zeros(9, dtype=complex)
                state[a * 3 + b] = 1.0
                result = CSUM @ state
                target = (a + b) % 3
                expected = np.zeros(9, dtype=complex)
                expected[a * 3 + target] = 1.0
                assert np.allclose(result, expected, atol=1e-12), \
                    f"CSUM failed for |{a},{b}⟩"

    def test_cphase_diagonal(self):
        """CPHASE is diagonal with phases ω^{ab}."""
        from triflowqc.gates.two_qutrit import CPHASE
        omega = np.exp(2j * np.pi / 3)
        for a in range(3):
            for b in range(3):
                idx = a * 3 + b
                expected_phase = omega ** (a * b)
                assert np.isclose(CPHASE[idx, idx], expected_phase, atol=1e-12)
                # Off-diagonal should be zero
                for other in range(9):
                    if other != idx:
                        assert np.isclose(CPHASE[idx, other], 0.0, atol=1e-12)
