"""Unit tests for math module."""

import numpy as np
import pytest
from triflowqc.math.su3 import (
    su3_generators,
    su3_random_element,
    is_su3,
    su3_basis_decompose,
    su3_reconstruct,
    su3_adjoint_representation,
    casimir_operator,
)
from triflowqc.math.algebra import (
    tensor_product,
    partial_trace_mixed,
    commutator,
    fidelity_pure,
    fidelity_mixed,
    trace_distance,
    is_unitary,
    is_hermitian,
)
from triflowqc.math.information import (
    von_neumann_entropy,
    linear_entropy,
    negativity,
    max_entangled_state,
)
from triflowqc.math.decompositions import (
    schmidt_decomposition,
    schmidt_rank,
    bloch_vector_qutrit,
)


class TestSU3:
    def test_generators_count(self):
        gens = su3_generators()
        assert len(gens) == 8

    def test_random_element_is_su3(self):
        U = su3_random_element()
        assert is_su3(U)

    def test_decompose_reconstruct(self):
        """Decompose and reconstruct a traceless Hermitian matrix."""
        gens = su3_generators()
        # Random traceless Hermitian
        coeffs = np.random.randn(8)
        M = sum(c * g for c, g in zip(coeffs, gens))
        recovered = su3_basis_decompose(M)
        M2 = su3_reconstruct(recovered)
        assert np.allclose(M, M2, atol=1e-10)

    def test_adjoint_representation(self):
        U = su3_random_element()
        ad = su3_adjoint_representation(U)
        assert ad.shape == (8, 8)
        # Should be orthogonal (for SU(3))
        assert np.allclose(ad @ ad.T, np.eye(8), atol=1e-6)

    def test_casimir(self):
        C = casimir_operator()
        # For fundamental rep: C₂ = (16/3)I
        assert np.allclose(C, (16.0 / 3.0) * np.eye(3), atol=1e-10)


class TestAlgebra:
    def test_tensor_product(self):
        A = np.eye(3, dtype=complex)
        B = np.eye(3, dtype=complex)
        C = tensor_product(A, B)
        assert C.shape == (9, 9)
        assert np.allclose(C, np.eye(9))

    def test_partial_trace(self):
        # Maximally mixed state for 2 qutrits
        rho = np.eye(9, dtype=complex) / 9.0
        rho_a = partial_trace_mixed(rho, keep=[0], dims=[3, 3])
        assert np.allclose(rho_a, np.eye(3) / 3.0)

    def test_commutator(self):
        from triflowqc.gates.gell_mann import LAMBDA_1, LAMBDA_2
        c = commutator(LAMBDA_1, LAMBDA_2)
        # [λ₁, λ₂] = 2i λ₃
        from triflowqc.gates.gell_mann import LAMBDA_3
        assert np.allclose(c, 2j * LAMBDA_3, atol=1e-10)

    def test_fidelity_pure_same(self):
        psi = np.array([1, 0, 0], dtype=complex)
        assert np.isclose(fidelity_pure(psi, psi), 1.0)

    def test_fidelity_pure_orthogonal(self):
        psi = np.array([1, 0, 0], dtype=complex)
        phi = np.array([0, 1, 0], dtype=complex)
        assert np.isclose(fidelity_pure(psi, phi), 0.0)

    def test_fidelity_mixed(self):
        rho = np.diag([1.0, 0, 0]).astype(complex)
        assert np.isclose(fidelity_mixed(rho, rho), 1.0)

    def test_trace_distance(self):
        rho = np.diag([1.0, 0, 0]).astype(complex)
        sigma = np.diag([0, 1.0, 0]).astype(complex)
        assert np.isclose(trace_distance(rho, sigma), 1.0)


class TestInformation:
    def test_entropy_pure(self):
        rho = np.diag([1.0, 0, 0]).astype(complex)
        assert np.isclose(von_neumann_entropy(rho), 0.0)

    def test_entropy_maximally_mixed(self):
        rho = np.eye(3, dtype=complex) / 3.0
        S = von_neumann_entropy(rho)
        assert np.isclose(S, np.log2(3), atol=1e-10)

    def test_linear_entropy_pure(self):
        rho = np.diag([1.0, 0, 0]).astype(complex)
        assert np.isclose(linear_entropy(rho), 0.0)

    def test_max_entangled(self):
        psi = max_entangled_state(3)
        assert np.isclose(np.linalg.norm(psi), 1.0)
        assert len(psi) == 9


class TestDecompositions:
    def test_schmidt_product_state(self):
        psi = np.zeros(9, dtype=complex)
        psi[0] = 1.0  # |00⟩
        coeffs, _, _ = schmidt_decomposition(psi, (3, 3))
        assert len(coeffs) == 1
        assert np.isclose(coeffs[0], 1.0)

    def test_schmidt_entangled(self):
        psi = max_entangled_state(3)
        rank = schmidt_rank(psi, (3, 3))
        assert rank == 3

    def test_bloch_vector(self):
        rho = np.diag([1.0, 0, 0]).astype(complex)
        bv = bloch_vector_qutrit(rho)
        assert bv.shape == (8,)
