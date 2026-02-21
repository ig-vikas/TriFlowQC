"""
Example 6: SU(3) Mathematics and Quantum Information

Demonstrates:
  - Gell-Mann matrices and SU(3) algebra
  - Structure constants and commutation relations
  - Random SU(3) elements
  - Von Neumann entropy and purity
  - Entanglement measures
  - Partial trace and tensor products
  - Schmidt decomposition
"""

import numpy as np
from triflowqc.math.su3 import (
    generators,
    structure_constants,
    random_element,
    is_su3,
    basis_decompose,
    reconstruct,
    su3_exponential,
    casimir_operator,
    adjoint_representation,
)
from triflowqc.math.algebra import (
    tensor_product,
    partial_trace_pure,
    partial_trace_mixed,
    commutator,
    anticommutator,
    fidelity_pure,
    fidelity_mixed,
    trace_distance,
    is_unitary,
    is_hermitian,
)
from triflowqc.math.information import (
    von_neumann_entropy,
    linear_entropy,
    purity,
    mutual_information,
    negativity,
    partial_transpose,
    max_entangled_state,
)
from triflowqc.math.decompositions import (
    schmidt_decomposition,
    schmidt_rank,
    bloch_vector_qutrit,
)

print("=" * 55)
print("  Example 6: SU(3) Mathematics & Quantum Information")
print("=" * 55)

# ── Gell-Mann Matrices ───────────────────────────────
print("\n1. Gell-Mann Matrices (SU(3) Generators)")
lambdas = generators()
print(f"  Number of generators: {len(lambdas)}")

for i, lam in enumerate(lambdas):
    tr = np.trace(lam)
    herm_err = np.max(np.abs(lam - lam.conj().T))
    print(f"  λ_{i+1}: Tr={tr.real:.4f}, Hermitian error={herm_err:.2e}")

# Trace orthonormality
print("\n  Trace orthonormality Tr(λ_i λ_j) = 2δ_ij:")
ortho_err = 0
for i in range(8):
    for j in range(8):
        tr = np.trace(lambdas[i] @ lambdas[j]).real
        expected = 2.0 if i == j else 0.0
        ortho_err = max(ortho_err, abs(tr - expected))
print(f"  Max orthonormality error: {ortho_err:.2e}")

# ── Structure Constants ──────────────────────────────
print("\n2. Structure Constants f_ijk")
f = structure_constants()
print(f"  Shape: {f.shape}")
print(f"  Known values:")
known = [(1,2,3,1.0), (1,4,7,0.5), (2,4,6,0.5), (3,4,5,0.5),
         (1,5,6,-0.5), (2,5,7,0.5), (3,6,7,-0.5)]
for i, j, k, expected in known:
    actual = f[i-1, j-1, k-1]
    print(f"    f_{i}{j}{k} = {actual:.4f} (expected {expected:.4f})")

# ── Random SU(3) Elements ────────────────────────────
print("\n3. Random SU(3) Elements")
np.random.seed(42)
for trial in range(5):
    U = random_element()
    det = np.linalg.det(U)
    uu_err = np.max(np.abs(U.conj().T @ U - np.eye(3)))
    print(f"  Trial {trial+1}: det={det:.6f}, U†U error={uu_err:.2e}, "
          f"is_su3={is_su3(U)}")

# ── Basis Decomposition ─────────────────────────────
print("\n4. SU(3) Basis Decomposition")
U = random_element()
coeffs = basis_decompose(U)
U_recon = reconstruct(coeffs)
recon_err = np.max(np.abs(U - U_recon))
print(f"  Decomposition coefficients: {coeffs.round(4)}")
print(f"  Reconstruction error: {recon_err:.2e}")

# ── Casimir Operator ─────────────────────────────────
print("\n5. Casimir Operator")
C = casimir_operator()
print(f"  C₂ = Σ λ_k²:")
print(f"  {C.real.round(4)}")
# Should be proportional to identity: C₂ = (4/3)I for fundamental rep
eigs = np.linalg.eigvalsh(C)
print(f"  Eigenvalues: {eigs.round(4)} (should all be {4/3:.4f})")

# ── Entropy and Purity ───────────────────────────────
print("\n6. Von Neumann Entropy and Purity")

# Pure state
psi = np.array([1, 0, 0], dtype=complex)
rho_pure = np.outer(psi, psi.conj())
print(f"  Pure |0⟩:")
print(f"    S = {von_neumann_entropy(rho_pure):.6f}")
print(f"    Purity = {purity(rho_pure):.6f}")

# Maximally mixed
rho_mixed = np.eye(3) / 3
print(f"\n  Maximally mixed I/3:")
print(f"    S = {von_neumann_entropy(rho_mixed):.6f} (max = ln(3) = {np.log(3):.6f})")
print(f"    Purity = {purity(rho_mixed):.6f} (min = 1/3 = {1/3:.6f})")

# Partial mixture
rho_partial = 0.7 * np.outer([1,0,0], [1,0,0]) + 0.3 * np.eye(3) / 3
print(f"\n  70% |0⟩ + 30% mixed:")
print(f"    S = {von_neumann_entropy(rho_partial):.6f}")
print(f"    Purity = {purity(rho_partial):.6f}")
print(f"    Linear entropy = {linear_entropy(rho_partial):.6f}")

# ── Entanglement ─────────────────────────────────────
print("\n7. Entanglement Measures")

# Separable state |00⟩
psi_sep = np.zeros(9, dtype=complex)
psi_sep[0] = 1.0
rho_sep = np.outer(psi_sep, psi_sep.conj())

# Bell-like state (|00⟩ + |11⟩ + |22⟩)/√3
psi_bell = np.zeros(9, dtype=complex)
psi_bell[0] = 1/np.sqrt(3)  # |00⟩
psi_bell[4] = 1/np.sqrt(3)  # |11⟩
psi_bell[8] = 1/np.sqrt(3)  # |22⟩
rho_bell = np.outer(psi_bell, psi_bell.conj())

print(f"  Separable state |00⟩:")
rho_A_sep = partial_trace_mixed(rho_sep, [1], d=3, n=2)
print(f"    S(ρ_A) = {von_neumann_entropy(rho_A_sep):.6f}")
print(f"    Negativity = {negativity(rho_sep, d=3):.6f}")

print(f"\n  Maximally entangled (|00⟩+|11⟩+|22⟩)/√3:")
rho_A_bell = partial_trace_mixed(rho_bell, [1], d=3, n=2)
print(f"    S(ρ_A) = {von_neumann_entropy(rho_A_bell):.6f} "
      f"(max = ln(3) = {np.log(3):.6f})")
print(f"    Negativity = {negativity(rho_bell, d=3):.6f}")

# ── Schmidt Decomposition ────────────────────────────
print("\n8. Schmidt Decomposition")
# Separable state
coeffs_sep, u_sep, v_sep = schmidt_decomposition(psi_sep, 3, 3)
print(f"  Separable |00⟩:")
print(f"    Schmidt coefficients: {coeffs_sep.round(4)}")
print(f"    Schmidt rank: {schmidt_rank(psi_sep, 3, 3)}")

# Entangled state
coeffs_bell, u_bell, v_bell = schmidt_decomposition(psi_bell, 3, 3)
print(f"\n  Entangled (|00⟩+|11⟩+|22⟩)/√3:")
print(f"    Schmidt coefficients: {coeffs_bell.round(4)}")
print(f"    Schmidt rank: {schmidt_rank(psi_bell, 3, 3)}")

# ── Generalized Bloch Vector ─────────────────────────
print("\n9. Generalized Bloch Vector (Qutrit)")
rho_0 = np.outer([1,0,0], np.conj([1,0,0]))
bloch_0 = bloch_vector_qutrit(rho_0)
print(f"  |0⟩ Bloch vector: {bloch_0.round(4)}")

rho_1 = np.outer([0,1,0], np.conj([0,1,0]))
bloch_1 = bloch_vector_qutrit(rho_1)
print(f"  |1⟩ Bloch vector: {bloch_1.round(4)}")

rho_2 = np.outer([0,0,1], np.conj([0,0,1]))
bloch_2 = bloch_vector_qutrit(rho_2)
print(f"  |2⟩ Bloch vector: {bloch_2.round(4)}")

# ── Tensor Products and Partial Trace ────────────────
print("\n10. Tensor Products and Partial Trace")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
AB = tensor_product(A, B)
print(f"  A⊗B shape: {AB.shape}")
print(f"  A⊗B:\n{AB}")

# Partial trace verification
psi = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=complex) / np.sqrt(3)
rho = np.outer(psi, psi.conj())
rho_A = partial_trace_mixed(rho, [1], d=3, n=2)
print(f"\n  Partial trace TrB(ρ):")
print(f"    Shape: {rho_A.shape}")
print(f"    Trace: {np.trace(rho_A).real:.4f} (should be 1)")
print(f"    Hermitian: {is_hermitian(rho_A)}")

# ── Fidelity and Distance ────────────────────────────
print("\n11. Fidelity and Trace Distance")
psi_a = np.array([1, 0, 0], dtype=complex)
psi_b = np.array([0, 1, 0], dtype=complex)
psi_c = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)

print(f"  F(|0⟩, |0⟩) = {fidelity_pure(psi_a, psi_a):.4f}")
print(f"  F(|0⟩, |1⟩) = {fidelity_pure(psi_a, psi_b):.4f}")
print(f"  F(|0⟩, |+⟩) = {fidelity_pure(psi_a, psi_c):.4f}")

rho_a = np.outer(psi_a, psi_a.conj())
rho_b = np.outer(psi_b, psi_b.conj())
print(f"\n  D(|0⟩, |1⟩) = {trace_distance(rho_a, rho_b):.4f}")
print(f"  F_mixed(|0⟩, |1⟩) = {fidelity_mixed(rho_a, rho_b):.4f}")

print("\n" + "=" * 55)
print("  Example complete!")
print("=" * 55)
