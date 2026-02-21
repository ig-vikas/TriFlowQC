"""
Example 1: Basic Qutrit Operations

Demonstrates fundamental qutrit operations:
  - Creating qutrits and registers
  - Applying single-qutrit gates (X3, Z3, H3)
  - Measuring states
  - Computing fidelity and Bloch vectors
"""

import numpy as np
from triflowqc.core.qutrit import Qutrit
from triflowqc.core.register import QutritRegister
from triflowqc.core.state_vector import StateVector
from triflowqc.core.density_matrix import DensityMatrix

print("=" * 50)
print("  Example 1: Basic Qutrit Operations")
print("=" * 50)

# ── Create a single qutrit ──────────────────────────
print("\n1. Single Qutrit Creation")
q = Qutrit()  # Default |0⟩ state
print(f"  State: {q.state}")
print(f"  Probabilities: {np.abs(q.state)**2}")

# Create in |1⟩ state
q1 = Qutrit(state=np.array([0, 1, 0], dtype=complex))
print(f"\n  |1⟩ state: {q1.state}")
print(f"  Probabilities: {np.abs(q1.state)**2}")

# ── Apply single-qutrit gates ───────────────────────
print("\n2. Single-Qutrit Gates")
from triflowqc.gates.single_qutrit import X3, Z3, H3

# X3 (shift) gate: |0⟩ → |1⟩ → |2⟩ → |0⟩
q = Qutrit()
q.apply_gate(X3)
print(f"  X3|0⟩ = |1⟩: probabilities = {np.abs(q.state)**2}")

q.apply_gate(X3)
print(f"  X3|1⟩ = |2⟩: probabilities = {np.abs(q.state)**2}")

q.apply_gate(X3)
print(f"  X3|2⟩ = |0⟩: probabilities = {np.abs(q.state)**2}")

# Z3 (clock) gate: adds phase ω^k to |k⟩
q = Qutrit(state=np.array([0, 1, 0], dtype=complex))
q.apply_gate(Z3)
print(f"\n  Z3|1⟩: {q.state}")
omega = np.exp(2j * np.pi / 3)
print(f"  Expected phase: ω = {omega:.4f}")

# H3 (Hadamard/QFT) gate: creates superposition
q = Qutrit()
q.apply_gate(H3)
print(f"\n  H3|0⟩: probabilities = {np.abs(q.state)**2}")
print(f"  Equal superposition: all ≈ 1/3 = {1/3:.4f}")

# ── Measurement ──────────────────────────────────────
print("\n3. Measurement")
q = Qutrit()
q.apply_gate(H3)
outcomes = [q.copy().measure() for _ in range(1000)]
for k in [0, 1, 2]:
    freq = outcomes.count(k) / len(outcomes)
    print(f"  P(|{k}⟩) = {freq:.3f} (expected ≈ 0.333)")

# ── Qutrit Register ─────────────────────────────────
print("\n4. Qutrit Register")
reg = QutritRegister(3)
print(f"  3-qutrit register: dimension = {len(reg.state)}")
print(f"  Initial state |000⟩: P(|000⟩) = {np.abs(reg.state[0])**2:.4f}")

# Apply X3 to first qutrit
reg.apply_gate(X3, [0])
idx_100 = 1 * 9 + 0 * 3 + 0  # |100⟩ index = 9
print(f"  After X3 on q0: P(|100⟩) = {np.abs(reg.state[idx_100])**2:.4f}")

# ── Fidelity ─────────────────────────────────────────
print("\n5. Fidelity")
q_a = Qutrit()  # |0⟩
q_b = Qutrit()  # |0⟩
print(f"  F(|0⟩, |0⟩) = {q_a.fidelity(q_b):.4f}")

q_b.apply_gate(X3)  # |1⟩
print(f"  F(|0⟩, |1⟩) = {q_a.fidelity(q_b):.4f}")

q_b = Qutrit()
q_b.apply_gate(H3)  # Superposition
print(f"  F(|0⟩, H3|0⟩) = {q_a.fidelity(q_b):.4f}")

# ── Bloch Vector (generalized) ──────────────────────
print("\n6. Bloch Vector (Generalized)")
q = Qutrit()
bloch = q.bloch_vector()
print(f"  |0⟩ Bloch vector (8 components): {bloch.round(4)}")

q.apply_gate(H3)
bloch = q.bloch_vector()
print(f"  H3|0⟩ Bloch vector: {bloch.round(4)}")

# ── Density Matrix ───────────────────────────────────
print("\n7. Density Matrix")
psi = np.array([1, 0, 0], dtype=complex)
rho = DensityMatrix.from_statevector(psi)
print(f"  |0⟩ density matrix:")
print(f"  {rho.matrix.real}")

psi_sup = np.array([1, 1, 1], dtype=complex) / np.sqrt(3)
rho_sup = DensityMatrix.from_statevector(psi_sup)
print(f"\n  Maximally superposed density matrix:")
print(f"  {rho_sup.matrix.real.round(4)}")

print(f"\n  Purity of |0⟩: {rho.purity():.4f}")
print(f"  Purity of superposition: {rho_sup.purity():.4f}")

print("\n" + "=" * 50)
print("  Example complete!")
print("=" * 50)
