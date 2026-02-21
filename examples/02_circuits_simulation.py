"""
Example 2: Qutrit Circuits and Simulation

Demonstrates:
  - Building qutrit circuits
  - Parametric gates
  - Entangling gates (CSUM)
  - Running on statevector and density matrix simulators
  - Analyzing measurement results
"""

import numpy as np
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator
from triflowqc.circuit.measurement import MeasurementResult

print("=" * 50)
print("  Example 2: Qutrit Circuits and Simulation")
print("=" * 50)

# ── Simple Circuit ───────────────────────────────────
print("\n1. Simple Circuit: X3 Gate")
qc = QutritCircuit(1)
qc.x(0)
print(f"  Circuit:\n{qc.draw()}")

sim = StatevectorSimulator()
result = sim.run(qc, shots=1000)
print(f"  Probabilities: {result.probabilities}")
print(f"  Most likely state: {result.most_likely()}")

# ── Hadamard Superposition Circuit ───────────────────
print("\n2. Hadamard Superposition")
qc = QutritCircuit(2)
qc.h(0)
qc.h(1)
print(f"  Circuit:\n{qc.draw()}")

result = sim.run(qc, shots=10000)
print(f"  Probabilities (should be ~1/9 each):")
for state, prob in sorted(result.probabilities.items()):
    print(f"    |{state}⟩: {prob:.4f}")

# ── Parametric Circuit ───────────────────────────────
print("\n3. Parametric Circuit")
qc = QutritCircuit(1)
theta = np.pi / 4
qc.rx(0, theta)
qc.ry(0, theta / 2)
qc.rz(0, theta / 3)
print(f"  Circuit with rotations θ = π/4:")
print(f"  {qc.draw()}")

result = sim.run(qc, shots=10000)
print(f"  Probabilities: {result.probabilities}")

# ── Entangling Circuit (Qutrit Bell State) ───────────
print("\n4. Qutrit Bell State: H3 + CSUM")
qc = QutritCircuit(2)
qc.h(0)
qc.csum(0, 1)
print(f"  Circuit:\n{qc.draw()}")

result = sim.run(qc, shots=30000)
print(f"  Bell state probabilities:")
print(f"    Should see |00⟩, |11⟩, |22⟩ each ≈ 1/3:")
for state, prob in sorted(result.probabilities.items()):
    print(f"    |{state}⟩: {prob:.4f}")

# Verify entanglement via statevector
sv = result.statevector
print(f"\n  Statevector (non-zero components):")
for i, amp in enumerate(sv):
    if abs(amp) > 1e-6:
        a, b = divmod(i, 3)
        print(f"    |{a}{b}⟩: {amp:.4f} (P = {abs(amp)**2:.4f})")

# ── GHZ-like State ───────────────────────────────────
print("\n5. GHZ-like Qutrit State")
qc = QutritCircuit(3)
qc.h(0)
qc.csum(0, 1)
qc.csum(0, 2)
print(f"  Circuit:\n{qc.draw()}")

result = sim.run(qc, shots=30000)
print(f"  GHZ state: should see |000⟩, |111⟩, |222⟩ each ≈ 1/3:")
for state, prob in sorted(result.probabilities.items()):
    if prob > 0.01:
        print(f"    |{state}⟩: {prob:.4f}")

# ── Density Matrix Simulation ────────────────────────
print("\n6. Density Matrix Simulation")
qc = QutritCircuit(1)
qc.h(0)

dm_sim = DensityMatrixSimulator()
dm_result = dm_sim.run(qc, shots=10000)
print(f"  Density matrix sim probabilities: {dm_result.probabilities}")

# Compare with statevector
sv_result = sim.run(qc, shots=10000)
print(f"  Statevector sim probabilities:    {sv_result.probabilities}")

# ── Circuit Analysis ─────────────────────────────────
print("\n7. Circuit Analysis")
qc = QutritCircuit(3)
for i in range(3):
    qc.h(i)
    qc.rx(i, 0.5 * i)
qc.csum(0, 1)
qc.csum(1, 2)
qc.csum(0, 2)

print(f"  Number of qutrits: {qc.n_qutrits}")
print(f"  Depth: {qc.depth()}")
print(f"  Number of gates: {len(qc.instructions)}")
print(f"  Number of parameters: {len(qc.parameters)}")
print(f"\n  Circuit:\n{qc.draw()}")

# ── Sweeping a Parameter ─────────────────────────────
print("\n8. Parameter Sweep")
print(f"  Sweeping RX3 angle from 0 to 2π:")
print(f"  {'θ':>8} {'P(|0⟩)':>8} {'P(|1⟩)':>8} {'P(|2⟩)':>8}")
for theta in np.linspace(0, 2 * np.pi, 9):
    qc = QutritCircuit(1)
    qc.rx(0, theta)
    result = sim.run(qc)
    sv = result.statevector
    probs = np.abs(sv) ** 2
    print(f"  {theta:>8.3f} {probs[0]:>8.4f} {probs[1]:>8.4f} {probs[2]:>8.4f}")

print("\n" + "=" * 50)
print("  Example complete!")
print("=" * 50)
