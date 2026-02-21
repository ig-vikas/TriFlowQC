"""
Example 4: Noisy Qutrit Simulation

Demonstrates:
  - Building noise models with various channels
  - Running noisy simulations
  - Comparing noisy vs noiseless results
  - Readout error modeling
  - Information-theoretic analysis of noise effects
"""

import numpy as np
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator
from triflowqc.simulator.noise_model import (
    NoiseModel,
    depolarising_channel,
    dephasing_channel,
    amplitude_damping_channel,
    trit_flip_channel,
)
from triflowqc.simulator.noisy_simulator import NoisySimulator
from triflowqc.math.information import von_neumann_entropy, purity

print("=" * 50)
print("  Example 4: Noisy Qutrit Simulation")
print("=" * 50)

# ── Noiseless Baseline ───────────────────────────────
print("\n1. Noiseless Baseline: Bell State")
qc = QutritCircuit(2)
qc.h(0)
qc.csum(0, 1)

sim = StatevectorSimulator()
result_clean = sim.run(qc, shots=10000)
print(f"  Noiseless probabilities:")
for state, prob in sorted(result_clean.probabilities.items()):
    if prob > 0.01:
        print(f"    |{state}⟩: {prob:.4f}")

# ── Depolarising Noise ───────────────────────────────
print("\n2. Depolarising Noise")
for p in [0.01, 0.05, 0.1, 0.2]:
    noise = NoiseModel()
    noise.add_gate_error('H3', depolarising_channel(p))
    noise.add_gate_error('CSUM', depolarising_channel(p, n_qutrits=2))
    noisy_sim = NoisySimulator(noise)
    result_noisy = noisy_sim.run(qc, shots=10000)

    # Dominant probabilities
    dominant = {k: v for k, v in result_noisy.probabilities.items() if v > 0.01}
    p_corr = sum(result_noisy.probabilities.get(s, 0) for s in ['00', '11', '22'])
    print(f"  p={p:.2f}: P(correlated) = {p_corr:.4f}")

# ── Dephasing Noise ──────────────────────────────────
print("\n3. Dephasing Noise")
qc_sup = QutritCircuit(1)
qc_sup.h(0)

dm_sim = DensityMatrixSimulator()
rho_pure = dm_sim.get_density_matrix(qc_sup)
print(f"  Pure state purity: {purity(rho_pure):.6f}")
print(f"  Pure state entropy: {von_neumann_entropy(rho_pure):.6f}")

for gamma in [0.1, 0.3, 0.5, 0.9]:
    noise = NoiseModel()
    noise.add_gate_error('H3', dephasing_channel(gamma))
    noisy_sim = NoisySimulator(noise)
    rho_noisy = noisy_sim.get_density_matrix(qc_sup)
    print(f"  γ={gamma:.1f}: purity={purity(rho_noisy):.4f}, "
          f"entropy={von_neumann_entropy(rho_noisy):.4f}")

# ── Amplitude Damping ────────────────────────────────
print("\n4. Amplitude Damping (T1 decay)")
qc_excited = QutritCircuit(1)
qc_excited.x(0)  # |1⟩

for gamma in [0.1, 0.3, 0.5, 0.8, 1.0]:
    noise = NoiseModel()
    noise.add_gate_error('X3', amplitude_damping_channel(gamma))
    noisy_sim = NoisySimulator(noise)
    result = noisy_sim.run(qc_excited, shots=10000)
    p0 = result.probabilities.get('0', 0)
    p1 = result.probabilities.get('1', 0)
    p2 = result.probabilities.get('2', 0)
    print(f"  γ={gamma:.1f}: P(|0⟩)={p0:.4f}, P(|1⟩)={p1:.4f}, P(|2⟩)={p2:.4f}")

# ── Trit-Flip Channel ───────────────────────────────
print("\n5. Trit-Flip Channel")
qc_zero = QutritCircuit(1)  # |0⟩

for p in [0.01, 0.1, 0.3, 0.5]:
    noise = NoiseModel()
    noise.add_local_error(0, trit_flip_channel(p))
    noisy_sim = NoisySimulator(noise)
    result = noisy_sim.run(qc_zero, shots=10000)
    probs = result.probabilities
    print(f"  p={p:.2f}: P(|0⟩)={probs.get('0',0):.4f}, "
          f"P(|1⟩)={probs.get('1',0):.4f}, P(|2⟩)={probs.get('2',0):.4f}")

# ── Readout Error ────────────────────────────────────
print("\n6. Readout Error")
qc_det = QutritCircuit(1)
qc_det.x(0)  # Deterministic |1⟩

# Perfect readout
noise_perfect = NoiseModel()
sim_perfect = NoisySimulator(noise_perfect)
result_perfect = sim_perfect.run(qc_det, shots=10000)
print(f"  Perfect readout: {result_perfect.probabilities}")

# Noisy readout (10% error)
noise_readout = NoiseModel()
confusion = np.array([
    [0.90, 0.05, 0.05],
    [0.05, 0.90, 0.05],
    [0.05, 0.05, 0.90],
])
noise_readout.add_readout_error(0, confusion)
sim_readout = NoisySimulator(noise_readout)
result_readout = sim_readout.run(qc_det, shots=10000)
print(f"  10% readout error: {result_readout.probabilities}")

# Heavy readout error (30%)
noise_heavy = NoiseModel()
confusion_heavy = np.array([
    [0.70, 0.15, 0.15],
    [0.15, 0.70, 0.15],
    [0.15, 0.15, 0.70],
])
noise_heavy.add_readout_error(0, confusion_heavy)
sim_heavy = NoisySimulator(noise_heavy)
result_heavy = sim_heavy.run(qc_det, shots=10000)
print(f"  30% readout error: {result_heavy.probabilities}")

# ── Combined Noise ───────────────────────────────────
print("\n7. Combined Noise Model")
qc_bell = QutritCircuit(2)
qc_bell.h(0)
qc_bell.csum(0, 1)

noise_combined = NoiseModel()
noise_combined.add_gate_error('H3', depolarising_channel(0.02))
noise_combined.add_gate_error('CSUM', depolarising_channel(0.05, n_qutrits=2))
confusion = np.array([[0.95, 0.025, 0.025],
                       [0.025, 0.95, 0.025],
                       [0.025, 0.025, 0.95]])
noise_combined.add_readout_error(0, confusion)
noise_combined.add_readout_error(1, confusion)

sim_combined = NoisySimulator(noise_combined)
result_combined = sim_combined.run(qc_bell, shots=30000)

print(f"  Combined noise model results:")
for state, prob in sorted(result_combined.probabilities.items()):
    if prob > 0.01:
        print(f"    |{state}⟩: {prob:.4f}")

# Compare fidelity
p_corr_noisy = sum(result_combined.probabilities.get(s, 0) for s in ['00', '11', '22'])
p_corr_clean = sum(result_clean.probabilities.get(s, 0) for s in ['00', '11', '22'])
print(f"\n  P(correlated) noiseless: {p_corr_clean:.4f}")
print(f"  P(correlated) noisy:     {p_corr_noisy:.4f}")

print("\n" + "=" * 50)
print("  Example complete!")
print("=" * 50)
