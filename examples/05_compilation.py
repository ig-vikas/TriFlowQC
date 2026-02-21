"""
Example 5: Qutrit-to-Qubit Compilation

Demonstrates:
  - Compiling qutrit circuits to qubit hardware
  - Encoding: |0⟩→|00⟩, |1⟩→|01⟩, |2⟩→|10⟩
  - Gate embedding and decomposition
  - Leakage detection and handling
  - Result mapping (qubit counts → trit counts)
"""

import numpy as np
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper
from triflowqc.gates.single_qutrit import X3, Z3, H3
from triflowqc.gates.two_qutrit import CSUM, SWAP3

print("=" * 55)
print("  Example 5: Qutrit-to-Qubit Compilation")
print("=" * 55)

# ── Encoding Scheme ──────────────────────────────────
print("\n1. Qutrit-to-Qubit Encoding")
print("  Each qutrit is encoded in 2 physical qubits:")
print("    |0⟩_trit → |00⟩_qubit")
print("    |1⟩_trit → |01⟩_qubit")
print("    |2⟩_trit → |10⟩_qubit")
print("    |11⟩_qubit → FORBIDDEN (leakage)")

compiler = QutritToQubitCompiler()

# Demonstrate encoding
for trit_val, trit_state in [(0, [1,0,0]), (1, [0,1,0]), (2, [0,0,1])]:
    trit_state = np.array(trit_state, dtype=complex)
    qubit_state = compiler.encode_state(trit_state)
    print(f"\n  |{trit_val}⟩_trit → qubit state: {qubit_state}")
    qubit_probs = np.abs(qubit_state) ** 2
    for i, p in enumerate(qubit_probs):
        if p > 0.01:
            print(f"    |{i:02b}⟩: P = {p:.4f}")

# ── Single-Gate Embedding ────────────────────────────
print("\n\n2. Single-Gate Embedding")
for name, gate in [('X3', X3), ('Z3', Z3), ('H3', H3)]:
    embedded = compiler.embed_single_qutrit_gate(gate)
    print(f"\n  {name} (3×3 → 4×4):")

    # Verify unitarity
    err = np.max(np.abs(embedded.conj().T @ embedded - np.eye(4)))
    print(f"    Unitarity error: {err:.2e}")

    # Show action on basis states
    for trit_idx in range(3):
        trit_in = np.zeros(3, dtype=complex)
        trit_in[trit_idx] = 1.0
        qubit_in = compiler.encode_state(trit_in)
        qubit_out = embedded @ qubit_in

        trit_out = gate @ trit_in
        expected = compiler.encode_state(trit_out)

        err = np.max(np.abs(qubit_out - expected))
        print(f"    {name}|{trit_idx}⟩: consistency error = {err:.2e}")

    # Forbidden state
    forbidden = np.array([0, 0, 0, 1], dtype=complex)
    result = embedded @ forbidden
    err = np.max(np.abs(result - forbidden))
    print(f"    |11⟩→|11⟩: {err:.2e} (should be identity)")

# ── Two-Qutrit Gate Embedding ────────────────────────
print("\n\n3. Two-Qutrit Gate Embedding")
for name, gate in [('CSUM', CSUM), ('SWAP3', SWAP3)]:
    embedded = compiler.embed_two_qutrit_gate(gate)
    err = np.max(np.abs(embedded.conj().T @ embedded - np.eye(16)))
    print(f"  {name} (9×9 → 16×16): unitarity error = {err:.2e}")

# ── Compile Full Circuit ─────────────────────────────
print("\n4. Full Circuit Compilation")
qc = QutritCircuit(2)
qc.h(0)
qc.rx(0, np.pi / 4)
qc.csum(0, 1)
qc.rz(1, np.pi / 3)
print(f"  Qutrit circuit:")
print(f"  {qc.draw()}")

qubit_circuit = compiler.compile(qc)
print(f"\n  Compiled to {qubit_circuit.num_qubits} qubits")
print(f"  Qubit circuit depth: {qubit_circuit.depth()}")
print(f"  Number of qubit gates: {len(qubit_circuit)}")

# ── Leakage Handling ─────────────────────────────────
print("\n5. Leakage Detection and Handling")

# Simulate some qubit counts with leakage
raw_counts = {
    '0000': 3000,  # |00⟩|00⟩ = |0⟩|0⟩
    '0001': 2500,  # |00⟩|01⟩ = |0⟩|1⟩
    '0100': 2000,  # |01⟩|00⟩ = |1⟩|0⟩
    '0101': 1500,  # |01⟩|01⟩ = |1⟩|1⟩
    '1100': 500,   # |11⟩|00⟩ = LEAKAGE on qutrit 0
    '0011': 300,   # |00⟩|11⟩ = LEAKAGE on qutrit 1
    '1111': 200,   # |11⟩|11⟩ = LEAKAGE on both
}
total = sum(raw_counts.values())
print(f"  Raw counts ({total} total):")
for state, count in sorted(raw_counts.items()):
    print(f"    |{state}⟩: {count}")

# Detect leakage
handler_ps = LeakageHandler(n_qutrits=2, mode='post_select')
rate = handler_ps.leakage_rate(raw_counts)
print(f"\n  Leakage rate: {rate:.4f} ({rate*100:.2f}%)")

# Post-selection
clean_ps = handler_ps.filter_counts(raw_counts)
print(f"\n  Post-selected counts ({sum(clean_ps.values())} total):")
for state, count in sorted(clean_ps.items()):
    print(f"    |{state}⟩: {count}")

# Nearest mapping
handler_near = LeakageHandler(n_qutrits=2, mode='nearest')
clean_near = handler_near.filter_counts(raw_counts)
print(f"\n  Nearest-mapped counts ({sum(clean_near.values())} total):")
for state, count in sorted(clean_near.items()):
    print(f"    |{state}⟩: {count}")

# ── Result Mapping ───────────────────────────────────
print("\n6. Result Mapping (Qubit → Trit)")
mapper = ResultMapper(n_qutrits=2)
trit_counts = mapper.map_counts(clean_ps)
print(f"  Trit-basis counts:")
for state, count in sorted(trit_counts.items()):
    print(f"    |{state}⟩_trit: {count}")

# Probabilities
trit_result = mapper.map_to_measurement_result(clean_ps)
print(f"\n  Trit probabilities:")
for state, prob in sorted(trit_result.probabilities.items()):
    print(f"    |{state}⟩: {prob:.4f}")

# ── Resource Estimation ──────────────────────────────
print("\n7. Resource Estimation")
for n in [1, 2, 3, 4, 5]:
    qc_test = QutritCircuit(n)
    for i in range(n):
        qc_test.h(i)
    for i in range(n - 1):
        qc_test.csum(i, i + 1)

    print(f"  {n} qutrits → {2*n} physical qubits, "
          f"{n + max(0, n-1)} gates → ~{n*3 + max(0, n-1)*20} qubit gates est.")

print("\n" + "=" * 55)
print("  Example complete!")
print("=" * 55)
