"""
Example 7: Running on IBM Quantum Hardware (Demonstration)

Demonstrates the full pipeline for running qutrit circuits on IBM hardware:
  1. Build qutrit circuit
  2. Compile to qubit circuit
  3. Submit to IBM Quantum (requires API token)
  4. Handle leakage in results
  5. Map qubit results back to trit basis

NOTE: This example requires:
  - pip install qiskit qiskit-ibm-runtime
  - A valid IBM Quantum API token
  - Set IBM_QUANTUM_TOKEN environment variable or pass token directly

Without hardware access, the compilation and result mapping steps
are still demonstrated with simulated data.
"""

import numpy as np
import os
import sys

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper

print("=" * 55)
print("  Example 7: IBM Quantum Hardware Pipeline")
print("=" * 55)

# ── Step 1: Build Qutrit Circuit ─────────────────────
print("\n1. Build Qutrit Circuit")
qc = QutritCircuit(2)
qc.h(0)
qc.csum(0, 1)
print(f"  Circuit: Qutrit Bell State")
print(f"  {qc.draw()}")
print(f"  Qutrits: {qc.n_qutrits}")
print(f"  Gates: {len(qc.instructions)}")

# ── Step 2: Simulate Locally ─────────────────────────
print("\n2. Local Simulation (reference)")
sim = StatevectorSimulator()
result_sim = sim.run(qc, shots=10000)
print(f"  Simulated probabilities:")
for state, prob in sorted(result_sim.probabilities.items()):
    if prob > 0.01:
        print(f"    |{state}⟩: {prob:.4f}")

# ── Step 3: Compile to Qubit Circuit ─────────────────
print("\n3. Compile to Qubit Circuit")
compiler = QutritToQubitCompiler()
qubit_circuit = compiler.compile(qc)
print(f"  Physical qubits: {qubit_circuit.num_qubits}")
print(f"  Circuit depth: {qubit_circuit.depth()}")
print(f"  Gate count: {len(qubit_circuit)}")

# ── Step 4: Check Hardware Availability ──────────────
print("\n4. Hardware Backend")
token = os.environ.get('IBM_QUANTUM_TOKEN', None)

if token:
    print("  IBM Quantum token found. Attempting connection...")
    try:
        from triflowqc.hardware.ibm_backend import IBMBackend
        from triflowqc.hardware.job_manager import JobManager
        from triflowqc.hardware.hardware_runner import HardwareRunner

        backend = IBMBackend(token=token, backend_name='ibm_brisbane')
        print(f"  Connected to: {backend.backend.name}")
        print(f"  Qubits available: {backend.backend.num_qubits}")

        # Submit job
        runner = HardwareRunner(backend=backend, n_qutrits=2)
        print("\n  Submitting circuit to hardware...")
        hw_result = runner.run(qc, shots=4096)
        print(f"\n  Hardware results:")
        for state, prob in sorted(hw_result.probabilities.items()):
            if prob > 0.01:
                print(f"    |{state}⟩: {prob:.4f}")

    except Exception as e:
        print(f"  Hardware error: {e}")
        print("  Falling back to simulated hardware results...")
        token = None

if not token:
    print("  No IBM Quantum token found.")
    print("  Demonstrating pipeline with simulated hardware-like results.")

    # Simulate what hardware results look like
    # (with realistic noise and leakage)
    print("\n  Simulated raw qubit counts (with noise and leakage):")
    raw_qubit_counts = {
        '0000': 1350,   # |00⟩|00⟩ = |0⟩|0⟩ ✓
        '0101': 1280,   # |01⟩|01⟩ = |1⟩|1⟩ ✓
        '1010': 1150,   # |10⟩|10⟩ = |2⟩|2⟩ ✓
        '0001': 85,     # |00⟩|01⟩ = |0⟩|1⟩ (noise)
        '0100': 70,     # |01⟩|00⟩ = |1⟩|0⟩ (noise)
        '1000': 60,     # |10⟩|00⟩ = |2⟩|0⟩ (noise)
        '0010': 45,     # |00⟩|10⟩ = |0⟩|2⟩ (noise)
        '1001': 40,     # |10⟩|01⟩ = |2⟩|1⟩ (noise)
        '0110': 35,     # |01⟩|10⟩ = |1⟩|2⟩ (noise)
        '1100': 30,     # |11⟩|00⟩ = LEAKAGE
        '0011': 25,     # |00⟩|11⟩ = LEAKAGE
        '1111': 10,     # |11⟩|11⟩ = LEAKAGE
        '1101': 8,      # |11⟩|01⟩ = LEAKAGE
        '0111': 6,      # |01⟩|11⟩ = LEAKAGE
        '1110': 5,      # |11⟩|10⟩ = LEAKAGE
        '1011': 4,      # |10⟩|11⟩ = LEAKAGE
    }
    total = sum(raw_qubit_counts.values())
    for state, count in sorted(raw_qubit_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    |{state}⟩: {count} ({count/total*100:.1f}%)")
    print(f"    ... ({total} total shots)")

# ── Step 5: Leakage Handling ─────────────────────────
print("\n5. Leakage Detection and Handling")
handler = LeakageHandler(n_qutrits=2, mode='post_select')

if 'raw_qubit_counts' not in dir():
    raw_qubit_counts = {
        '0000': 1350, '0101': 1280, '1010': 1150,
        '0001': 85, '0100': 70, '1000': 60, '0010': 45,
        '1001': 40, '0110': 35, '1100': 30, '0011': 25,
        '1111': 10, '1101': 8, '0111': 6, '1110': 5, '1011': 4,
    }

rate = handler.leakage_rate(raw_qubit_counts)
print(f"  Leakage rate: {rate:.4f} ({rate*100:.2f}%)")

clean_counts = handler.filter_counts(raw_qubit_counts)
clean_total = sum(clean_counts.values())
print(f"  After post-selection: {clean_total} valid shots")

# ── Step 6: Result Mapping ───────────────────────────
print("\n6. Map Qubit Results → Trit Basis")
mapper = ResultMapper(n_qutrits=2)
trit_result = mapper.map_to_measurement_result(clean_counts)

print(f"  Trit-basis probabilities:")
for state, prob in sorted(trit_result.probabilities.items()):
    if prob > 0.005:
        marker = " ← target" if state in ['00', '11', '22'] else ""
        print(f"    |{state}⟩: {prob:.4f}{marker}")

# ── Step 7: Compare with Theory ──────────────────────
print("\n7. Comparison with Theory")
print(f"  {'State':>6} {'Theory':>8} {'Hardware':>10} {'Error':>8}")
print(f"  {'-'*6:>6} {'-'*8:>8} {'-'*10:>10} {'-'*8:>8}")
for state in ['00', '11', '22', '01', '02', '10', '12', '20', '21']:
    theory = result_sim.probabilities.get(state, 0)
    hw = trit_result.probabilities.get(state, 0)
    err = abs(theory - hw)
    if theory > 0.01 or hw > 0.01:
        print(f"  |{state}⟩  {theory:>8.4f} {hw:>10.4f} {err:>8.4f}")

# ── Step 8: Resource Summary ─────────────────────────
print("\n8. Resource Summary")
print(f"  Qutrit circuit:")
print(f"    Qutrits: {qc.n_qutrits}")
print(f"    Gates: {len(qc.instructions)}")
print(f"    Depth: {qc.depth()}")
print(f"  Compiled qubit circuit:")
print(f"    Physical qubits: {qubit_circuit.num_qubits}")
print(f"    Gate count: {len(qubit_circuit)}")
print(f"    Depth: {qubit_circuit.depth()}")
print(f"  Processing:")
print(f"    Shots: {total}")
print(f"    Leakage rate: {rate:.4f}")
print(f"    Valid shots: {clean_total}")

print("\n" + "=" * 55)
print("  Example complete!")
print("=" * 55)
