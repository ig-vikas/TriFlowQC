"""
Validation script: Verify qutrit-to-qubit encoding correctness.

Checks:
  - Single-qutrit encoding: |0⟩→|00⟩, |1⟩→|01⟩, |2⟩→|10⟩
  - Forbidden state isolation: |11⟩ never produced from valid operations
  - Multi-qutrit encoding consistency
  - Embedded gate unitarity and action preservation
  - Superposition encoding fidelity
  - Leakage handler effectiveness
"""

import numpy as np
import sys

from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper
from triflowqc.gates.single_qutrit import X3, Z3, H3
from triflowqc.gates.two_qutrit import CSUM, SWAP3
from triflowqc.gates.parametric import RX3, RY3, RZ3


def validate_encoding_map():
    """Validate the fundamental encoding map."""
    print("\n=== Encoding Map ===")
    compiler = QutritToQubitCompiler()
    all_pass = True

    # |0⟩ → |00⟩
    trit_0 = np.array([1, 0, 0], dtype=complex)
    qubit_00 = np.array([1, 0, 0, 0], dtype=complex)

    # |1⟩ → |01⟩
    trit_1 = np.array([0, 1, 0], dtype=complex)
    qubit_01 = np.array([0, 1, 0, 0], dtype=complex)

    # |2⟩ → |10⟩
    trit_2 = np.array([0, 0, 1], dtype=complex)
    qubit_10 = np.array([0, 0, 1, 0], dtype=complex)

    encodings = [
        ('|0⟩→|00⟩', trit_0, qubit_00),
        ('|1⟩→|01⟩', trit_1, qubit_01),
        ('|2⟩→|10⟩', trit_2, qubit_10),
    ]

    for name, trit_state, expected_qubit in encodings:
        # Embed the trit state
        embedded = np.zeros(4, dtype=complex)
        for i in range(3):
            if abs(trit_state[i]) > 1e-12:
                embedded += trit_state[i] * np.array(
                    [1 if j == [0, 1, 2][i] else 0 for j in range(4)],
                    dtype=complex
                )
        # The encoding maps: trit index 0→qubit 0, 1→qubit 1, 2→qubit 2
        # Actually using the compiler's encoding
        encoded = compiler.encode_state(trit_state)
        err = np.max(np.abs(encoded - expected_qubit))
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  [{status}] {name} (error = {err:.2e})")
        all_pass = all_pass and (err < 1e-10)

    return all_pass


def validate_single_gate_embedding():
    """Validate single-qutrit gate embeddings."""
    print("\n=== Single-Qutrit Gate Embedding ===")
    compiler = QutritToQubitCompiler()
    all_pass = True

    gates = [('X3', X3), ('Z3', Z3), ('H3', H3)]

    for name, gate in gates:
        embedded = compiler.embed_single_qutrit_gate(gate)

        # Check dimension
        if embedded.shape != (4, 4):
            print(f"  [FAIL] {name}: shape = {embedded.shape}, expected (4, 4)")
            all_pass = False
            continue
        print(f"  [PASS] {name}: shape = {embedded.shape}")

        # Check unitarity
        product = embedded.conj().T @ embedded
        err = np.max(np.abs(product - np.eye(4)))
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  [{status}] {name}: unitarity error = {err:.2e}")
        all_pass = all_pass and (err < 1e-10)

        # Check action preservation on qutrit subspace
        for trit_idx in range(3):
            trit_state = np.zeros(3, dtype=complex)
            trit_state[trit_idx] = 1.0
            qubit_state = compiler.encode_state(trit_state)

            # Apply gate in qutrit space
            trit_result = gate @ trit_state
            expected_qubit = compiler.encode_state(trit_result)

            # Apply embedded gate in qubit space
            actual_qubit = embedded @ qubit_state

            err = np.max(np.abs(actual_qubit - expected_qubit))
            status = "PASS" if err < 1e-10 else "FAIL"
            print(f"  [{status}] {name}|{trit_idx}⟩: encoding consistency error = {err:.2e}")
            all_pass = all_pass and (err < 1e-10)

        # Check forbidden state invariance
        forbidden = np.array([0, 0, 0, 1], dtype=complex)
        result = embedded @ forbidden
        err = np.max(np.abs(result - forbidden))
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  [{status}] {name}: forbidden state invariance error = {err:.2e}")
        all_pass = all_pass and (err < 1e-10)

    return all_pass


def validate_two_qutrit_embedding():
    """Validate two-qutrit gate embeddings."""
    print("\n=== Two-Qutrit Gate Embedding ===")
    compiler = QutritToQubitCompiler()
    all_pass = True

    gates = [('CSUM', CSUM), ('SWAP3', SWAP3)]

    for name, gate in gates:
        embedded = compiler.embed_two_qutrit_gate(gate)

        # Check dimension
        if embedded.shape != (16, 16):
            print(f"  [FAIL] {name}: shape = {embedded.shape}, expected (16, 16)")
            all_pass = False
            continue
        print(f"  [PASS] {name}: shape = {embedded.shape}")

        # Check unitarity
        product = embedded.conj().T @ embedded
        err = np.max(np.abs(product - np.eye(16)))
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  [{status}] {name}: unitarity error = {err:.2e}")
        all_pass = all_pass and (err < 1e-10)

        # Check action on all valid 2-qutrit basis states
        for a in range(3):
            for b in range(3):
                # Qutrit state
                trit_state = np.zeros(9, dtype=complex)
                trit_state[a * 3 + b] = 1.0
                trit_result = gate @ trit_state

                # Encode both states
                trit_a = np.zeros(3, dtype=complex)
                trit_a[a] = 1.0
                trit_b = np.zeros(3, dtype=complex)
                trit_b[b] = 1.0
                qubit_state = np.kron(
                    compiler.encode_state(trit_a),
                    compiler.encode_state(trit_b)
                )

                # Expected: encode the result
                # Find which basis states are non-zero
                expected_qubit = np.zeros(16, dtype=complex)
                for idx in range(9):
                    if abs(trit_result[idx]) > 1e-12:
                        a_out = idx // 3
                        b_out = idx % 3
                        ta = np.zeros(3, dtype=complex)
                        ta[a_out] = 1.0
                        tb = np.zeros(3, dtype=complex)
                        tb[b_out] = 1.0
                        expected_qubit += trit_result[idx] * np.kron(
                            compiler.encode_state(ta),
                            compiler.encode_state(tb)
                        )

                actual_qubit = embedded @ qubit_state
                err = np.max(np.abs(actual_qubit - expected_qubit))
                if err > 1e-8:
                    print(f"  [FAIL] {name}|{a}{b}⟩: consistency error = {err:.2e}")
                    all_pass = False

        print(f"  [{'PASS' if all_pass else 'FAIL'}] {name}: all basis state actions verified")

    return all_pass


def validate_parametric_embedding():
    """Validate parametric gate embeddings."""
    print("\n=== Parametric Gate Embedding ===")
    compiler = QutritToQubitCompiler()
    all_pass = True

    np.random.seed(42)
    angles = np.random.uniform(-np.pi, np.pi, 5)

    for theta in angles:
        for name, gate_fn in [('RX3', RX3), ('RY3', RY3), ('RZ3', RZ3)]:
            gate = gate_fn(theta)
            embedded = compiler.embed_single_qutrit_gate(gate)

            # Unitarity
            product = embedded.conj().T @ embedded
            err = np.max(np.abs(product - np.eye(4)))
            if err > 1e-10:
                print(f"  [FAIL] {name}({theta:.3f}): unitarity error = {err:.2e}")
                all_pass = False

            # Action consistency
            for trit_idx in range(3):
                trit_state = np.zeros(3, dtype=complex)
                trit_state[trit_idx] = 1.0
                qubit_state = compiler.encode_state(trit_state)

                trit_result = gate @ trit_state
                expected = compiler.encode_state(trit_result)
                actual = embedded @ qubit_state

                err = np.max(np.abs(actual - expected))
                if err > 1e-8:
                    print(f"  [FAIL] {name}({theta:.3f})|{trit_idx}⟩: "
                          f"error = {err:.2e}")
                    all_pass = False

    print(f"  [{'PASS' if all_pass else 'FAIL'}] All parametric embeddings verified")
    return all_pass


def validate_result_mapping():
    """Validate qubit→trit result mapping."""
    print("\n=== Result Mapping ===")
    all_pass = True

    # Single qutrit
    mapper = ResultMapper(n_qutrits=1)
    test_cases = [
        ({'00': 100}, {'0': 100}),
        ({'01': 200}, {'1': 200}),
        ({'10': 300}, {'2': 300}),
        ({'00': 50, '01': 30, '10': 20}, {'0': 50, '1': 30, '2': 20}),
    ]

    for qubit_counts, expected_trit in test_cases:
        trit_counts = mapper.map_counts(qubit_counts)
        match = all(trit_counts.get(k, 0) == v for k, v in expected_trit.items())
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {qubit_counts} → {trit_counts} (expected {expected_trit})")
        all_pass = all_pass and match

    # Two qutrits
    mapper2 = ResultMapper(n_qutrits=2)
    test_cases_2 = [
        ({'0000': 100}, {'00': 100}),
        ({'0001': 100}, {'01': 100}),
        ({'0100': 100}, {'10': 100}),
        ({'0110': 100}, {'12': 100}),
        ({'1001': 100}, {'21': 100}),
    ]

    for qubit_counts, expected_trit in test_cases_2:
        trit_counts = mapper2.map_counts(qubit_counts)
        match = all(trit_counts.get(k, 0) == v for k, v in expected_trit.items())
        status = "PASS" if match else "FAIL"
        print(f"  [{status}] {qubit_counts} → {trit_counts} (expected {expected_trit})")
        all_pass = all_pass and match

    return all_pass


def validate_leakage_detection():
    """Validate leakage detection and handling."""
    print("\n=== Leakage Detection ===")
    all_pass = True

    # Single qutrit: '11' is forbidden
    handler = LeakageHandler(n_qutrits=1, post_select=True)

    counts_with_leakage = {'00': 700, '01': 200, '10': 50, '11': 50}
    rate = handler.leakage_rate(counts_with_leakage)
    expected_rate = 50 / 1000
    err = abs(rate - expected_rate)
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  [{status}] Leakage rate: {rate:.4f} (expected {expected_rate:.4f})")
    all_pass = all_pass and (err < 1e-10)

    # Post-selection
    valid, leaked = handler.filter_counts(counts_with_leakage)
    has_forbidden = '11' in valid
    status = "PASS" if not has_forbidden else "FAIL"
    print(f"  [{status}] Post-selection removes forbidden states: {'11' not in valid}")
    all_pass = all_pass and (not has_forbidden)

    total_clean = sum(valid.values())
    expected_total = 950
    status = "PASS" if total_clean == expected_total else "FAIL"
    print(f"  [{status}] Clean total: {total_clean} (expected {expected_total})")
    all_pass = all_pass and (total_clean == expected_total)

    # Nearest mode
    handler_nearest = LeakageHandler(n_qutrits=1, post_select=False, leakage_fill='nearest')
    valid_nearest, leaked_nearest = handler_nearest.filter_counts(counts_with_leakage)
    total_nearest = sum(valid_nearest.values())
    status = "PASS" if total_nearest == 1000 else "FAIL"
    print(f"  [{status}] Nearest mode preserves total: {total_nearest}")
    all_pass = all_pass and (total_nearest == 1000)
    has_forbidden = '11' in valid_nearest
    status = "PASS" if not has_forbidden else "FAIL"
    print(f"  [{status}] Nearest mode removes forbidden: {'11' not in valid_nearest}")
    all_pass = all_pass and (not has_forbidden)

    return all_pass


def main():
    """Run all encoding validations."""
    print("=" * 60)
    print("  TriFlowQC Encoding Validation Suite")
    print("=" * 60)

    results = {}
    results['Encoding Map'] = validate_encoding_map()
    results['Single Gate Embedding'] = validate_single_gate_embedding()
    results['Two-Qutrit Embedding'] = validate_two_qutrit_embedding()
    results['Parametric Embedding'] = validate_parametric_embedding()
    results['Result Mapping'] = validate_result_mapping()
    results['Leakage Detection'] = validate_leakage_detection()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and passed

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
