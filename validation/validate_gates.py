"""
Validation script: Verify all gate matrices satisfy mathematical properties.

Checks:
  - Unitarity: U†U = I
  - Correct dimensions
  - Special properties (X3³=I, Z3³=I, H3 creates uniform superposition)
  - Parametric gates at known angles
  - Two-qutrit gate properties
"""

import numpy as np
import sys


def check_unitary(name, U, tol=1e-10):
    """Check that U†U = I."""
    d = U.shape[0]
    product = U.conj().T @ U
    identity = np.eye(d)
    error = np.max(np.abs(product - identity))
    status = "PASS" if error < tol else "FAIL"
    print(f"  [{status}] {name}: unitarity error = {error:.2e}")
    return error < tol


def check_dimension(name, U, expected_dim):
    """Check matrix dimensions."""
    actual = U.shape
    status = "PASS" if actual == (expected_dim, expected_dim) else "FAIL"
    print(f"  [{status}] {name}: shape = {actual}, expected ({expected_dim}, {expected_dim})")
    return actual == (expected_dim, expected_dim)


def validate_gell_mann_matrices():
    """Validate all 8 Gell-Mann matrices."""
    print("\n=== Gell-Mann Matrices ===")
    from triflowqc.gates.gell_mann import gell_mann_matrices, structure_constants

    lambdas = gell_mann_matrices()
    all_pass = True

    # Check count
    assert len(lambdas) == 8, f"Expected 8, got {len(lambdas)}"
    print(f"  [PASS] Count: {len(lambdas)} matrices")

    # Check each is Hermitian and traceless
    for i, lam in enumerate(lambdas):
        name = f"λ_{i+1}"
        if not check_dimension(name, lam, 3):
            all_pass = False
        # Hermitian
        herm_err = np.max(np.abs(lam - lam.conj().T))
        if herm_err > 1e-12:
            print(f"  [FAIL] {name}: not Hermitian, error = {herm_err:.2e}")
            all_pass = False
        else:
            print(f"  [PASS] {name}: Hermitian (error = {herm_err:.2e})")
        # Traceless
        tr = np.abs(np.trace(lam))
        if tr > 1e-12:
            print(f"  [FAIL] {name}: not traceless, Tr = {tr:.2e}")
            all_pass = False
        else:
            print(f"  [PASS] {name}: traceless (Tr = {tr:.2e})")

    # Check trace orthonormality: Tr(λ_i λ_j) = 2δ_ij
    print("\n  Trace orthonormality:")
    for i in range(8):
        for j in range(8):
            tr = np.trace(lambdas[i] @ lambdas[j]).real
            expected = 2.0 if i == j else 0.0
            if abs(tr - expected) > 1e-10:
                print(f"  [FAIL] Tr(λ_{i+1} λ_{j+1}) = {tr:.6f}, expected {expected}")
                all_pass = False
    print(f"  [{'PASS' if all_pass else 'FAIL'}] Trace orthonormality")

    # Check structure constants
    print("\n  Structure constants:")
    f = structure_constants()
    # f_123 = 1
    f_123 = f[0, 1, 2]
    status = "PASS" if abs(f_123 - 1.0) < 1e-10 else "FAIL"
    print(f"  [{status}] f_123 = {f_123:.6f} (expected 1.0)")
    all_pass = all_pass and (abs(f_123 - 1.0) < 1e-10)

    # Antisymmetry
    antisym = True
    for i in range(8):
        for j in range(8):
            for k in range(8):
                if abs(f[i, j, k] + f[j, i, k]) > 1e-10:
                    antisym = False
    print(f"  [{'PASS' if antisym else 'FAIL'}] Antisymmetry f_ijk = -f_jik")
    all_pass = all_pass and antisym

    return all_pass


def validate_single_qutrit_gates():
    """Validate single-qutrit gates."""
    print("\n=== Single-Qutrit Gates ===")
    from triflowqc.gates.single_qutrit import X3, Z3, H3, T3, S3

    all_pass = True

    for name, gate in [('X3', X3), ('Z3', Z3), ('H3', H3), ('T3', T3), ('S3', S3)]:
        if not check_dimension(name, gate, 3):
            all_pass = False
        if not check_unitary(name, gate):
            all_pass = False

    # X3³ = I
    x3_cubed = X3 @ X3 @ X3
    err = np.max(np.abs(x3_cubed - np.eye(3)))
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  [{status}] X3³ = I (error = {err:.2e})")
    all_pass = all_pass and (err < 1e-10)

    # Z3³ = I
    z3_cubed = Z3 @ Z3 @ Z3
    err = np.max(np.abs(z3_cubed - np.eye(3)))
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  [{status}] Z3³ = I (error = {err:.2e})")
    all_pass = all_pass and (err < 1e-10)

    # H3 on |0⟩ gives uniform superposition
    state_0 = np.array([1, 0, 0], dtype=complex)
    h_state = H3 @ state_0
    probs = np.abs(h_state) ** 2
    uniform = np.allclose(probs, 1.0 / 3.0, atol=1e-10)
    print(f"  [{'PASS' if uniform else 'FAIL'}] H3|0⟩ uniform superposition")
    all_pass = all_pass and uniform

    return all_pass


def validate_parametric_gates():
    """Validate parametric rotation gates."""
    print("\n=== Parametric Gates ===")
    from triflowqc.gates.parametric import RX3, RY3, RZ3, R_GM

    all_pass = True

    # Identity at theta=0
    for name, gate_fn in [('RX3', RX3), ('RY3', RY3), ('RZ3', RZ3)]:
        U0 = gate_fn(0.0)
        err = np.max(np.abs(U0 - np.eye(3)))
        status = "PASS" if err < 1e-10 else "FAIL"
        print(f"  [{status}] {name}(0) = I (error = {err:.2e})")
        all_pass = all_pass and (err < 1e-10)

    # Unitarity at random angles
    np.random.seed(42)
    for _ in range(10):
        theta = np.random.uniform(-2 * np.pi, 2 * np.pi)
        for name, gate_fn in [('RX3', RX3), ('RY3', RY3), ('RZ3', RZ3)]:
            U = gate_fn(theta)
            if not check_unitary(f"{name}({theta:.3f})", U):
                all_pass = False

    # R_GM should be unitary for all generators
    print("\n  R_GM unitarity for each generator:")
    for k in range(8):
        U = R_GM(k, np.pi / 3)
        if not check_unitary(f"R_GM(k={k}, π/3)", U):
            all_pass = False

    return all_pass


def validate_two_qutrit_gates():
    """Validate two-qutrit gates."""
    print("\n=== Two-Qutrit Gates ===")
    from triflowqc.gates.two_qutrit import CSUM, CPHASE, SWAP3, CX3

    all_pass = True

    for name, gate in [('CSUM', CSUM), ('CPHASE', CPHASE), ('SWAP3', SWAP3), ('CX3', CX3)]:
        if not check_dimension(name, gate, 9):
            all_pass = False
        if not check_unitary(name, gate):
            all_pass = False

    # CSUM: |a⟩|b⟩ → |a⟩|(a+b) mod 3⟩
    print("\n  CSUM action verification:")
    for a in range(3):
        for b in range(3):
            state = np.zeros(9, dtype=complex)
            state[a * 3 + b] = 1.0
            result = CSUM @ state
            expected_b = (a + b) % 3
            expected_idx = a * 3 + expected_b
            correct = abs(result[expected_idx]) ** 2 > 0.999
            if not correct:
                print(f"  [FAIL] CSUM|{a}⟩|{b}⟩: expected |{a}⟩|{expected_b}⟩")
                all_pass = False
    print(f"  [{'PASS' if all_pass else 'FAIL'}] CSUM action on all basis states")

    # SWAP: |a⟩|b⟩ → |b⟩|a⟩
    swap_ok = True
    for a in range(3):
        for b in range(3):
            state = np.zeros(9, dtype=complex)
            state[a * 3 + b] = 1.0
            result = SWAP3 @ state
            expected_idx = b * 3 + a
            if abs(result[expected_idx]) ** 2 < 0.999:
                swap_ok = False
    print(f"  [{'PASS' if swap_ok else 'FAIL'}] SWAP3 action on all basis states")
    all_pass = all_pass and swap_ok

    return all_pass


def validate_three_qutrit_gates():
    """Validate three-qutrit gates."""
    print("\n=== Three-Qutrit Gates ===")
    from triflowqc.gates.three_qutrit import Toffoli3

    all_pass = True

    if not check_dimension('Toffoli3', Toffoli3, 27):
        all_pass = False
    if not check_unitary('Toffoli3', Toffoli3):
        all_pass = False

    return all_pass


def main():
    """Run all gate validations."""
    print("=" * 60)
    print("  TriFlowQC Gate Validation Suite")
    print("=" * 60)

    results = {}
    results['Gell-Mann'] = validate_gell_mann_matrices()
    results['Single-Qutrit'] = validate_single_qutrit_gates()
    results['Parametric'] = validate_parametric_gates()
    results['Two-Qutrit'] = validate_two_qutrit_gates()
    results['Three-Qutrit'] = validate_three_qutrit_gates()

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
