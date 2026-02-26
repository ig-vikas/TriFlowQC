# Gell-Mann Algebra Test Suite — Full Documentation

## Overview

The file **`tests/math_validation/test_gell_mann_algebra.py`** provides a rigorous mathematical validation suite for the Gell-Mann matrix implementation and the qutrit gate algebra in TriFlowQC. It ensures that the framework's core mathematical building blocks — Gell-Mann matrices, SU(3) structure constants, parametric rotation gates, and single/two-qutrit gates — satisfy the exact analytical identities mandated by quantum mechanics and group theory.

This test file is the mathematical backbone of TriFlowQC's correctness guarantees. Every gate, rotation, and algebraic relation used during circuit simulation and quantum neural network training is anchored here.

---

## Table of Contents

1. [Background: What Are Gell-Mann Matrices?](#1-background-what-are-gell-mann-matrices)
2. [Source Modules Under Test](#2-source-modules-under-test)
3. [Test Classes and Methods](#3-test-classes-and-methods)
   - 3.1 [TestGellMannAlgebra](#31-testgellmannalgebra)
   - 3.2 [TestParametricGateAlgebra](#32-testparametricgatealgebra)
   - 3.3 [TestQutritGateProperties](#33-testqutritgateproperties)
4. [How to Run the Tests](#4-how-to-run-the-tests)
5. [Implementation Details](#5-implementation-details)
6. [Mathematical Reference](#6-mathematical-reference)
7. [Extending the Tests](#7-extending-the-tests)

---

## 1. Background: What Are Gell-Mann Matrices?

Gell-Mann matrices are eight 3×3 Hermitian, traceless matrices (denoted λ₁ through λ₈) that form the generators of the **SU(3) Lie algebra**. They are the qutrit analogue of the Pauli matrices (σₓ, σᵧ, σᵤ) that serve qubits.

### Key Properties

| Property | Formula | Meaning |
|---|---|---|
| **Hermiticity** | λ† = λ | Each matrix equals its own conjugate transpose |
| **Traceless** | Tr(λ) = 0 | Sum of diagonal elements is zero |
| **Orthogonality** | Tr(λᵢ λⱼ) = 2 δᵢⱼ | Matrices are orthogonal under the trace inner product |
| **Commutation** | [λᵢ, λⱼ] = 2i Σₖ fᵢⱼₖ λₖ | Closed under commutation with structure constants fᵢⱼₖ |
| **Casimir** | C₂ = Σₖ λₖ² = (16/3) I | Quadratic Casimir is proportional to the identity |

### The Eight Matrices

| Matrix | Subspace | Type | Explicit Form |
|---|---|---|---|
| λ₁ | 0 ↔ 1 | Real off-diagonal | `[[0,1,0],[1,0,0],[0,0,0]]` |
| λ₂ | 0 ↔ 1 | Imaginary off-diagonal | `[[0,-i,0],[i,0,0],[0,0,0]]` |
| λ₃ | 0 – 1 | Diagonal | `[[1,0,0],[0,-1,0],[0,0,0]]` |
| λ₄ | 0 ↔ 2 | Real off-diagonal | `[[0,0,1],[0,0,0],[1,0,0]]` |
| λ₅ | 0 ↔ 2 | Imaginary off-diagonal | `[[0,0,-i],[0,0,0],[i,0,0]]` |
| λ₆ | 1 ↔ 2 | Real off-diagonal | `[[0,0,0],[0,0,1],[0,1,0]]` |
| λ₇ | 1 ↔ 2 | Imaginary off-diagonal | `[[0,0,0],[0,0,-i],[0,i,0]]` |
| λ₈ | Diagonal | Diagonal | `(1/√3) [[1,0,0],[0,1,0],[0,0,-2]]` |

### Structure Constants

The structure constants fᵢⱼₖ encode the Lie bracket (commutator) relations. They are **totally antisymmetric** (change sign under any index swap). Key non-zero values:

- f₁₂₃ = 1
- f₁₄₇ = f₂₄₆ = f₂₅₇ = f₃₄₅ = ½
- f₄₅₈ = f₆₇₈ = √3/2

---

## 2. Source Modules Under Test

The test file validates the following four source modules:

### 2.1 `triflowqc.gates.gell_mann`

**Purpose:** Defines the 8 Gell-Mann matrices as NumPy arrays along with helper functions.

| Symbol / Function | Description |
|---|---|
| `LAMBDA_1` … `LAMBDA_8` | Pre-built 3×3 `complex128` NumPy arrays for each Gell-Mann matrix |
| `gell_mann(k)` | Returns a copy of the k-th Gell-Mann matrix (1 ≤ k ≤ 8) |
| `gell_mann_matrices()` | Returns all eight as a list |
| `is_gell_mann_valid(matrix)` | Checks if a given matrix is Hermitian, traceless, and has Tr(M²) = 2 |
| `commutator(A, B)` | Computes [A, B] = AB − BA |
| `anticommutator(A, B)` | Computes {A, B} = AB + BA |
| `structure_constants()` | Computes the full 8×8×8 tensor fᵢⱼₖ from the commutator definition |

### 2.2 `triflowqc.math.su3`

**Purpose:** SU(3) Lie group utilities — generators, Casimir operator, random elements, basis decomposition.

| Function | Description |
|---|---|
| `su3_generators()` | Returns the 8 Gell-Mann matrices (delegates to `gell_mann` module) |
| `su3_random_element(rng)` | Generates a Haar-random SU(3) matrix via QR decomposition |
| `is_su3(U)` | Validates U†U = I and det(U) = 1 |
| `casimir_operator()` | Computes C₂ = Σₖ λₖ² = (16/3) I |
| `su3_basis_decompose(M)` | Decomposes a traceless Hermitian matrix in the Gell-Mann basis |
| `su3_reconstruct(coeffs)` | Reconstructs M = Σₖ cₖ λₖ from coefficients |
| `su3_exponential(coeffs)` | Computes exp(−i Σₖ cₖ λₖ / 2) |
| `su3_adjoint_representation(U)` | Computes the 8×8 adjoint representation of an SU(3) element |

### 2.3 `triflowqc.gates.parametric`

**Purpose:** Parametric rotation gates built from Gell-Mann generators.

| Function | Description |
|---|---|
| `R_GM(k, theta)` | General rotation: exp(−iθλₖ/2) for generator k ∈ {1…8} |
| `RX3(theta)` | Alias for `R_GM(1, theta)` — rotation in the \|0⟩–\|1⟩ subspace |
| `RY3(theta)` | Alias for `R_GM(2, theta)` |
| `RZ3(theta)` | Alias for `R_GM(3, theta)` |
| `general_su3_rotation(thetas)` | Product R₁(θ₁)·R₂(θ₂)·…·R₈(θ₈) — a general parameterised SU(3) element |

### 2.4 `triflowqc.gates.single_qutrit` and `triflowqc.gates.two_qutrit`

**Purpose:** Fixed (non-parametric) single- and two-qutrit gate matrices.

| Gate | Size | Definition |
|---|---|---|
| `X3` | 3×3 | Cyclic shift: \|0⟩→\|1⟩→\|2⟩→\|0⟩. X3³ = I |
| `Z3` | 3×3 | Phase gate: diag(1, ω, ω²) where ω = e^{2πi/3}. Z3³ = I |
| `H3` | 3×3 | Qutrit Hadamard (3-point QFT / √3) |
| `CSUM` | 9×9 | Controlled-SUM: \|a,b⟩ → \|a, (a+b) mod 3⟩ (qutrit CNOT) |
| `CPHASE` | 9×9 | Controlled-phase: \|a,b⟩ → ω^{ab} \|a,b⟩ |

---

## 3. Test Classes and Methods

### 3.1 `TestGellMannAlgebra`

Validates that the Gell-Mann matrices satisfy the SU(3) Lie algebra relations. These are **exact analytical identities** — any failure indicates a bug in the matrix definitions or structure constant computation.

#### `test_tr_lambda_i_lambda_j`

**What it checks:** Tr(λᵢ λⱼ) = 2 δᵢⱼ (orthogonality under the trace inner product).

**How it works:**
1. Retrieves all 8 generators via `su3_generators()`.
2. Iterates over all 64 (i, j) pairs.
3. Computes `np.trace(gens[i] @ gens[j])`.
4. Asserts the result is 2.0 when i == j and 0.0 otherwise (tolerance: 1e-12).

**Why it matters:** This identity is the foundation for decomposing arbitrary operators in the Gell-Mann basis. It ensures the generators form a properly normalised basis for the su(3) Lie algebra.

---

#### `test_commutation_relations`

**What it checks:** [λᵢ, λⱼ] = 2i Σₖ fᵢⱼₖ λₖ (the Lie bracket closes properly).

**How it works:**
1. Computes the commutator `gens[i] @ gens[j] - gens[j] @ gens[i]` for all 64 pairs.
2. Reconstructs the expected right-hand side using `structure_constants()`.
3. Asserts element-wise agreement (tolerance: 1e-10).

**Why it matters:** This verifies the entire Lie algebra structure. If the commutation relations are wrong, parametric gates built from these generators (used in quantum neural networks) would compute incorrect gradients.

---

#### `test_known_structure_constants`

**What it checks:** Specific analytically known values of fᵢⱼₖ.

**Verified values:**
| Constant | Expected Value |
|---|---|
| f₁₂₃ | 1 |
| f₄₅₈ | √3/2 |
| f₆₇₈ | √3/2 |
| f₁₄₇ | 1/2 |

**Why it matters:** Cross-checks the computed structure constants against textbook values. Catches both sign errors and normalisation mistakes.

---

#### `test_structure_constants_totally_antisymmetric`

**What it checks:** fᵢⱼₖ = −fⱼᵢₖ = −fᵢₖⱼ (total antisymmetry).

**How it works:** Iterates over all 8³ = 512 index combinations and verifies both swap rules.

**Why it matters:** Total antisymmetry is a defining property of SU(3) structure constants. Any violation would indicate an error in the commutator-based computation.

---

#### `test_casimir_operator_value`

**What it checks:** C₂ = Σₖ λₖ² = (16/3) I₃.

**How it works:** Calls `casimir_operator()` and compares with `(16/3) * np.eye(3)`.

**Why it matters:** The Casimir operator commutes with all group elements and has a fixed value in each irreducible representation. For the fundamental (3-dimensional) representation of SU(3), that value is 16/3. This serves as an integrated check over all 8 generators simultaneously.

---

### 3.2 `TestParametricGateAlgebra`

Validates mathematical properties of the parametric rotation gates R_k(θ) = exp(−iθλₖ/2).

#### `test_rotation_at_zero_is_identity`

**What it checks:** R_k(0) = I for all k ∈ {1, …, 8}.

**Why it matters:** The matrix exponential of the zero matrix must be the identity. This is the simplest sanity check for `R_GM()`.

---

#### `test_rotation_determinant`

**What it checks:** |det(R_k(θ))| = 1 for random angles θ.

**How it works:**
1. For each of the 8 generators, draws 10 random angles in [−2π, 2π] (seeded RNG for reproducibility).
2. Computes `np.abs(np.linalg.det(R_GM(k, theta)))` and asserts ≈ 1.0.

**Why it matters:** SU(3) elements must have unit determinant. A deviation would mean `R_GM` is producing matrices outside SU(3).

---

#### `test_rotation_unitarity`

**What it checks:** R_k(θ)† R_k(θ) = I for random angles.

**How it works:** Same random sampling strategy; checks `U @ U.conj().T ≈ np.eye(3)`.

**Why it matters:** Unitarity preserves quantum state norms. Non-unitary gates would cause probability leakage and physically meaningless results.

---

#### `test_rotation_composition`

**What it checks:** R_k(α) · R_k(β) = R_k(α + β) for diagonal generators k = 3 and k = 8.

**How it works:** Uses α = 1.2, β = 0.7, computes both sides, and asserts equality.

**Why it matters:** For diagonal (mutually commuting) generators, the rotation group is Abelian and parameter addition holds exactly. This confirms the exponential map behaves correctly for the Cartan subalgebra.

> **Note:** This property does *not* hold for non-diagonal generators (k = 1, 2, 4, 5, 6, 7) because their exponentials do not commute in general.

---

#### `test_rotation_period`

**What it checks:** R₃(4π) = I.

**How it works:** λ₃ has eigenvalues {+1, −1, 0}, so exp(−iθλ₃/2) has eigenvalues {e^{−iθ/2}, e^{iθ/2}, 1}. Setting θ = 4π gives {e^{−2πi}, e^{2πi}, 1} = {1, 1, 1}.

**Why it matters:** Validates the periodicity of the rotation, which is critical for the parameter-shift rule used in gradient computation.

---

### 3.3 `TestQutritGateProperties`

Validates algebraic identities of fixed (non-parametric) qutrit gates.

#### `test_x3_cubed_is_identity`

**What it checks:** X₃³ = I (the cyclic shift gate has order 3).

**Source:** `triflowqc.gates.single_qutrit.X3`

**Why it matters:** X₃ cycles |0⟩→|1⟩→|2⟩→|0⟩. After 3 applications the original state must be restored. This is the qutrit analogue of σₓ² = I for qubits.

---

#### `test_z3_cubed_is_identity`

**What it checks:** Z₃³ = I.

**Source:** `triflowqc.gates.single_qutrit.Z3` — `diag(1, ω, ω²)` where ω = e^{2πi/3}.

**Why it matters:** ω³ = 1 (cube root of unity), so Z₃³ = diag(1, ω³, ω⁶) = diag(1, 1, 1) = I.

---

#### `test_h3_creates_equal_superposition`

**What it checks:** H₃|0⟩ produces equal probabilities (1/3, 1/3, 1/3).

**Source:** `triflowqc.gates.single_qutrit.H3` — the 3×3 QFT matrix normalised by 1/√3.

**Why it matters:** The Hadamard (or QFT) gate is used to create uniform superposition, a prerequisite for many quantum algorithms. If the output probabilities are not uniform, circuits using H₃ would produce incorrect results.

---

#### `test_csum_associativity`

**What it checks:** CSUM|a, b⟩ = |a, (a+b) mod 3⟩ for all 9 basis states.

**Source:** `triflowqc.gates.two_qutrit.CSUM` — a 9×9 permutation matrix.

**How it works:**
1. Loops over all (a, b) with a, b ∈ {0, 1, 2}.
2. Builds the input basis state as a 9-element vector.
3. Applies CSUM and verifies the output matches |a, (a+b) mod 3⟩.

**Why it matters:** CSUM is the qutrit analogue of CNOT. It forms the entangling layer in many qutrit circuit architectures and must implement modular addition exactly.

---

#### `test_cphase_diagonal`

**What it checks:** CPHASE is diagonal with entries ω^{ab} and all off-diagonal entries are zero.

**Source:** `triflowqc.gates.two_qutrit.CPHASE`

**How it works:**
1. For each basis state index (a·3 + b), verifies the diagonal element equals ω^{ab}.
2. For every off-diagonal entry in that row, verifies it is zero.

**Why it matters:** CPHASE applies a relative phase that depends on both qutrits. It is the qutrit analogue of CZ and is used in creating entanglement. Being diagonal, it should not map between basis states — only apply phases.

---

## 4. How to Run the Tests

### Prerequisites

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # includes pytest
```

### Run the Full Gell-Mann Algebra Suite

```bash
# From the project root
pytest tests/math_validation/test_gell_mann_algebra.py -v
```

Expected output:

```
tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_tr_lambda_i_lambda_j PASSED
tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_commutation_relations PASSED
tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_known_structure_constants PASSED
tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_structure_constants_totally_antisymmetric PASSED
tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_casimir_operator_value PASSED
tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra::test_rotation_at_zero_is_identity PASSED
tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra::test_rotation_determinant PASSED
tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra::test_rotation_unitarity PASSED
tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra::test_rotation_composition PASSED
tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra::test_rotation_period PASSED
tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties::test_x3_cubed_is_identity PASSED
tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties::test_z3_cubed_is_identity PASSED
tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties::test_h3_creates_equal_superposition PASSED
tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties::test_csum_associativity PASSED
tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties::test_cphase_diagonal PASSED
```

### Run a Specific Test Class

```bash
pytest tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra -v
pytest tests/math_validation/test_gell_mann_algebra.py::TestParametricGateAlgebra -v
pytest tests/math_validation/test_gell_mann_algebra.py::TestQutritGateProperties -v
```

### Run a Single Test

```bash
pytest tests/math_validation/test_gell_mann_algebra.py::TestGellMannAlgebra::test_commutation_relations -v
```

### Run All Math Validation Tests

```bash
pytest tests/math_validation/ -v
```

---

## 5. Implementation Details

### Numerical Tolerances

The tests use two tolerance thresholds:

| Tolerance | Used For |
|---|---|
| `atol=1e-12` | Exact identities involving only integer/rational matrix entries (trace orthogonality, gate powers) |
| `atol=1e-10` | Identities involving floating-point structure constants or matrix exponentials |

The looser tolerance for structure constants accounts for accumulated rounding in the triple loop computation of fᵢⱼₖ.

### Reproducible Random Testing

Tests that involve random parameters (`test_rotation_determinant`, `test_rotation_unitarity`) use:

```python
rng = np.random.default_rng(42)
```

This seeded generator ensures **deterministic and reproducible** test results across runs and platforms.

### Import Strategy

The test file uses **two import patterns**:

1. **Module-level imports** for `gell_mann`, `parametric`, and `su3` — these are used by multiple tests.
2. **Local imports** inside test methods for `single_qutrit` and `two_qutrit` — this keeps the import scope tight and makes it clear which gate each test validates.

### Dependency Graph

```
test_gell_mann_algebra.py
├── triflowqc.gates.gell_mann      ← Gell-Mann matrices + structure constants
│   ├── LAMBDA_1 … LAMBDA_8
│   ├── gell_mann(k)
│   ├── structure_constants()
│   └── commutator()
├── triflowqc.gates.parametric     ← Parametric rotation gates
│   └── R_GM(k, theta)
├── triflowqc.math.su3             ← SU(3) utilities
│   ├── su3_generators()
│   ├── is_su3()
│   └── casimir_operator()
├── triflowqc.gates.single_qutrit  ← Fixed single-qutrit gates
│   ├── X3, Z3, H3
│   └── (imported locally in tests)
└── triflowqc.gates.two_qutrit     ← Fixed two-qutrit gates
    ├── CSUM, CPHASE
    └── (imported locally in tests)
```

---

## 6. Mathematical Reference

### Commutation Relation Derivation

The structure constants are extracted from the commutator via the trace trick:

```
[λᵢ, λⱼ] = 2i Σₖ fᵢⱼₖ λₖ
```

Multiply both sides by λₘ and take the trace:

```
Tr([λᵢ, λⱼ] λₘ) = 2i Σₖ fᵢⱼₖ Tr(λₖ λₘ) = 2i Σₖ fᵢⱼₖ · 2δₖₘ = 4i fᵢⱼₘ
```

Therefore:

```
fᵢⱼₖ = Tr([λᵢ, λⱼ] λₖ) / (4i)
```

This is exactly the formula used in `structure_constants()`.

### Parametric Gate Exponential

```
R_k(θ) = exp(−iθλₖ/2)
```

This is computed via `scipy.linalg.expm()`. For diagonal generators (λ₃, λ₈), the exponential reduces to a simple diagonal matrix:

```
R₃(θ) = diag(e^{−iθ/2}, e^{iθ/2}, 1)
```

For off-diagonal generators, the exponential involves trigonometric functions in the relevant 2×2 subspace:

```
R₁(θ) acts on the |0⟩–|1⟩ block as:
    [[cos(θ/2),  −i sin(θ/2)],
     [−i sin(θ/2),  cos(θ/2)]]
with |2⟩ unchanged.
```

### Casimir Operator

The quadratic Casimir of SU(3) in the fundamental representation:

```
C₂ = Σₖ₌₁⁸ λₖ² = (16/3) I₃
```

By Schur's lemma, C₂ commutes with all SU(3) elements and is therefore a scalar multiple of the identity in any irreducible representation. The value 16/3 is specific to the fundamental (dimension-3) representation.

### Cyclic Group Orders

| Gate | Order | Explanation |
|---|---|---|
| X₃ | 3 | Permutation (0 1 2) is a 3-cycle |
| Z₃ | 3 | ω = e^{2πi/3}, so ω³ = 1 |
| R₃(θ) | θ = 4π | Eigenvalues e^{±iθ/2} return to 1 when θ = 4π |

---

## 7. Extending the Tests

### Adding a New Gell-Mann Identity

To verify a new algebraic identity:

```python
class TestGellMannAlgebra:
    # ... existing tests ...

    def test_anticommutation_d_constants(self):
        """Verify {λ_i, λ_j} = (4/3)δ_{ij}I + 2 Σ_k d_{ijk} λ_k."""
        from triflowqc.gates.gell_mann import anticommutator
        gens = su3_generators()
        for i in range(8):
            for j in range(8):
                acomm = anticommutator(gens[i], gens[j])
                # Extract d constants and verify
                trace_part = (4.0 / 3.0) * (1 if i == j else 0) * np.eye(3)
                remainder = acomm - trace_part
                # remainder should decompose in Gell-Mann basis → d_{ijk}
                for k in range(8):
                    d_ijk = np.real(np.trace(remainder @ gens[k])) / 4.0
                    # d_{ijk} is totally symmetric — add assertion here
```

### Adding a New Gate Test

```python
class TestQutritGateProperties:
    # ... existing tests ...

    def test_swap3_is_involution(self):
        """SWAP² = I."""
        from triflowqc.gates.two_qutrit import SWAP3
        assert np.allclose(SWAP3 @ SWAP3, np.eye(9), atol=1e-12)
```

### Adding a New Parametric Gate Property

```python
class TestParametricGateAlgebra:
    # ... existing tests ...

    def test_inverse_property(self):
        """R_k(θ) R_k(-θ) = I."""
        rng = np.random.default_rng(99)
        for k in range(1, 9):
            theta = rng.uniform(-2 * np.pi, 2 * np.pi)
            U = R_GM(k, theta)
            U_inv = R_GM(k, -theta)
            assert np.allclose(U @ U_inv, np.eye(3), atol=1e-10)
```

---

## Summary

| Test Class | # Tests | What Is Validated |
|---|---|---|
| `TestGellMannAlgebra` | 5 | Gell-Mann matrix properties, commutation relations, structure constants, Casimir operator |
| `TestParametricGateAlgebra` | 5 | Parametric rotation identity, unitarity, determinant, composition, periodicity |
| `TestQutritGateProperties` | 5 | X₃/Z₃ order, H₃ superposition, CSUM modular addition, CPHASE diagonal phases |
| **Total** | **15** | **Complete algebraic validation of the qutrit gate framework** |

These 15 tests collectively guarantee that TriFlowQC's mathematical core — from raw matrices to compiled gates — is analytically correct. They should be run before every release and after any modification to the `gates/` or `math/` modules.

