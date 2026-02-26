"""
Validate the qutrit-to-qubit encoding preserves quantum information.
"""

import numpy as np
import pytest
from triflowqc.compiler.qutrit_to_qubit import (
    embed_single_qutrit_gate,
    embed_two_qutrit_gate,
    ENCODING,
)
from triflowqc.gates.single_qutrit import X3, H3, Z3
from triflowqc.gates.two_qutrit import CSUM, SWAP3, CPHASE
from triflowqc.gates.parametric import R_GM


class TestEncodingConsistency:
    """Verify that qutrit-to-qubit encoding preserves gate actions."""

    def _qutrit_to_qubit_state(self, trit_state):
        """Convert a qutrit state vector to qubit encoding."""
        d = len(trit_state)
        n_qutrits = int(np.round(np.log(d) / np.log(3)))
        n_qubits = 2 * n_qutrits
        qubit_dim = 2 ** n_qubits
        qubit_state = np.zeros(qubit_dim, dtype=complex)

        for idx in range(d):
            amp = trit_state[idx]
            if np.abs(amp) < 1e-15:
                continue
            # Convert trit index to qubit index
            trits = []
            val = idx
            for _ in range(n_qutrits):
                trits.append(val % 3)
                val //= 3
            trits = list(reversed(trits))

            qubit_idx = 0
            for q, t in enumerate(trits):
                qa, qb = ENCODING[t]
                qubit_idx |= (qa << (2 * n_qutrits - 1 - 2 * q))
                qubit_idx |= (qb << (2 * n_qutrits - 2 - 2 * q))
            qubit_state[qubit_idx] = amp

        return qubit_state

    def test_single_gate_encoding_consistency(self):
        """
        For each basis state |k⟩, verify:
        encode(U|k⟩) = V·encode(|k⟩)
        where V = embed(U).
        """
        gates = [X3, H3, Z3]
        for U in gates:
            V = embed_single_qutrit_gate(U)
            for k in range(3):
                # Qutrit: apply U to |k⟩
                trit_in = np.zeros(3, dtype=complex)
                trit_in[k] = 1.0
                trit_out = U @ trit_in

                # Encode both
                qubit_in = self._qutrit_to_qubit_state(trit_in)
                qubit_out_expected = self._qutrit_to_qubit_state(trit_out)

                # Apply embedded gate
                qubit_out_actual = V @ qubit_in

                assert np.allclose(qubit_out_actual, qubit_out_expected, atol=1e-10), \
                    f"Encoding mismatch for gate {U} on |{k}⟩"

    def test_two_qutrit_encoding_consistency(self):
        """Verify encoding consistency for two-qutrit gates."""
        for U in [CSUM, SWAP3, CPHASE]:
            V = embed_two_qutrit_gate(U)
            for a in range(3):
                for b in range(3):
                    trit_in = np.zeros(9, dtype=complex)
                    trit_in[a * 3 + b] = 1.0
                    trit_out = U @ trit_in

                    qubit_in = self._qutrit_to_qubit_state(trit_in)
                    qubit_out_expected = self._qutrit_to_qubit_state(trit_out)
                    qubit_out_actual = V @ qubit_in

                    assert np.allclose(
                        qubit_out_actual, qubit_out_expected, atol=1e-10
                    ), f"2-qutrit encoding mismatch for |{a},{b}⟩"

    def test_parametric_gate_encoding(self):
        """Verify encoding works for parametric gates."""
        rng = np.random.default_rng(42)
        for k in range(1, 9):
            theta = rng.uniform(-np.pi, np.pi)
            U = R_GM(k, theta)
            V = embed_single_qutrit_gate(U)

            for basis in range(3):
                trit_in = np.zeros(3, dtype=complex)
                trit_in[basis] = 1.0
                trit_out = U @ trit_in

                qubit_in = self._qutrit_to_qubit_state(trit_in)
                qubit_expected = self._qutrit_to_qubit_state(trit_out)
                qubit_actual = V @ qubit_in

                assert np.allclose(qubit_actual, qubit_expected, atol=1e-10)

    def test_forbidden_state_isolation(self):
        """
        The |11⟩ state should be mapped to itself (identity) and
        should not leak into the qutrit subspace.
        """
        V = embed_single_qutrit_gate(X3)
        # |11⟩ = index 3 in 4-dim space
        forbidden = np.array([0, 0, 0, 1], dtype=complex)
        result = V @ forbidden
        assert np.allclose(result, forbidden, atol=1e-12), \
            "Forbidden state |11⟩ should be unchanged"

    def test_superposition_encoding(self):
        """Encoding of a superposition state."""
        # |ψ⟩ = (|0⟩ + |1⟩ + |2⟩)/√3
        trit_state = np.ones(3, dtype=complex) / np.sqrt(3)
        qubit_state = self._qutrit_to_qubit_state(trit_state)

        # Should have equal amplitudes at |00⟩, |01⟩, |10⟩ and zero at |11⟩
        expected = np.array([1, 1, 1, 0], dtype=complex) / np.sqrt(3)
        assert np.allclose(qubit_state, expected, atol=1e-12)
