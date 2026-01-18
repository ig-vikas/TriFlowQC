"""
Multi-qutrit quantum register.

An *n*-qutrit register lives in :math:`3^n`-dimensional Hilbert space.
The state is stored as a complex state vector of length :math:`3^n`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Sequence, Union
import functools


class QutritRegister:
    """
    N-qutrit quantum register.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits (≥ 1).
    initial_state : array-like or None, optional
        Full :math:`3^n`-dimensional initial state vector. If *None* the
        register is initialised to |00…0⟩.

    Examples
    --------
    >>> reg = QutritRegister(2)             # |00⟩, 9-dim vector
    >>> reg.apply_single_gate(X3, 0)
    >>> reg.measure_all()                   # e.g. [1, 0]
    """

    def __init__(
        self,
        num_qutrits: int,
        initial_state: Optional[NDArray[np.complexfloating]] = None,
    ) -> None:
        if num_qutrits < 1:
            raise ValueError("num_qutrits must be ≥ 1.")
        self._n: int = num_qutrits
        self._dim: int = 3 ** num_qutrits

        if initial_state is not None:
            sv = np.asarray(initial_state, dtype=np.complex128).ravel()
            if sv.shape[0] != self._dim:
                raise ValueError(
                    f"State vector length must be {self._dim}, got {sv.shape[0]}"
                )
            norm = np.linalg.norm(sv)
            if norm < 1e-15:
                raise ValueError("State vector cannot be zero.")
            self._state: NDArray[np.complexfloating] = sv / norm
        else:
            self._state = np.zeros(self._dim, dtype=np.complex128)
            self._state[0] = 1.0 + 0.0j

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_qutrits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def state_vector(self) -> NDArray[np.complexfloating]:
        return self._state.copy()

    @state_vector.setter
    def state_vector(self, sv: NDArray[np.complexfloating]) -> None:
        sv = np.asarray(sv, dtype=np.complex128).ravel()
        if sv.shape[0] != self._dim:
            raise ValueError(f"State vector length must be {self._dim}")
        self._state = sv / np.linalg.norm(sv)

    # ------------------------------------------------------------------
    # Gate application (tensor contraction approach)
    # ------------------------------------------------------------------
    def apply_single_gate(
        self, gate: NDArray[np.complexfloating], target: int
    ) -> "QutritRegister":
        """
        Apply a 3×3 gate to *target* qutrit using efficient tensor contraction.

        Parameters
        ----------
        gate : ndarray, shape (3, 3)
            Unitary gate matrix.
        target : int
            Index of the target qutrit (0-indexed).

        Returns
        -------
        QutritRegister
            Self, for chaining.
        """
        gate = np.asarray(gate, dtype=np.complex128)
        if gate.shape != (3, 3):
            raise ValueError(f"Single-qutrit gate must be 3×3, got {gate.shape}")
        if target < 0 or target >= self._n:
            raise IndexError(f"target={target} out of range for {self._n}-qutrit register")

        # Reshape state into tensor of shape (3, 3, ..., 3)
        tensor = self._state.reshape([3] * self._n)
        # Contract gate along the target axis
        tensor = np.tensordot(gate, tensor, axes=([1], [target]))
        # tensordot puts the new axis at position 0; move it back
        tensor = np.moveaxis(tensor, 0, target)
        self._state = tensor.reshape(self._dim)
        return self

    def apply_two_gate(
        self,
        gate: NDArray[np.complexfloating],
        qutrits: Sequence[int],
    ) -> "QutritRegister":
        """
        Apply a 9×9 two-qutrit gate on the specified pair.

        Parameters
        ----------
        gate : ndarray, shape (9, 9)
            Two-qutrit unitary gate.
        qutrits : sequence of 2 ints
            Indices [control, target] or [qutrit_a, qutrit_b].

        Returns
        -------
        QutritRegister
            Self, for chaining.
        """
        gate = np.asarray(gate, dtype=np.complex128)
        if gate.shape != (9, 9):
            raise ValueError(f"Two-qutrit gate must be 9×9, got {gate.shape}")
        if len(qutrits) != 2:
            raise ValueError("Must specify exactly 2 qutrit indices.")
        q0, q1 = int(qutrits[0]), int(qutrits[1])
        if q0 == q1:
            raise ValueError("Qutrit indices must be distinct.")
        for q in (q0, q1):
            if q < 0 or q >= self._n:
                raise IndexError(f"qutrit index {q} out of range")

        # Reshape gate to (3, 3, 3, 3)
        gate_tensor = gate.reshape(3, 3, 3, 3)  # (out0, out1, in0, in1)
        # Reshape state
        tensor = self._state.reshape([3] * self._n)

        # Contract: sum over in0 (axis q0) and in1 (axis q1)
        # We permute so that q0, q1 are last two axes, contract, permute back
        n = self._n
        axes_order = [i for i in range(n) if i not in (q0, q1)] + [q0, q1]
        inv_order = [0] * n
        for new_pos, old_pos in enumerate(axes_order):
            inv_order[old_pos] = new_pos
        tensor = np.transpose(tensor, axes_order)
        # tensor shape: (..., 3, 3) last two are q0, q1
        old_shape = tensor.shape
        batch = int(np.prod(old_shape[:-2])) if n > 2 else 1
        tensor = tensor.reshape(batch, 3, 3)
        # gate_tensor: (o0, o1, i0, i1), tensor column: (batch, i0, i1)
        # result (batch, o0, o1)
        result = np.einsum("abcd,xcd->xab", gate_tensor, tensor)
        result = result.reshape(old_shape)
        tensor = np.transpose(result, [inv_order[i] for i in range(n)])
        self._state = tensor.reshape(self._dim)
        return self

    def apply_gate(
        self, gate: NDArray[np.complexfloating], targets: Sequence[int]
    ) -> "QutritRegister":
        """
        Apply a gate to the specified target qutrits.

        Dispatches to ``apply_single_gate`` or ``apply_two_gate`` based
        on the number of targets.
        """
        targets = list(targets)
        if len(targets) == 1:
            return self.apply_single_gate(gate, targets[0])
        elif len(targets) == 2:
            return self.apply_two_gate(gate, targets)
        else:
            # General multi-qutrit gate via full matrix
            k = len(targets)
            size = 3 ** k
            gate = np.asarray(gate, dtype=np.complex128)
            if gate.shape != (size, size):
                raise ValueError(f"Gate for {k} qutrits must be {size}×{size}")
            full = self._build_full_matrix(gate, targets)
            self._state = full @ self._state
            return self

    def _build_full_matrix(
        self, gate: NDArray[np.complexfloating], targets: Sequence[int]
    ) -> NDArray[np.complexfloating]:
        """Build the full :math:`3^n × 3^n` matrix by Kronecker expansion."""
        n = self._n
        dim = self._dim
        full = np.zeros((dim, dim), dtype=np.complex128)
        targets = list(targets)
        k = len(targets)
        size = 3 ** k
        for row in range(dim):
            digits_row = self._index_to_trits(row)
            for col in range(dim):
                digits_col = self._index_to_trits(col)
                # Check that non-target digits match
                match = True
                for i in range(n):
                    if i not in targets and digits_row[i] != digits_col[i]:
                        match = False
                        break
                if not match:
                    continue
                # Extract target sub-indices
                sub_row = sum(digits_row[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
                sub_col = sum(digits_col[targets[j]] * (3 ** (k - 1 - j)) for j in range(k))
                full[row, col] = gate[sub_row, sub_col]
        return full

    def _index_to_trits(self, index: int) -> List[int]:
        """Convert flat index to list of trit values."""
        digits = []
        for _ in range(self._n):
            digits.append(index % 3)
            index //= 3
        return list(reversed(digits))

    def _trits_to_index(self, trits: Sequence[int]) -> int:
        """Convert list of trit values to flat index."""
        idx = 0
        for t in trits:
            idx = idx * 3 + t
        return idx

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def probabilities(self) -> NDArray[np.floating]:
        """Return array of :math:`3^n` measurement probabilities."""
        return np.abs(self._state) ** 2

    def measure_all(self, rng: Optional[np.random.Generator] = None) -> List[int]:
        """
        Measure all qutrits, collapsing the state.

        Returns
        -------
        list of int
            List of outcomes [q0, q1, ..., q_{n-1}], each 0, 1, or 2.
        """
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        outcome = int(rng.choice(self._dim, p=probs))
        trits = self._index_to_trits(outcome)
        # Collapse
        self._state = np.zeros(self._dim, dtype=np.complex128)
        self._state[outcome] = 1.0 + 0.0j
        return trits

    def measure_qutrit(
        self, target: int, rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Measure a single qutrit, collapsing the state.

        Parameters
        ----------
        target : int
            Qutrit index to measure.

        Returns
        -------
        int
            Outcome 0, 1, or 2.
        """
        if rng is None:
            rng = np.random.default_rng()
        if target < 0 or target >= self._n:
            raise IndexError(f"target={target} out of range")

        # Compute marginal probabilities for the target qutrit
        tensor = self._state.reshape([3] * self._n)
        axes_to_sum = tuple(i for i in range(self._n) if i != target)
        marginal = np.sum(np.abs(tensor) ** 2, axis=axes_to_sum)
        marginal = np.real(marginal)
        marginal /= marginal.sum()  # guard against fp drift

        outcome = int(rng.choice(3, p=marginal))

        # Collapse: zero out amplitudes where target qutrit != outcome
        new_tensor = tensor.copy()
        for v in range(3):
            if v != outcome:
                idx = [slice(None)] * self._n
                idx[target] = v
                new_tensor[tuple(idx)] = 0.0
        # Re-normalise
        new_state = new_tensor.reshape(self._dim)
        norm = np.linalg.norm(new_state)
        if norm > 1e-15:
            new_state /= norm
        self._state = new_state
        return outcome

    # ------------------------------------------------------------------
    # Partial trace
    # ------------------------------------------------------------------
    def partial_trace(self, keep: Sequence[int]) -> NDArray[np.complexfloating]:
        """
        Compute the reduced density matrix by tracing out all qutrits not
        in *keep*.

        Parameters
        ----------
        keep : sequence of int
            Indices of qutrits to keep.

        Returns
        -------
        ndarray
            Density matrix of shape :math:`(3^k, 3^k)` where *k* = len(keep).
        """
        keep = sorted(keep)
        n = self._n
        trace_out = sorted(set(range(n)) - set(keep))
        k = len(keep)
        dim_keep = 3 ** k

        rho = np.outer(self._state, np.conj(self._state))
        rho_tensor = rho.reshape([3] * (2 * n))

        # Trace over axes for traced-out qutrits
        # In the tensor rho_tensor[i0,..,i_{n-1}, j0,..,j_{n-1}],
        # axis k corresponds to ket index k, axis n+k to bra index k.
        # We need to set ket_idx == bra_idx for traced-out qutrits and sum.
        for q in reversed(trace_out):
            rho_tensor = np.trace(rho_tensor, axis1=q, axis2=n + q - (n - rho_tensor.ndim // 2))
        
        # Alternative cleaner approach: build rho from scratch
        rho_reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        for r in range(dim_keep):
            trits_r = []
            tmp = r
            for _ in range(k):
                trits_r.append(tmp % 3)
                tmp //= 3
            trits_r = list(reversed(trits_r))
            for c in range(dim_keep):
                trits_c = []
                tmp = c
                for _ in range(k):
                    trits_c.append(tmp % 3)
                    tmp //= 3
                trits_c = list(reversed(trits_c))

                val = 0.0 + 0.0j
                dim_traced = 3 ** len(trace_out)
                for t in range(dim_traced):
                    trits_t = []
                    tmp2 = t
                    for _ in range(len(trace_out)):
                        trits_t.append(tmp2 % 3)
                        tmp2 //= 3
                    trits_t = list(reversed(trits_t))

                    full_row = [0] * n
                    full_col = [0] * n
                    for idx_k, qk in enumerate(keep):
                        full_row[qk] = trits_r[idx_k]
                        full_col[qk] = trits_c[idx_k]
                    for idx_t, qt in enumerate(trace_out):
                        full_row[qt] = trits_t[idx_t]
                        full_col[qt] = trits_t[idx_t]

                    flat_row = self._trits_to_index(full_row)
                    flat_col = self._trits_to_index(full_col)
                    val += self._state[flat_row] * np.conj(self._state[flat_col])
                rho_reduced[r, c] = val
        return rho_reduced

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> "QutritRegister":
        """Reset register to |00…0⟩."""
        self._state = np.zeros(self._dim, dtype=np.complex128)
        self._state[0] = 1.0 + 0.0j
        return self

    def copy(self) -> "QutritRegister":
        """Return a deep copy."""
        reg = QutritRegister.__new__(QutritRegister)
        reg._n = self._n
        reg._dim = self._dim
        reg._state = self._state.copy()
        return reg

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        nonzero = []
        for i in range(self._dim):
            if np.abs(self._state[i]) > 1e-10:
                trits = self._index_to_trits(i)
                label = "".join(str(t) for t in trits)
                amp = self._state[i]
                nonzero.append(f"{amp:.4f}|{label}⟩")
        return f"QutritRegister(n={self._n}, state={' + '.join(nonzero[:8])}{'...' if len(nonzero) > 8 else ''})"
