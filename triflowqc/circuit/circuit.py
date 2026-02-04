"""
Qutrit quantum circuit builder.

Stores a sequence of gate operations and provides methods to add gates,
set/get parameters, visualise the circuit, and compile it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

from triflowqc.gates.single_qutrit import (
    X3, X3_01, X3_12, X3_02, Z3, H3, T3, S3, Phase3, Identity3,
)
from triflowqc.gates.parametric import RX3, RY3, RZ3, R_GM
from triflowqc.gates.two_qutrit import CSUM, CPHASE, SWAP3, CX3

# Type for a circuit instruction
# (gate_name, matrix_or_callable, targets, param_indices)
Instruction = Tuple[str, Any, List[int], List[int]]


class QutritCircuit:
    """
    Quantum circuit for qutrits.

    Parameters
    ----------
    num_qutrits : int
        Number of qutrits in the circuit.

    Examples
    --------
    >>> circ = QutritCircuit(3)
    >>> circ.h3(0)
    >>> circ.csum(0, 1)
    >>> circ.rx3(0.5, 2)
    >>> circ.measure_all()
    >>> print(circ.draw())
    """

    def __init__(self, num_qutrits: int) -> None:
        if num_qutrits < 1:
            raise ValueError("num_qutrits must be ≥ 1.")
        self._num_qutrits = num_qutrits
        self._instructions: List[Instruction] = []
        self._parameters: List[float] = []
        self._measurements: List[int] = []  # qutrits to measure

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_qutrits(self) -> int:
        return self._num_qutrits

    @property
    def num_parameters(self) -> int:
        return len(self._parameters)

    @property
    def parameters(self) -> List[float]:
        return list(self._parameters)

    @property
    def depth(self) -> int:
        """Circuit depth (number of time-step layers)."""
        if not self._instructions:
            return 0
        # Simple greedy depth calculation
        qutrit_depth = [0] * self._num_qutrits
        for _, _, targets, _ in self._instructions:
            max_d = max(qutrit_depth[t] for t in targets) + 1
            for t in targets:
                qutrit_depth[t] = max_d
        return max(qutrit_depth) if qutrit_depth else 0

    @property
    def instructions(self) -> List[Instruction]:
        return list(self._instructions)

    @property
    def measurements(self) -> List[int]:
        return list(self._measurements)

    def set_parameters(self, values: Sequence[float]) -> None:
        """Set all trainable parameters at once."""
        values = list(values)
        if len(values) != len(self._parameters):
            raise ValueError(
                f"Expected {len(self._parameters)} parameters, got {len(values)}"
            )
        self._parameters = [float(v) for v in values]

    def get_parameter(self, index: int) -> float:
        return self._parameters[index]

    def set_parameter(self, index: int, value: float) -> None:
        self._parameters[index] = float(value)

    # ------------------------------------------------------------------
    # Single-qutrit gates (fixed)
    # ------------------------------------------------------------------
    def x3(self, target: int) -> "QutritCircuit":
        """Add X3 (cyclic shift) gate on *target*."""
        self._check_target(target)
        self._instructions.append(("X3", X3, [target], []))
        return self

    def x3_01(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("X3_01", X3_01, [target], []))
        return self

    def x3_12(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("X3_12", X3_12, [target], []))
        return self

    def x3_02(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("X3_02", X3_02, [target], []))
        return self

    def z3(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("Z3", Z3, [target], []))
        return self

    def h3(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("H3", H3, [target], []))
        return self

    def t3(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("T3", T3, [target], []))
        return self

    def s3(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("S3", S3, [target], []))
        return self

    def identity(self, target: int) -> "QutritCircuit":
        self._check_target(target)
        self._instructions.append(("I3", Identity3, [target], []))
        return self

    # ------------------------------------------------------------------
    # Single-qutrit gates (parametric)
    # ------------------------------------------------------------------
    def rx3(self, theta: float, target: int) -> "QutritCircuit":
        """Add RX3(θ) gate on *target*."""
        self._check_target(target)
        pidx = len(self._parameters)
        self._parameters.append(float(theta))
        self._instructions.append(("RX3", lambda th: RX3(th), [target], [pidx]))
        return self

    def ry3(self, theta: float, target: int) -> "QutritCircuit":
        self._check_target(target)
        pidx = len(self._parameters)
        self._parameters.append(float(theta))
        self._instructions.append(("RY3", lambda th: RY3(th), [target], [pidx]))
        return self

    def rz3(self, theta: float, target: int) -> "QutritCircuit":
        self._check_target(target)
        pidx = len(self._parameters)
        self._parameters.append(float(theta))
        self._instructions.append(("RZ3", lambda th: RZ3(th), [target], [pidx]))
        return self

    def r_gm(self, k: int, theta: float, target: int) -> "QutritCircuit":
        """Add R_GM(k, θ) rotation on *target*."""
        self._check_target(target)
        pidx = len(self._parameters)
        self._parameters.append(float(theta))
        self._instructions.append(
            (f"R{k}", lambda th, _k=k: R_GM(_k, th), [target], [pidx])
        )
        return self

    def phase3(self, phi1: float, phi2: float, target: int) -> "QutritCircuit":
        self._check_target(target)
        pidx1 = len(self._parameters)
        self._parameters.append(float(phi1))
        pidx2 = len(self._parameters)
        self._parameters.append(float(phi2))
        self._instructions.append(
            ("Phase3", lambda p1, p2: Phase3(p1, p2), [target], [pidx1, pidx2])
        )
        return self

    # ------------------------------------------------------------------
    # Two-qutrit gates
    # ------------------------------------------------------------------
    def csum(self, control: int, target: int) -> "QutritCircuit":
        """Add CSUM gate (qutrit CNOT analogue)."""
        self._check_target(control)
        self._check_target(target)
        if control == target:
            raise ValueError("Control and target must be different.")
        self._instructions.append(("CSUM", CSUM, [control, target], []))
        return self

    def cphase(self, control: int, target: int) -> "QutritCircuit":
        self._check_target(control)
        self._check_target(target)
        if control == target:
            raise ValueError("Control and target must be different.")
        self._instructions.append(("CPHASE", CPHASE, [control, target], []))
        return self

    def swap3(self, q0: int, q1: int) -> "QutritCircuit":
        self._check_target(q0)
        self._check_target(q1)
        if q0 == q1:
            raise ValueError("Swap targets must be different.")
        self._instructions.append(("SWAP3", SWAP3, [q0, q1], []))
        return self

    def cx3(self, control: int, target: int) -> "QutritCircuit":
        self._check_target(control)
        self._check_target(target)
        if control == target:
            raise ValueError("Control and target must be different.")
        self._instructions.append(("CX3", CX3, [control, target], []))
        return self

    # ------------------------------------------------------------------
    # Custom gate
    # ------------------------------------------------------------------
    def custom_gate(
        self, matrix: NDArray[np.complexfloating], targets: Sequence[int], name: str = "U"
    ) -> "QutritCircuit":
        """Add an arbitrary unitary gate to the circuit."""
        targets = list(targets)
        for t in targets:
            self._check_target(t)
        matrix = np.asarray(matrix, dtype=np.complex128)
        expected = 3 ** len(targets)
        if matrix.shape != (expected, expected):
            raise ValueError(f"Gate for {len(targets)} qutrits must be {expected}×{expected}")
        self._instructions.append((name, matrix, targets, []))
        return self

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------
    def measure(self, target: int) -> "QutritCircuit":
        """Mark *target* qutrit for measurement."""
        self._check_target(target)
        if target not in self._measurements:
            self._measurements.append(target)
        return self

    def measure_all(self) -> "QutritCircuit":
        """Mark all qutrits for measurement."""
        self._measurements = list(range(self._num_qutrits))
        return self

    # ------------------------------------------------------------------
    # Gate resolution (materialise parametric gates)
    # ------------------------------------------------------------------
    def resolve_gate(self, instruction: Instruction) -> NDArray[np.complexfloating]:
        """
        Resolve an instruction to a concrete matrix using current parameters.
        """
        name, gate_or_fn, targets, param_indices = instruction
        if callable(gate_or_fn) and not isinstance(gate_or_fn, np.ndarray):
            args = [self._parameters[pi] for pi in param_indices]
            return np.asarray(gate_or_fn(*args), dtype=np.complex128)
        return np.asarray(gate_or_fn, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw(self, style: str = "ascii") -> str:
        """
        Draw the circuit.

        Parameters
        ----------
        style : str
            ``'ascii'`` for text diagram.

        Returns
        -------
        str
        """
        if style == "ascii":
            return self._draw_ascii()
        raise ValueError(f"Unknown style: {style}")

    def _draw_ascii(self) -> str:
        lines = [f"q{i}: ──" for i in range(self._num_qutrits)]
        for name, _, targets, param_indices in self._instructions:
            label = name
            if param_indices:
                params_str = ",".join(f"{self._parameters[p]:.2f}" for p in param_indices)
                label = f"{name}({params_str})"
            max_len = max(len(l) for l in lines)
            lines = [l.ljust(max_len, "─") for l in lines]
            if len(targets) == 1:
                t = targets[0]
                lines[t] += f"─[{label}]─"
            elif len(targets) == 2:
                t_min, t_max = min(targets), max(targets)
                for i in range(self._num_qutrits):
                    if i == targets[0]:
                        lines[i] += f"─●{label}─"
                    elif i == targets[1]:
                        lines[i] += f"─⊕{label}─"
                    elif t_min < i < t_max:
                        lines[i] += "─" * (len(label) + 3)
                    else:
                        lines[i] += "─" * (len(label) + 3)
            else:
                for i in range(self._num_qutrits):
                    if i in targets:
                        lines[i] += f"─[{label}]─"
                    else:
                        lines[i] += "─" * (len(label) + 4)

        # Measurements
        if self._measurements:
            max_len = max(len(l) for l in lines)
            lines = [l.ljust(max_len, "─") for l in lines]
            for m in self._measurements:
                lines[m] += "─M─"

        max_len = max(len(l) for l in lines)
        lines = [l.ljust(max_len, "─") for l in lines]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------
    def copy(self) -> "QutritCircuit":
        return deepcopy(self)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_target(self, t: int) -> None:
        if t < 0 or t >= self._num_qutrits:
            raise IndexError(f"Target {t} out of range for {self._num_qutrits}-qutrit circuit")

    def __repr__(self) -> str:
        return (
            f"QutritCircuit(num_qutrits={self._num_qutrits}, "
            f"depth={self.depth}, params={self.num_parameters})"
        )
