"""
Circuit transpiler / optimiser.

Performs basic circuit optimisations such as gate cancellation and
merging adjacent single-qutrit gates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


def optimise(circuit: "QutritCircuit", level: int = 1) -> "QutritCircuit":
    """
    Optimise a qutrit circuit.

    Parameters
    ----------
    circuit : QutritCircuit
        Circuit to optimise.
    level : int
        Optimisation level (0 = none, 1 = basic, 2 = aggressive).

    Returns
    -------
    QutritCircuit
        Optimised circuit.
    """
    from triflowqc.circuit.circuit import QutritCircuit

    new_circ = circuit.copy()
    if level >= 1:
        new_circ = _cancel_identity_pairs(new_circ)
    if level >= 2:
        new_circ = _merge_single_qubit_gates(new_circ)
    return new_circ


def _cancel_identity_pairs(circuit: "QutritCircuit") -> "QutritCircuit":
    """Cancel adjacent pairs of gates that multiply to identity."""
    from triflowqc.circuit.circuit import QutritCircuit

    instructions = list(circuit.instructions)
    changed = True
    while changed:
        changed = False
        new_instructions = []
        skip = set()
        for i in range(len(instructions)):
            if i in skip:
                continue
            if i + 1 < len(instructions) and (i + 1) not in skip:
                name_a, gate_a, targets_a, params_a = instructions[i]
                name_b, gate_b, targets_b, params_b = instructions[i + 1]
                # Only cancel fixed (non-parametric) single-qutrit gates on same target
                if (
                    targets_a == targets_b
                    and len(targets_a) == 1
                    and not params_a
                    and not params_b
                    and isinstance(gate_a, np.ndarray)
                    and isinstance(gate_b, np.ndarray)
                ):
                    product = gate_b @ gate_a
                    if np.allclose(product, np.eye(3), atol=1e-10):
                        skip.add(i)
                        skip.add(i + 1)
                        changed = True
                        continue
            new_instructions.append(instructions[i])
        instructions = new_instructions

    result = QutritCircuit(circuit.num_qutrits)
    result._instructions = instructions
    result._parameters = list(circuit.parameters)
    result._measurements = list(circuit.measurements)
    return result


def _merge_single_qubit_gates(circuit: "QutritCircuit") -> "QutritCircuit":
    """Merge consecutive fixed single-qutrit gates on the same qutrit into one."""
    from triflowqc.circuit.circuit import QutritCircuit

    instructions = list(circuit.instructions)
    merged = []
    i = 0
    while i < len(instructions):
        name, gate, targets, params = instructions[i]
        if len(targets) == 1 and not params and isinstance(gate, np.ndarray):
            combined = gate.copy()
            combined_name = name
            j = i + 1
            while j < len(instructions):
                n2, g2, t2, p2 = instructions[j]
                if t2 == targets and not p2 and isinstance(g2, np.ndarray) and len(t2) == 1:
                    combined = g2 @ combined
                    combined_name += "·" + n2
                    j += 1
                else:
                    break
            if j > i + 1:
                merged.append((combined_name, combined, targets, []))
            else:
                merged.append(instructions[i])
            i = j
        else:
            merged.append(instructions[i])
            i += 1

    result = QutritCircuit(circuit.num_qutrits)
    result._instructions = merged
    result._parameters = list(circuit.parameters)
    result._measurements = list(circuit.measurements)
    return result
