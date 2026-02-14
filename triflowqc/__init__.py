"""
TriFlowQC — Qutrit Quantum Neural Network Framework.

A TensorFlow-style framework for 3-state quantum computing (qutrits) with
real IBM Quantum hardware support via qutrit-to-qubit compilation.

Core subpackages:
    core       — Qutrit state, register, density matrix, variables
    gates      — Gell-Mann matrices, single/two/three-qutrit gates
    circuit    — Circuit builder, measurement, visualization
    nn         — Neural network layers, models, losses, metrics
    optim      — Optimizers and gradient computation
    simulator  — State-vector and density matrix simulators
    compiler   — Qutrit-to-qubit compilation for real hardware
    hardware   — IBM Quantum backend interface
    data       — Dataset and data loading utilities
    math       — Linear algebra, SU(3), fidelity, entropy
    utils      — Visualization, serialization, logging, validators
"""

__version__ = "1.0.0"
__author__ = "TriFlowQC Contributors"

from triflowqc.core.qutrit import Qutrit
from triflowqc.core.register import QutritRegister
from triflowqc.core.variable import Variable

__all__ = [
    "Qutrit",
    "QutritRegister",
    "Variable",
    "__version__",
]
