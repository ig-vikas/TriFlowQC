"""Math sub-package — qutrit algebra, SU(3) utilities, and information theory."""

from triflowqc.math.su3 import (
    su3_generators,
    su3_structure_constants,
    su3_basis_decompose,
    su3_random_element,
    is_su3,
)
from triflowqc.math.algebra import (
    tensor_product,
    partial_trace_pure,
    partial_trace_mixed,
    commutator,
    anticommutator,
    matrix_exponential,
    fidelity_pure,
    fidelity_mixed,
)
from triflowqc.math.information import (
    von_neumann_entropy,
    linear_entropy,
    mutual_information,
    concurrence_qutrit,
    negativity,
)
from triflowqc.math.decompositions import (
    qr_decomposition,
    polar_decomposition,
    spectral_decomposition,
    schmidt_decomposition,
)

__all__ = [
    "su3_generators", "su3_structure_constants", "su3_basis_decompose",
    "su3_random_element", "is_su3",
    "tensor_product", "partial_trace_pure", "partial_trace_mixed",
    "commutator", "anticommutator", "matrix_exponential",
    "fidelity_pure", "fidelity_mixed",
    "von_neumann_entropy", "linear_entropy", "mutual_information",
    "concurrence_qutrit", "negativity",
    "qr_decomposition", "polar_decomposition", "spectral_decomposition",
    "schmidt_decomposition",
]
