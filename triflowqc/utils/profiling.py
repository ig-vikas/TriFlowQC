"""
Profiling and benchmarking utilities.
"""

from __future__ import annotations

import time
import sys
import numpy as np
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit


class Timer:
    """
    Simple timer context manager.

    Usage
    -----
    >>> with Timer("simulation") as t:
    ...     result = sim.run(circuit)
    >>> print(t.elapsed)
    """

    def __init__(self, label: str = ""):
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            from triflowqc.utils.logging import get_logger
            logger = get_logger("profiling")
            logger.info(f"{self.label}: {self.elapsed:.4f}s")

    def __repr__(self) -> str:
        return f"Timer({self.label!r}, elapsed={self.elapsed:.4f}s)"


def profile_circuit(
    circuit: "QutritCircuit",
    shots_list: Optional[list] = None,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """
    Profile the execution time of a circuit across different shot counts.

    Parameters
    ----------
    circuit : QutritCircuit
    shots_list : list of int or None
        Shot counts to benchmark. Default: [0, 100, 1000, 10000].
    n_repeats : int
        Number of repeats per configuration.

    Returns
    -------
    dict
        Profiling results.
    """
    from triflowqc.simulator.statevector_sim import StatevectorSimulator

    if shots_list is None:
        shots_list = [0, 100, 1000, 10000]

    sim = StatevectorSimulator()
    results: Dict[str, Any] = {
        "num_qutrits": circuit.num_qutrits,
        "num_gates": len(circuit.instructions),
        "depth": circuit.depth,
        "benchmarks": [],
    }

    for shots in shots_list:
        times = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            sim.run(circuit, shots=shots)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        results["benchmarks"].append({
            "shots": shots,
            "mean_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
        })

    return results


def memory_usage(
    n_qutrits: int,
    representation: str = "statevector",
) -> Dict[str, Any]:
    """
    Estimate memory usage for a given number of qutrits.

    Parameters
    ----------
    n_qutrits : int
    representation : str
        'statevector' or 'density_matrix'.

    Returns
    -------
    dict
        Memory estimates.
    """
    dim = 3 ** n_qutrits
    bytes_per_complex = 16  # complex128

    if representation == "statevector":
        memory_bytes = dim * bytes_per_complex
    elif representation == "density_matrix":
        memory_bytes = dim * dim * bytes_per_complex
    else:
        raise ValueError(f"Unknown representation: {representation}")

    return {
        "n_qutrits": n_qutrits,
        "hilbert_dim": dim,
        "representation": representation,
        "memory_bytes": memory_bytes,
        "memory_kb": memory_bytes / 1024,
        "memory_mb": memory_bytes / (1024 ** 2),
        "memory_gb": memory_bytes / (1024 ** 3),
    }


def estimate_scaling(max_qutrits: int = 10) -> Dict[str, list]:
    """
    Show how resources scale with qutrits.

    Returns
    -------
    dict with keys: n_qutrits, hilbert_dim, sv_memory_mb, dm_memory_mb
    """
    data: Dict[str, list] = {
        "n_qutrits": [],
        "hilbert_dim": [],
        "sv_memory_mb": [],
        "dm_memory_mb": [],
    }

    for n in range(1, max_qutrits + 1):
        dim = 3 ** n
        sv_bytes = dim * 16
        dm_bytes = dim * dim * 16
        data["n_qutrits"].append(n)
        data["hilbert_dim"].append(dim)
        data["sv_memory_mb"].append(sv_bytes / (1024 ** 2))
        data["dm_memory_mb"].append(dm_bytes / (1024 ** 2))

    return data


def circuit_summary(circuit: "QutritCircuit") -> str:
    """
    Generate a text summary of circuit resources.

    Parameters
    ----------
    circuit : QutritCircuit

    Returns
    -------
    str
    """
    lines = [
        f"Circuit Summary",
        f"{'='*40}",
        f"Qutrits:     {circuit.num_qutrits}",
        f"Hilbert dim: {3 ** circuit.num_qutrits}",
        f"Parameters:  {circuit.num_parameters}",
        f"Total gates: {len(circuit.instructions)}",
        f"Depth:       {circuit.depth}",
    ]

    gate_counts: Dict[str, int] = {}
    for name, _, _, _ in circuit.instructions:
        gate_counts[name] = gate_counts.get(name, 0) + 1

    lines.append(f"\nGate counts:")
    for name, count in sorted(gate_counts.items()):
        lines.append(f"  {name}: {count}")

    mem = memory_usage(circuit.num_qutrits, "statevector")
    lines.append(f"\nMemory (statevector): {mem['memory_mb']:.2f} MB")
    mem_dm = memory_usage(circuit.num_qutrits, "density_matrix")
    lines.append(f"Memory (density matrix): {mem_dm['memory_mb']:.2f} MB")

    return "\n".join(lines)
