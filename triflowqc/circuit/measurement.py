"""
Measurement operations and result container.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional


class MeasurementResult:
    """
    Container for circuit execution results.

    Attributes
    ----------
    counts : dict[str, int]
        Mapping of base-3 outcome strings to counts.
    probabilities : ndarray
        Array of :math:`3^n` probabilities.
    statevector : ndarray or None
        Full state vector (only if simulation was run in statevector mode).
    shots : int
        Number of shots.
    num_qutrits : int
        Number of qutrits measured.
    """

    def __init__(
        self,
        counts: Dict[str, int],
        probabilities: NDArray[np.floating],
        statevector: Optional[NDArray[np.complexfloating]] = None,
        shots: int = 0,
        num_qutrits: int = 1,
    ) -> None:
        self.counts = dict(counts)
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.statevector = statevector
        self.shots = shots
        self.num_qutrits = num_qutrits

    def expectation(self, observable: NDArray[np.complexfloating]) -> complex:
        """Compute ⟨ψ|O|ψ⟩ from the state vector."""
        if self.statevector is None:
            raise RuntimeError("No state vector available; run simulation with shots=0.")
        return complex(np.conj(self.statevector) @ observable @ self.statevector)

    def most_probable(self) -> str:
        """Return the outcome string with the highest count."""
        if not self.counts:
            return ""
        return max(self.counts, key=lambda k: self.counts[k])

    def marginal(self, qutrit_indices: List[int]) -> Dict[str, int]:
        """
        Compute marginal counts over a subset of qutrits.

        Parameters
        ----------
        qutrit_indices : list of int
            Which qutrits to keep (0-indexed).

        Returns
        -------
        dict[str, int]
            Marginal counts.
        """
        result: Dict[str, int] = {}
        for bitstring, count in self.counts.items():
            sub = "".join(bitstring[i] for i in qutrit_indices)
            result[sub] = result.get(sub, 0) + count
        return result

    def histogram(self) -> Dict[str, float]:
        """Return normalised histogram (probabilities from counts)."""
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.counts.items()}

    def __repr__(self) -> str:
        top = sorted(self.counts.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"'{k}': {v}" for k, v in top)
        return f"MeasurementResult(shots={self.shots}, top_counts={{{top_str}}})"


def sample_from_probabilities(
    probabilities: NDArray[np.floating],
    num_qutrits: int,
    shots: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    """
    Sample measurement outcomes from a probability distribution.

    Parameters
    ----------
    probabilities : ndarray, shape (3^n,)
        Born-rule probabilities.
    num_qutrits : int
        Number of qutrits.
    shots : int
        Number of samples.
    rng : numpy random generator, optional

    Returns
    -------
    dict[str, int]
        Outcome string → count.
    """
    if rng is None:
        rng = np.random.default_rng()
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.maximum(probs, 0.0)
    total = probs.sum()
    if total > 0:
        probs /= total
    else:
        probs = np.ones_like(probs) / len(probs)

    outcomes = rng.choice(len(probs), size=shots, p=probs)
    counts: Dict[str, int] = {}
    for o in outcomes:
        label = _index_to_trit_string(int(o), num_qutrits)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _index_to_trit_string(index: int, num_qutrits: int) -> str:
    """Convert flat index to base-3 string of length *num_qutrits*."""
    digits = []
    for _ in range(num_qutrits):
        digits.append(str(index % 3))
        index //= 3
    return "".join(reversed(digits))
