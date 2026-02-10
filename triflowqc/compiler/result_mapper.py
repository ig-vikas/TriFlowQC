"""
Result mapper for converting qubit measurement results back to qutrit space.

Maps IBM Quantum hardware results (qubit counts) to qutrit distributions
using the qutrit-to-qubit encoding and leakage handling.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple

from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.circuit.measurement import MeasurementResult


class ResultMapper:
    """
    Maps qubit measurement results from compiled circuits back to
    qutrit measurement results.

    Parameters
    ----------
    n_qutrits : int
        Number of qutrits in the original circuit.
    post_select : bool
        Whether to post-select against leakage.
    error_mitigate : bool
        Whether to apply basic readout error mitigation.
    """

    def __init__(
        self,
        n_qutrits: int,
        post_select: bool = True,
        error_mitigate: bool = False,
    ):
        self.n_qutrits = n_qutrits
        self.post_select = post_select
        self.error_mitigate = error_mitigate
        self.leakage_handler = LeakageHandler(n_qutrits, post_select=post_select)

    def map_counts(
        self, qubit_counts: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Convert qubit counts to qutrit (trit) counts.

        Parameters
        ----------
        qubit_counts : dict
            Raw {qubit_bitstring: count} from hardware.

        Returns
        -------
        dict
            {trit_string: count}
        """
        return self.leakage_handler.process(qubit_counts)

    def map_to_measurement_result(
        self,
        qubit_counts: Dict[str, int],
        shots: Optional[int] = None,
    ) -> MeasurementResult:
        """
        Convert qubit counts to a full MeasurementResult in qutrit space.

        Parameters
        ----------
        qubit_counts : dict
        shots : int or None
            Total shots. If None, inferred from counts.

        Returns
        -------
        MeasurementResult
        """
        trit_counts = self.map_counts(qubit_counts)
        total = sum(trit_counts.values())
        if shots is None:
            shots = sum(qubit_counts.values())

        # Build probability array
        dim = 3 ** self.n_qutrits
        probs = np.zeros(dim, dtype=np.float64)
        for trit_string, count in trit_counts.items():
            if trit_string.startswith("LEAKED:"):
                continue
            idx = self._trit_string_to_index(trit_string)
            probs[idx] = count

        if total > 0:
            probs /= total

        return MeasurementResult(
            counts=trit_counts,
            probabilities=probs,
            statevector=None,
            shots=shots,
            num_qutrits=self.n_qutrits,
        )

    def expectation_from_counts(
        self,
        qubit_counts: Dict[str, int],
        observable: NDArray[np.complexfloating],
    ) -> float:
        """
        Compute expectation value of a qutrit observable from qubit counts.

        Parameters
        ----------
        qubit_counts : dict
            Raw hardware counts.
        observable : ndarray
            Qutrit-space observable matrix.

        Returns
        -------
        float
            Estimated expectation value.
        """
        result = self.map_to_measurement_result(qubit_counts)
        probs = result.probabilities
        dim = len(probs)
        diag = np.real(np.diag(observable[:dim, :dim]))
        return float(np.dot(probs, diag))

    def leakage_statistics(
        self, qubit_counts: Dict[str, int]
    ) -> Dict:
        """Get leakage report."""
        return self.leakage_handler.leakage_report(qubit_counts)

    def marginal_trit_probs(
        self,
        qubit_counts: Dict[str, int],
        qutrit_indices: List[int],
    ) -> NDArray[np.float64]:
        """
        Compute marginal probability distribution over a subset of qutrits.

        Parameters
        ----------
        qubit_counts : dict
        qutrit_indices : list of int

        Returns
        -------
        ndarray
            Probability array of shape (3^k,) where k = len(qutrit_indices).
        """
        trit_counts = self.map_counts(qubit_counts)
        total = sum(c for s, c in trit_counts.items() if not s.startswith("LEAKED:"))

        k = len(qutrit_indices)
        dim = 3 ** k
        probs = np.zeros(dim, dtype=np.float64)

        for trit_string, count in trit_counts.items():
            if trit_string.startswith("LEAKED:"):
                continue
            sub_trits = [int(trit_string[i]) for i in qutrit_indices]
            idx = 0
            for j, t in enumerate(sub_trits):
                idx += t * (3 ** (k - 1 - j))
            probs[idx] += count

        if total > 0:
            probs /= total
        return probs

    def _trit_string_to_index(self, trit_string: str) -> int:
        """Convert a trit string like '012' to a flat index."""
        n = len(trit_string)
        idx = 0
        for i, ch in enumerate(trit_string):
            idx += int(ch) * (3 ** (n - 1 - i))
        return idx

    def __repr__(self) -> str:
        return (
            f"ResultMapper(n_qutrits={self.n_qutrits}, "
            f"post_select={self.post_select})"
        )
