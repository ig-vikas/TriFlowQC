"""
Leakage handler for the qutrit-to-qubit encoding.

Detects and mitigates leakage into the forbidden |11⟩ qubit sub-space that
lies outside the valid qutrit encoding.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional, List, Tuple


# Valid qubit-pair states and their qutrit mapping
VALID_QUBIT_PAIRS = {(0, 0): 0, (0, 1): 1, (1, 0): 2}
FORBIDDEN_PAIR = (1, 1)


class LeakageHandler:
    """
    Handles leakage detection and post-selection for qutrit-to-qubit experiments.

    When qutrit circuits are compiled to qubit circuits, the |11⟩ state of
    each qubit pair is outside the qutrit code space. This handler:
    1. Identifies leaked shots.
    2. Optionally post-selects valid shots.
    3. Reports leakage statistics.

    Parameters
    ----------
    n_qutrits : int
        Number of qutrits (each = 2 qubits).
    post_select : bool
        If True, discard leaked shots.
    leakage_fill : str
        How to handle leaked shots when not post-selecting:
        'nearest' — map |11⟩ to nearest valid state |10⟩.
        'discard' — drop leaked shots.
        'flag' — keep but flag them.
    """

    def __init__(
        self,
        n_qutrits: int,
        post_select: bool = True,
        leakage_fill: str = "discard",
    ):
        self.n_qutrits = n_qutrits
        self.n_qubits = 2 * n_qutrits
        self.post_select = post_select
        self.leakage_fill = leakage_fill

    def is_leaked(self, bitstring: str) -> bool:
        """
        Check if a qubit bitstring contains leakage.

        Parameters
        ----------
        bitstring : str
            Binary string of length ``2 * n_qutrits`` (MSB first).

        Returns
        -------
        bool
        """
        bits = [int(b) for b in bitstring]
        for q in range(self.n_qutrits):
            a = bits[2 * q]
            b = bits[2 * q + 1]
            if (a, b) == FORBIDDEN_PAIR:
                return True
        return False

    def bitstring_to_trits(self, bitstring: str) -> Optional[str]:
        """
        Convert a qubit bitstring to a trit string.

        Returns None if the bitstring contains leakage.

        Parameters
        ----------
        bitstring : str
            Binary string of length 2*n_qutrits.

        Returns
        -------
        str or None
            Trit string (base-3 digits) or None if leaked.
        """
        bits = [int(b) for b in bitstring]
        trits = []
        for q in range(self.n_qutrits):
            pair = (bits[2 * q], bits[2 * q + 1])
            if pair == FORBIDDEN_PAIR:
                if self.leakage_fill == "nearest":
                    trits.append(2)  # |11⟩ closest to |10⟩ = |2⟩
                else:
                    return None
            else:
                trits.append(VALID_QUBIT_PAIRS[pair])
        return "".join(str(t) for t in trits)

    def filter_counts(
        self, qubit_counts: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Separate valid and leaked counts.

        Parameters
        ----------
        qubit_counts : dict
            {qubit_bitstring: count}

        Returns
        -------
        valid_trit_counts : dict
            {trit_string: count} for valid (non-leaked) shots.
        leaked_counts : dict
            {qubit_bitstring: count} for leaked shots.
        """
        valid: Dict[str, int] = {}
        leaked: Dict[str, int] = {}

        for bitstring, count in qubit_counts.items():
            trit_string = self.bitstring_to_trits(bitstring)
            if trit_string is not None:
                valid[trit_string] = valid.get(trit_string, 0) + count
            else:
                leaked[bitstring] = leaked.get(bitstring, 0) + count

        return valid, leaked

    def post_select_counts(
        self, qubit_counts: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Return only valid (non-leaked) trit counts.

        Parameters
        ----------
        qubit_counts : dict
            Raw qubit measurement counts.

        Returns
        -------
        dict
            {trit_string: count}
        """
        valid, _ = self.filter_counts(qubit_counts)
        return valid

    def leakage_rate(self, qubit_counts: Dict[str, int]) -> float:
        """
        Compute the fraction of shots that leaked.

        Parameters
        ----------
        qubit_counts : dict

        Returns
        -------
        float
            Leakage rate in [0, 1].
        """
        total = sum(qubit_counts.values())
        if total == 0:
            return 0.0
        _, leaked = self.filter_counts(qubit_counts)
        leaked_total = sum(leaked.values())
        return leaked_total / total

    def leakage_report(self, qubit_counts: Dict[str, int]) -> Dict:
        """
        Generate a detailed leakage report.

        Returns
        -------
        dict
            Keys: total_shots, valid_shots, leaked_shots, leakage_rate,
            valid_trit_counts, leaked_qubit_counts.
        """
        valid, leaked = self.filter_counts(qubit_counts)
        total = sum(qubit_counts.values())
        valid_total = sum(valid.values())
        leaked_total = sum(leaked.values())

        return {
            "total_shots": total,
            "valid_shots": valid_total,
            "leaked_shots": leaked_total,
            "leakage_rate": leaked_total / total if total > 0 else 0.0,
            "valid_trit_counts": valid,
            "leaked_qubit_counts": leaked,
        }

    def process(self, qubit_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Main entry point: process qubit counts into trit counts with
        leakage handling according to configuration.

        Returns
        -------
        dict
            {trit_string: count}
        """
        if self.post_select:
            return self.post_select_counts(qubit_counts)

        # Non-post-selection modes
        valid: Dict[str, int] = {}
        for bitstring, count in qubit_counts.items():
            trit_string = self.bitstring_to_trits(bitstring)
            if trit_string is not None:
                valid[trit_string] = valid.get(trit_string, 0) + count
            elif self.leakage_fill == "flag":
                # Keep the raw bitstring prefixed with "LEAKED:"
                valid[f"LEAKED:{bitstring}"] = (
                    valid.get(f"LEAKED:{bitstring}", 0) + count
                )
            # 'discard' → simply drop
        return valid
