"""
Classical-to-quantum data encoders for qutrit circuits.

Each encoder maps classical feature vectors to qutrit state preparations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional


class AngleEncoder:
    """
    Encode features as rotation angles on individual qutrits.

    Each feature value is mapped to a Gell-Mann rotation angle using
    the formula: θ = feature * scale.

    Parameters
    ----------
    n_qutrits : int
        Number of qutrits available.
    gm_indices : list of int or None
        Which Gell-Mann generators (1–8) to use. Default [1, 2, 3].
    scale : float
        Scaling factor for features → angles. Default π.
    """

    def __init__(
        self,
        n_qutrits: int,
        gm_indices: Optional[list] = None,
        scale: float = np.pi,
    ):
        self.n_qutrits = n_qutrits
        self.gm_indices = gm_indices or [1, 2, 3]
        self.scale = scale

    def encode(self, features: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Encode features to rotation angles.

        Parameters
        ----------
        features : ndarray (n_features,)

        Returns
        -------
        ndarray
            Angles for circuit parameters.
        """
        features = np.asarray(features, dtype=np.float64)
        n_rotations_per_qutrit = len(self.gm_indices)
        total_params = self.n_qutrits * n_rotations_per_qutrit

        angles = np.zeros(total_params, dtype=np.float64)
        n_features = len(features)

        for i in range(min(n_features, total_params)):
            angles[i] = features[i] * self.scale

        return angles

    def encode_batch(
        self, features: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Encode a batch of feature vectors."""
        return np.array([self.encode(f) for f in features])

    @property
    def n_parameters(self) -> int:
        return self.n_qutrits * len(self.gm_indices)

    def __repr__(self) -> str:
        return (
            f"AngleEncoder(n_qutrits={self.n_qutrits}, "
            f"gm_indices={self.gm_indices}, scale={self.scale:.3f})"
        )


class AmplitudeEncoder:
    """
    Encode features directly as qutrit state amplitudes.

    The feature vector is normalised to unit L2 norm and padded/truncated
    to match the Hilbert space dimension (3^n_qutrits).

    Parameters
    ----------
    n_qutrits : int
    """

    def __init__(self, n_qutrits: int):
        self.n_qutrits = n_qutrits
        self.dim = 3 ** n_qutrits

    def encode(self, features: NDArray[np.floating]) -> NDArray[np.complexfloating]:
        """
        Encode features as a normalised state vector.

        Parameters
        ----------
        features : ndarray

        Returns
        -------
        ndarray of shape (3^n_qutrits,)
            Normalised complex state vector.
        """
        features = np.asarray(features, dtype=np.complex128)

        # Pad or truncate
        state = np.zeros(self.dim, dtype=np.complex128)
        n = min(len(features), self.dim)
        state[:n] = features[:n]

        norm = np.linalg.norm(state)
        if norm < 1e-15:
            state[0] = 1.0
        else:
            state /= norm

        return state

    def encode_batch(
        self, features: NDArray[np.floating]
    ) -> NDArray[np.complexfloating]:
        return np.array([self.encode(f) for f in features])

    def __repr__(self) -> str:
        return f"AmplitudeEncoder(n_qutrits={self.n_qutrits}, dim={self.dim})"


class BasisEncoder:
    """
    Encode integer labels as computational basis states.

    Each integer in [0, 3^n_qutrits) maps to the corresponding basis state.

    Parameters
    ----------
    n_qutrits : int
    """

    def __init__(self, n_qutrits: int):
        self.n_qutrits = n_qutrits
        self.dim = 3 ** n_qutrits

    def encode(self, label: int) -> NDArray[np.complexfloating]:
        """
        Encode an integer label as a basis state.

        Parameters
        ----------
        label : int

        Returns
        -------
        ndarray of shape (3^n_qutrits,)
        """
        if not 0 <= label < self.dim:
            raise ValueError(
                f"Label {label} out of range [0, {self.dim})"
            )
        state = np.zeros(self.dim, dtype=np.complex128)
        state[label] = 1.0
        return state

    def encode_batch(self, labels: NDArray[np.integer]) -> NDArray[np.complexfloating]:
        return np.array([self.encode(int(l)) for l in labels])

    def __repr__(self) -> str:
        return f"BasisEncoder(n_qutrits={self.n_qutrits})"


class HybridEncoder:
    """
    Hybrid encoder combining angle and amplitude encoding.

    Uses angle encoding for the first layer and optionally feeds
    remaining features through amplitude encoding on additional qutrits.

    Parameters
    ----------
    n_angle_qutrits : int
        Qutrits used for angle encoding.
    n_amplitude_qutrits : int
        Qutrits used for amplitude encoding.
    gm_indices : list of int or None
    scale : float
    """

    def __init__(
        self,
        n_angle_qutrits: int,
        n_amplitude_qutrits: int = 0,
        gm_indices: Optional[list] = None,
        scale: float = np.pi,
    ):
        self.angle_encoder = AngleEncoder(n_angle_qutrits, gm_indices, scale)
        self.n_amplitude_qutrits = n_amplitude_qutrits
        if n_amplitude_qutrits > 0:
            self.amplitude_encoder = AmplitudeEncoder(n_amplitude_qutrits)
        else:
            self.amplitude_encoder = None

        self.n_total_qutrits = n_angle_qutrits + n_amplitude_qutrits

    def encode(
        self, features: NDArray[np.floating]
    ) -> dict:
        """
        Encode features into angle parameters and optional amplitude state.

        Returns
        -------
        dict
            'angles': ndarray — angle encoding parameters
            'amplitude_state': ndarray or None — amplitude state vector
        """
        n_angle_features = self.angle_encoder.n_parameters
        angle_features = features[:n_angle_features]
        angles = self.angle_encoder.encode(angle_features)

        amp_state = None
        if self.amplitude_encoder is not None:
            amp_features = features[n_angle_features:]
            amp_state = self.amplitude_encoder.encode(amp_features)

        return {"angles": angles, "amplitude_state": amp_state}

    def __repr__(self) -> str:
        return (
            f"HybridEncoder(angle={self.angle_encoder.n_qutrits}, "
            f"amplitude={self.n_amplitude_qutrits})"
        )
