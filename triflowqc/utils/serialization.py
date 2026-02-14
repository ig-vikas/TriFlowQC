"""
Serialization utilities for models, circuits, and parameters.
"""

from __future__ import annotations

import json
import pickle
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from triflowqc.nn.models.sequential import Sequential
    from triflowqc.circuit.circuit import QutritCircuit


def save_model(model: Any, path: str) -> None:
    """
    Save a TriFlowQC model to disk via pickle.

    Parameters
    ----------
    model : Sequential or HybridModel
    path : str
        File path (recommended extension: .tfqc).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str) -> Any:
    """
    Load a TriFlowQC model from disk.

    Parameters
    ----------
    path : str

    Returns
    -------
    Model object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_circuit(
    circuit: "QutritCircuit",
    path: str,
    parameters: Optional[NDArray[np.floating]] = None,
) -> None:
    """
    Save a circuit definition to JSON.

    Parameters
    ----------
    circuit : QutritCircuit
    path : str
    parameters : ndarray or None
    """
    data = {
        "num_qutrits": circuit.num_qutrits,
        "num_parameters": circuit.num_parameters,
        "instructions": [],
    }

    for name, _, targets, param_indices in circuit.instructions:
        entry = {
            "name": name,
            "targets": targets,
            "param_indices": param_indices if param_indices else None,
        }
        data["instructions"].append(entry)

    if parameters is not None:
        data["parameters"] = parameters.tolist()
    elif circuit.parameters:
        data["parameters"] = [p.value for p in circuit.parameters]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_circuit(path: str) -> dict:
    """
    Load circuit data from JSON.

    Returns a dict with keys: num_qutrits, instructions, parameters.
    The caller is responsible for reconstructing the QutritCircuit.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict
    """
    with open(path, "r") as f:
        return json.load(f)


def save_parameters(
    parameters: NDArray[np.floating],
    path: str,
) -> None:
    """Save parameters to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, parameters)


def load_parameters(path: str) -> NDArray[np.float64]:
    """Load parameters from a .npy file."""
    return np.load(path)


def save_results(results: dict, path: str) -> None:
    """
    Save experiment results to JSON.

    Handles numpy arrays by converting to lists.
    """
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.complexfloating,)):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return obj

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Deep-convert numpy types
    clean = json.loads(json.dumps(results, default=_convert))
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r") as f:
        return json.load(f)
