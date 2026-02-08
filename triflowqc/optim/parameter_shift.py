"""
Parameter-shift rule for qutrit circuits.

For qutrits with Gell-Mann generators the shift is ±2π/3:

    ∂f/∂θ_k = [f(θ_k + 2π/3) − f(θ_k − 2π/3)] / (2 sin(2π/3))

This requires 2 circuit evaluations per parameter per gradient.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, List

_SHIFT = 2.0 * np.pi / 3.0
_DENOMINATOR = 2.0 * np.sin(_SHIFT)  # 2 sin(2π/3) = √3


def compute_gradient(
    model: Any,
    parameters: List[float],
    loss_fn: Any,
    X: NDArray[np.floating],
    y: NDArray,
) -> NDArray[np.floating]:
    """
    Compute the gradient of the loss w.r.t. all trainable parameters
    using the qutrit parameter-shift rule.

    Parameters
    ----------
    model : Sequential or similar
        Model with ``all_parameters`` setter and ``predict`` method.
    parameters : list of float
        Current parameter values.
    loss_fn : callable
        Loss function ``loss_fn(predictions, targets) → float``.
    X : ndarray
        Input batch.
    y : ndarray
        Target batch.

    Returns
    -------
    ndarray, shape (num_parameters,)
        Gradient vector.
    """
    n_params = len(parameters)
    grad = np.zeros(n_params, dtype=np.float64)

    for k in range(n_params):
        grad[k] = compute_gradient_single_param(model, parameters, k, loss_fn, X, y)

    return grad


def compute_gradient_single_param(
    model: Any,
    parameters: List[float],
    param_index: int,
    loss_fn: Any,
    X: NDArray[np.floating],
    y: NDArray,
    shift: float = _SHIFT,
) -> float:
    """
    Compute the gradient for a single parameter using parameter-shift.

    ∂L/∂θ_k = [L(θ_k + s) − L(θ_k − s)] / (2 sin(s))

    Parameters
    ----------
    model : model object
    parameters : list of float
    param_index : int
    loss_fn : callable
    X, y : data
    shift : float
        Shift amount (default 2π/3 for qutrits).

    Returns
    -------
    float
        Gradient for parameter ``param_index``.
    """
    denom = 2.0 * np.sin(shift)

    # Shift +
    params_plus = list(parameters)
    params_plus[param_index] += shift
    model.all_parameters = params_plus
    preds_plus = model.predict(X, shots=0)
    loss_plus = loss_fn(preds_plus, y)

    # Shift −
    params_minus = list(parameters)
    params_minus[param_index] -= shift
    model.all_parameters = params_minus
    preds_minus = model.predict(X, shots=0)
    loss_minus = loss_fn(preds_minus, y)

    # Restore original
    model.all_parameters = list(parameters)

    return float((loss_plus - loss_minus) / denom)


# ------------------------------------------------------------------
# Standalone parameter-shift gradient for simple scalar functions
# ------------------------------------------------------------------

def parameter_shift_gradient(
    fn: Callable[[NDArray[np.floating]], float],
    params: NDArray[np.floating],
    shift: float = _SHIFT,
) -> NDArray[np.floating]:
    """
    Compute parameter-shift gradient for a simple scalar function.

    Parameters
    ----------
    fn : callable
        Function ``fn(params) → float`` that returns a scalar cost.
    params : ndarray
        Parameter vector.
    shift : float
        Shift amount (default 2π/3 for qutrits).

    Returns
    -------
    ndarray, shape (len(params),)
        Gradient vector.
    """
    params = np.asarray(params, dtype=np.float64)
    n = len(params)
    grad = np.zeros(n, dtype=np.float64)
    denom = 2.0 * np.sin(shift)

    for k in range(n):
        p_plus = params.copy()
        p_plus[k] += shift
        p_minus = params.copy()
        p_minus[k] -= shift
        grad[k] = (fn(p_plus) - fn(p_minus)) / denom

    return grad
