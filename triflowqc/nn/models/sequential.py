"""
Sequential model — stack layers linearly (Keras-style).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import time
import json
import pickle

from triflowqc.nn.layers.base_layer import BaseQutritLayer
from triflowqc.nn.layers.measurement_layer import MeasurementLayer
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.circuit.measurement import MeasurementResult


class Sequential:
    """
    Sequential model for qutrit quantum neural networks.

    Parameters
    ----------
    layers : list of BaseQutritLayer
        Ordered list of layers.

    Examples
    --------
    >>> model = Sequential([
    ...     AngleEmbedding(4),
    ...     DenseQutrit(4, num_layers=2),
    ...     EntanglingLayer(4, 'ring'),
    ...     MeasurementLayer(4, num_classes=3),
    ... ])
    >>> model.compile(optimizer=Adam(lr=0.01), loss=CrossEntropyLoss())
    >>> history = model.fit(X_train, y_train, epochs=50)
    """

    def __init__(self, layers: Sequence[BaseQutritLayer]) -> None:
        self._layers = list(layers)
        self._optimizer: Any = None
        self._loss_fn: Any = None
        self._metrics: List[Any] = []
        self._compiled = False
        self._history: Dict[str, List[float]] = {}
        self._hw_token: Optional[str] = None
        self._hw_backend: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def layers(self) -> List[BaseQutritLayer]:
        return list(self._layers)

    @property
    def num_parameters(self) -> int:
        return sum(layer.num_parameters for layer in self._layers)

    @property
    def all_parameters(self) -> List[float]:
        """Flat list of all trainable parameters across layers."""
        params = []
        for layer in self._layers:
            params.extend(layer.parameters)
        return params

    @all_parameters.setter
    def all_parameters(self, values: List[float]) -> None:
        idx = 0
        for layer in self._layers:
            n = layer.num_parameters
            layer.parameters = values[idx: idx + n]
            idx += n

    @property
    def num_qutrits(self) -> int:
        if self._layers:
            return self._layers[0].num_qutrits
        return 0

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    def compile(
        self,
        optimizer: Any = None,
        loss: Any = None,
        metrics: Optional[List[Any]] = None,
    ) -> None:
        """Configure the model for training."""
        if optimizer is None:
            from triflowqc.optim.adam import Adam
            optimizer = Adam(lr=0.01)
        if loss is None:
            from triflowqc.nn.losses import CrossEntropyLoss
            loss = CrossEntropyLoss()
        self._optimizer = optimizer
        self._loss_fn = loss
        self._metrics = list(metrics) if metrics else []
        self._compiled = True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Print and return layer-by-layer summary."""
        lines = [
            "=" * 60,
            "TriFlowQC Sequential Model",
            "=" * 60,
        ]
        total_params = 0
        for i, layer in enumerate(self._layers):
            n = layer.num_parameters
            total_params += n
            lines.append(f"  [{i}] {layer.summary_line()}")
        lines.append("-" * 60)
        lines.append(f"  Total parameters: {total_params}")
        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Forward pass (single sample)
    # ------------------------------------------------------------------
    def _forward_single(
        self, x: NDArray[np.floating], shots: int = 0
    ) -> NDArray[np.floating]:
        """
        Run a single input through the model.

        Parameters
        ----------
        x : ndarray
            Single input sample.
        shots : int
            If 0, use exact statevector probabilities. Otherwise sample.

        Returns
        -------
        ndarray
            Output class probabilities.
        """
        from triflowqc.simulator.statevector_sim import StatevectorSimulator

        n = self.num_qutrits
        circ = QutritCircuit(n)

        # Check for amplitude embedding (needs special initial state)
        amplitude_state = None

        for layer in self._layers:
            layer.build_circuit(circ, x=x)
            if hasattr(circ, '_amplitude_state') and circ._amplitude_state is not None:
                amplitude_state = circ._amplitude_state
                circ._amplitude_state = None

        sim = StatevectorSimulator()
        result = sim.run(circ, shots=shots, initial_state=amplitude_state)
        probs = result.probabilities

        # Find the measurement layer to extract class probabilities
        meas_layer = None
        for layer in self._layers:
            if isinstance(layer, MeasurementLayer):
                meas_layer = layer
                break
        if meas_layer is not None:
            return meas_layer.extract_probabilities(probs, n)
        else:
            return probs

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(
        self,
        X: NDArray[np.floating],
        use_hardware: bool = False,
        shots: int = 0,
    ) -> NDArray[np.floating]:
        """
        Predict class probabilities for a batch of inputs.

        Parameters
        ----------
        X : ndarray, shape (num_samples, num_features)
            Input data.
        use_hardware : bool
            If True, run on IBM Quantum hardware.
        shots : int
            Number of shots (0 = exact).

        Returns
        -------
        ndarray, shape (num_samples, num_classes)
        """
        X = np.atleast_2d(X)
        results = []
        for i in range(X.shape[0]):
            if use_hardware:
                pred = self._forward_hardware(X[i], shots=shots if shots > 0 else 8192)
            else:
                pred = self._forward_single(X[i], shots=shots)
            results.append(pred)
        return np.array(results)

    def _forward_hardware(
        self, x: NDArray[np.floating], shots: int = 8192
    ) -> NDArray[np.floating]:
        """Run single sample on IBM Quantum hardware."""
        from triflowqc.hardware.hardware_runner import HardwareRunner

        n = self.num_qutrits
        circ = QutritCircuit(n)
        for layer in self._layers:
            layer.build_circuit(circ, x=x)

        runner = HardwareRunner(ibm_token=self._hw_token)
        result = runner.run(circ, shots=shots, backend_name=self._hw_backend)
        probs = result.qutrit_probabilities

        meas_layer = None
        for layer in self._layers:
            if isinstance(layer, MeasurementLayer):
                meas_layer = layer
                break
        if meas_layer is not None:
            return meas_layer.extract_probabilities(probs, n)
        return probs

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: NDArray[np.floating],
        y_train: NDArray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[NDArray, NDArray]] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Parameters
        ----------
        X_train : ndarray, shape (n_samples, n_features)
        y_train : ndarray, shape (n_samples,) or (n_samples, n_classes)
        epochs : int
        batch_size : int
        validation_data : tuple (X_val, y_val) or None
        callbacks : list of callback objects
        verbose : int

        Returns
        -------
        dict
            Training history with keys 'loss', 'val_loss', metric names, etc.
        """
        if not self._compiled:
            raise RuntimeError("Model must be compiled before training. Call model.compile().")

        from triflowqc.optim.parameter_shift import compute_gradient

        X_train = np.atleast_2d(X_train)
        n_samples = X_train.shape[0]
        callbacks = callbacks or []

        history: Dict[str, List[float]] = {"loss": []}
        for m in self._metrics:
            history[m.name] = []
        if validation_data is not None:
            history["val_loss"] = []
            for m in self._metrics:
                history[f"val_{m.name}"] = []

        # Callback: on_train_begin
        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(self)

        for epoch in range(epochs):
            t0 = time.time()
            epoch_losses: List[float] = []
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                batch_idx = indices[start: start + batch_size]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Compute loss and gradients
                all_params = self.all_parameters
                grad = compute_gradient(
                    model=self,
                    parameters=all_params,
                    loss_fn=self._loss_fn,
                    X=X_batch,
                    y=y_batch,
                )

                # Update parameters
                all_params = self.all_parameters
                new_params = self._optimizer.step(all_params, grad)
                self.all_parameters = new_params

                # Compute batch loss
                preds = self.predict(X_batch, shots=0)
                batch_loss = self._loss_fn(preds, y_batch)
                epoch_losses.append(float(batch_loss))

            avg_loss = float(np.mean(epoch_losses))
            history["loss"].append(avg_loss)

            # Metrics
            if self._metrics:
                preds_all = self.predict(X_train[:min(200, n_samples)], shots=0)
                for m in self._metrics:
                    val = m(preds_all, y_train[:min(200, n_samples)])
                    history[m.name].append(float(val))

            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = np.atleast_2d(X_val)
                val_preds = self.predict(X_val, shots=0)
                val_loss = self._loss_fn(val_preds, y_val)
                history["val_loss"].append(float(val_loss))
                for m in self._metrics:
                    val_m = m(val_preds, y_val)
                    history[f"val_{m.name}"].append(float(val_m))

            dt = time.time() - t0
            if verbose >= 1:
                msg = f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}"
                if "val_loss" in history and history["val_loss"]:
                    msg += f" — val_loss: {history['val_loss'][-1]:.4f}"
                for m in self._metrics:
                    if m.name in history and history[m.name]:
                        msg += f" — {m.name}: {history[m.name][-1]:.4f}"
                msg += f" — {dt:.1f}s"
                print(msg)

            # Callbacks: on_epoch_end
            stop = False
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    result = cb.on_epoch_end(epoch, history, self)
                    if result is False:
                        stop = True
            if stop:
                if verbose:
                    print("Early stopping triggered.")
                break

        self._history = history
        return history

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: NDArray[np.floating],
        y_test: NDArray,
        shots: int = 0,
    ) -> Tuple[float, ...]:
        """
        Evaluate the model on test data.

        Returns
        -------
        tuple
            (loss, metric_1, metric_2, …)
        """
        X_test = np.atleast_2d(X_test)
        preds = self.predict(X_test, shots=shots)
        loss_val = float(self._loss_fn(preds, y_test))
        results = [loss_val]
        for m in self._metrics:
            results.append(float(m(preds, y_test)))
        return tuple(results)

    # ------------------------------------------------------------------
    # Hardware backend
    # ------------------------------------------------------------------
    def set_backend(self, backend: str = "ibm_quantum", token: Optional[str] = None) -> None:
        """Configure hardware backend for prediction."""
        self._hw_token = token
        self._hw_backend = backend if backend != "ibm_quantum" else None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        """Save model to file."""
        data = {
            "layers": self._layers,
            "optimizer": self._optimizer,
            "loss_fn": self._loss_fn,
            "metrics": self._metrics,
            "history": self._history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "Sequential":
        """Load model from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        model = cls(data["layers"])
        model._optimizer = data.get("optimizer")
        model._loss_fn = data.get("loss_fn")
        model._metrics = data.get("metrics", [])
        model._history = data.get("history", {})
        model._compiled = model._optimizer is not None
        return model

    def __repr__(self) -> str:
        return f"Sequential(layers={len(self._layers)}, params={self.num_parameters})"
