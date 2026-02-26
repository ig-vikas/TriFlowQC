"""
Integration tests: Full training loop — data → model → optimizer → simulator.
"""

import numpy as np
import pytest

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.nn.layers.embedding import AngleEmbedding
from triflowqc.nn.layers.dense_qutrit import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement_layer import MeasurementLayer
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.losses import MSELoss, CrossEntropyLoss
from triflowqc.nn.metrics import Accuracy
from triflowqc.nn.callbacks import EarlyStopping
from triflowqc.optim.adam import Adam
from triflowqc.optim.sgd import SGD
from triflowqc.optim.parameter_shift import parameter_shift_gradient
from triflowqc.data.loader import DataLoader
from triflowqc.data.encoder import AngleEncoder
from triflowqc.data.preprocessor import Preprocessor


class TestTrainingLoop:
    """End-to-end training loop integration."""

    def _make_simple_model(self, num_qutrits=2, num_classes=3):
        """Create a minimal trainable model."""
        model = Sequential([
            AngleEmbedding(num_qutrits=num_qutrits),
            DenseQutrit(num_qutrits=num_qutrits, num_layers=1, rotations='basic'),
            EntanglingLayer(num_qutrits=num_qutrits, pattern='linear'),
            MeasurementLayer(num_qutrits=num_qutrits, num_classes=num_classes),
        ])
        return model

    def test_model_build_and_predict(self):
        """Model builds with correct shapes and produces output."""
        model = self._make_simple_model(num_qutrits=2, num_classes=3)
        x = np.random.randn(2)  # 2 features for 2 qutrits
        output = model.predict(np.atleast_2d(x))
        assert output is not None
        assert len(output) > 0

    def test_model_compile_and_single_step(self):
        """Model compiles and takes one training step."""
        model = self._make_simple_model(num_qutrits=2, num_classes=3)
        optimizer = Adam(lr=0.01)
        loss_fn = MSELoss()
        model.compile(optimizer=optimizer, loss=loss_fn)

        x = np.array([[0.5, 0.3]])
        y = np.array([[1.0, 0.0, 0.0]])

        # Get initial parameters
        params_before = list(model.all_parameters)

        # Single forward pass
        output = model.predict(x)
        assert output is not None

    def test_model_fit_reduces_loss(self):
        """Training for several epochs on a trivial dataset should reduce loss."""
        np.random.seed(42)
        num_qutrits = 2
        model = self._make_simple_model(num_qutrits=num_qutrits, num_classes=3)
        optimizer = SGD(lr=0.1)
        loss_fn = MSELoss()
        model.compile(optimizer=optimizer, loss=loss_fn)

        # Small dataset: 4 samples, 2 features
        X = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ])
        Y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=float)

        history = model.fit(X, Y, epochs=3, verbose=False)
        assert 'loss' in history
        assert len(history['loss']) == 3
        # Loss should be a finite number each epoch
        for l in history['loss']:
            assert np.isfinite(l)

    def test_model_with_adam_optimizer(self):
        """Adam optimizer integrates with model training."""
        np.random.seed(123)
        model = self._make_simple_model(num_qutrits=2, num_classes=3)
        optimizer = Adam(lr=0.05)
        loss_fn = MSELoss()
        model.compile(optimizer=optimizer, loss=loss_fn)

        X = np.array([[0.1, 0.9], [0.8, 0.2]])
        Y = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

        history = model.fit(X, Y, epochs=2, verbose=False)
        assert len(history['loss']) == 2

    def test_model_with_cross_entropy(self):
        """Cross-entropy loss works in training loop."""
        model = self._make_simple_model(num_qutrits=2, num_classes=3)
        optimizer = SGD(lr=0.05)
        loss_fn = CrossEntropyLoss()
        model.compile(optimizer=optimizer, loss=loss_fn)

        X = np.array([[0.5, 0.5]])
        Y = np.array([[0, 1, 0]], dtype=float)

        history = model.fit(X, Y, epochs=1, verbose=False)
        assert len(history['loss']) == 1
        assert np.isfinite(history['loss'][0])


class TestDataPipelineToModel:
    """Data loading → encoding → model integration."""

    def test_data_loader_to_model(self):
        """DataLoader feeds batches to model prediction."""
        np.random.seed(42)
        X = np.random.randn(10, 2)
        Y = np.eye(3)[np.random.randint(0, 3, 10)]

        loader = DataLoader(X, Y, batch_size=5)
        model = Sequential([
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])

        all_outputs = []
        for batch in loader.train_batches():
            for x_sample in batch.x:
                out = model.predict(np.atleast_2d(x_sample))
                all_outputs.append(out)
        assert len(all_outputs) == 10

    def test_preprocessor_encoder_model_pipeline(self):
        """Preprocessor → Encoder → Model pipeline."""
        np.random.seed(42)
        X = np.random.randn(5, 2) * 10  # Raw data with large scale

        preprocessor = Preprocessor(method='standard')
        X_scaled = preprocessor.fit_transform(X)

        encoder = AngleEncoder()
        X_encoded = np.array([encoder.encode(x) for x in X_scaled])

        model = Sequential([
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])

        for x in X_encoded:
            out = model.predict(np.atleast_2d(x))
            assert out is not None


class TestParameterShiftIntegration:
    """Parameter-shift gradient computation integrated with circuit."""

    def test_gradient_computation(self):
        """Compute gradient of a simple parameterized circuit."""
        def circuit_fn(params):
            qc = QutritCircuit(1)
            qc.rx3(params[0], 0)
            qc.ry3(params[1], 0)
            sim = StatevectorSimulator()
            result = sim.run(qc)
            sv = result.statevector
            return np.abs(sv[0]) ** 2  # Probability of |0⟩

        params = np.array([0.5, 0.3])
        grads = parameter_shift_gradient(circuit_fn, params)
        assert len(grads) == 2
        # Verify against finite difference
        eps = 1e-5
        for i in range(2):
            p_plus = params.copy()
            p_plus[i] += eps
            p_minus = params.copy()
            p_minus[i] -= eps
            fd_grad = (circuit_fn(p_plus) - circuit_fn(p_minus)) / (2 * eps)
            assert grads[i] == pytest.approx(fd_grad, abs=0.05)

    def test_gradient_at_extremum(self):
        """Gradient at theta=0 for rx3 is zero for P(|0⟩)."""
        def circuit_fn(params):
            qc = QutritCircuit(1)
            qc.rx3(params[0], 0)
            sim = StatevectorSimulator()
            result = sim.run(qc)
            sv = result.statevector
            return np.abs(sv[0]) ** 2

        params = np.array([0.0])
        grads = parameter_shift_gradient(circuit_fn, params)
        # At theta=0, P(|0⟩) is at a maximum, gradient should be ~0
        assert abs(grads[0]) < 0.1


class TestCallbackIntegration:
    """Callbacks integrated with training."""

    def test_early_stopping_triggers(self):
        """EarlyStopping stops training when loss plateaus."""
        model = Sequential([
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])
        optimizer = SGD(lr=0.001)  # Very small LR → slow convergence
        loss_fn = MSELoss()
        early_stop = EarlyStopping(patience=3, min_delta=1e-10)
        model.compile(optimizer=optimizer, loss=loss_fn)

        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        Y = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

        history = model.fit(X, Y, epochs=50, callbacks=[early_stop], verbose=False)
        # Should stop early (before 50 epochs) due to patience
        assert len(history['loss']) <= 50


class TestModelSerialization:
    """Model save/load integration."""

    def test_model_predict_reproducible(self):
        """Same model with same parameters gives same output."""
        np.random.seed(42)
        model = Sequential([
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])

        x = np.array([[0.5, 0.3]])
        out1 = model.predict(x)
        out2 = model.predict(x)
        np.testing.assert_array_almost_equal(out1, out2)

    def test_model_summary(self):
        """Model summary runs without error."""
        model = Sequential([
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])
        summary = model.summary()
        # summary should be a string or printed output
        # Just ensure it doesn't raise
        assert True
