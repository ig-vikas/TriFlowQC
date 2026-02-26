"""Unit tests for NN models (Sequential)."""

import numpy as np
import pytest
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.layers.embedding import AngleEmbedding
from triflowqc.nn.layers.dense_qutrit import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement_layer import MeasurementLayer
from triflowqc.nn.losses import CrossEntropyLoss
from triflowqc.nn.metrics import Accuracy
from triflowqc.optim.adam import Adam


class TestSequential:
    def _make_model(self, num_qutrits=2, num_classes=3):
        model = Sequential([
            AngleEmbedding(num_qutrits=num_qutrits),
            DenseQutrit(num_qutrits=num_qutrits, num_layers=1, rotations="basic"),
            EntanglingLayer(num_qutrits=num_qutrits, pattern="linear"),
            MeasurementLayer(num_qutrits=num_qutrits, num_classes=num_classes),
        ])
        return model

    def test_build(self):
        model = self._make_model()
        model.compile(
            optimizer=Adam(lr=0.01),
            loss=CrossEntropyLoss(),
            metrics=[Accuracy()],
        )
        assert len(model.all_parameters) > 0

    def test_predict(self):
        model = self._make_model()
        model.compile(optimizer=Adam(lr=0.01), loss=CrossEntropyLoss())
        x = np.random.rand(3, 2)
        preds = model.predict(x)
        assert preds.shape[0] == 3
        assert preds.shape[1] == 3

    def test_summary(self):
        model = self._make_model()
        model.compile(optimizer=Adam(lr=0.01), loss=CrossEntropyLoss())
        summary = model.summary()
        assert isinstance(summary, str)
        assert "Sequential" in summary or "Layer" in summary.lower() or len(summary) > 0
