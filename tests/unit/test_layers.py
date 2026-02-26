"""Unit tests for NN layers."""

import numpy as np
import pytest
from triflowqc.nn.layers.embedding import (
    AngleEmbedding,
    AmplitudeEmbedding,
    BasisEmbedding,
)
from triflowqc.nn.layers.dense_qutrit import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement_layer import MeasurementLayer


class TestAngleEmbedding:
    def test_construction(self):
        layer = AngleEmbedding(num_qutrits=2)
        assert layer.num_qutrits == 2

    def test_build_circuit_returns_instructions(self):
        layer = AngleEmbedding(num_qutrits=2)
        from triflowqc.circuit.circuit import QutritCircuit
        qc = QutritCircuit(2)
        features = np.array([0.5, 1.0])
        layer.build_circuit(qc, x=features)
        assert len(qc.instructions) > 0


class TestDenseQutrit:
    def test_construction(self):
        layer = DenseQutrit(num_qutrits=2, num_layers=1, rotations="basic")
        assert len(layer.parameters) > 0

    def test_parameters_are_floats(self):
        layer = DenseQutrit(num_qutrits=2, num_layers=1, rotations="basic")
        params = layer.parameters
        assert all(isinstance(p, float) for p in params)


class TestEntanglingLayer:
    def test_construction(self):
        layer = EntanglingLayer(num_qutrits=3, pattern="linear")
        assert layer.num_qutrits == 3

    def test_patterns(self):
        for pattern in ["linear", "ring", "full"]:
            layer = EntanglingLayer(num_qutrits=3, pattern=pattern)
            from triflowqc.circuit.circuit import QutritCircuit
            qc = QutritCircuit(3)
            layer.build_circuit(qc)
            assert len(qc.instructions) > 0


class TestMeasurementLayer:
    def test_construction(self):
        layer = MeasurementLayer(num_qutrits=2, num_classes=3)
        assert layer.num_classes == 3
