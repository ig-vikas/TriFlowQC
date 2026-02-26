"""Unit tests for circuit module."""

import numpy as np
import pytest
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.circuit.measurement import MeasurementResult


class TestQutritCircuit:
    def test_init(self):
        qc = QutritCircuit(3)
        assert qc.num_qutrits == 3
        assert len(qc.instructions) == 0

    def test_add_gate(self):
        qc = QutritCircuit(2)
        qc.h3(0)
        assert len(qc.instructions) == 1

    def test_multiple_gates(self):
        qc = QutritCircuit(2)
        qc.x3(0)
        qc.h3(1)
        qc.csum(0, 1)
        assert len(qc.instructions) == 3

    def test_parametric_gate(self):
        qc = QutritCircuit(2)
        from triflowqc.core.variable import Variable
        theta = Variable(1.5)
        qc.rx3(0, theta)
        assert qc.num_parameters >= 1

    def test_depth(self):
        qc = QutritCircuit(2)
        qc.x3(0)
        qc.x3(1)
        # Parallel gates → depth 1
        assert qc.depth >= 1

    def test_draw(self):
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        text = qc.draw()
        assert isinstance(text, str)
        assert len(text) > 0


class TestMeasurementResult:
    def test_basic(self):
        result = MeasurementResult(
            counts={"00": 50, "01": 50},
            probabilities=np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]),
            statevector=None,
            shots=100,
            num_qutrits=2,
        )
        assert result.shots == 100
        assert result.most_probable() == "00"

    def test_expectation(self):
        probs = np.zeros(3)
        probs[0] = 1.0
        result = MeasurementResult(
            counts={"0": 100},
            probabilities=probs,
            statevector=None,
            shots=100,
            num_qutrits=1,
        )
        obs = np.diag([1.0, 0.0, -1.0])
        exp = result.expectation(obs)
        assert np.isclose(exp, 1.0)
