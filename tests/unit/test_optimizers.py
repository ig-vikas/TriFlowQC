"""Unit tests for optimizers."""

import numpy as np
import pytest
from triflowqc.optim.sgd import SGD
from triflowqc.optim.adam import Adam


class TestSGD:
    def test_step(self):
        value = 1.0
        grad = 0.5
        opt = SGD(lr=0.1)
        new_values = opt.step([value], np.array([grad]))
        assert new_values[0] < 1.0  # Should decrease

    def test_momentum(self):
        value = 1.0
        opt = SGD(lr=0.1, momentum=0.9)
        for _ in range(5):
            new_values = opt.step([value], np.array([1.0]))
            value = new_values[0]
        assert value < 1.0


class TestAdam:
    def test_step(self):
        value = 1.0
        grad = 0.5
        opt = Adam(lr=0.01)
        new_values = opt.step([value], np.array([grad]))
        assert new_values[0] != 1.0

    def test_convergence(self):
        """Adam should move towards minimum with constant gradient."""
        value = 5.0
        opt = Adam(lr=0.1)
        for _ in range(100):
            grad = 2 * value  # gradient of x²
            new_values = opt.step([value], np.array([grad]))
            value = new_values[0]
        assert abs(value) < 1.0  # Should approach 0
