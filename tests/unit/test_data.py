"""Unit tests for data module."""

import numpy as np
import pytest
from triflowqc.data.loader import DataLoader
from triflowqc.data.encoder import AngleEncoder, AmplitudeEncoder, BasisEncoder
from triflowqc.data.preprocessor import Preprocessor


class TestDataLoader:
    def test_creation(self):
        x = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100).astype(float)
        dl = DataLoader(x, y, batch_size=16)
        assert dl.n_train == 100
        assert dl.n_batches == 7  # ceil(100/16)

    def test_batches(self):
        x = np.random.rand(50, 4)
        y = np.random.randint(0, 3, 50).astype(float)
        dl = DataLoader(x, y, batch_size=10, shuffle=False)
        batches = list(dl.train_batches())
        assert len(batches) == 5
        assert batches[0].x.shape[0] == 10

    def test_validation_split(self):
        x = np.random.rand(100, 4)
        y = np.random.randint(0, 3, 100).astype(float)
        dl = DataLoader(x, y, validation_split=0.2)
        assert dl.n_val >= 15  # ~20
        assert dl.n_train + dl.n_val == 100


class TestAngleEncoder:
    def test_encode(self):
        enc = AngleEncoder(n_qutrits=2, gm_indices=[1, 2, 3])
        angles = enc.encode(np.array([0.5, 1.0, 0.3, 0.7, 0.2, 0.1]))
        assert len(angles) == 6

    def test_n_parameters(self):
        enc = AngleEncoder(n_qutrits=3)
        assert enc.n_parameters == 9


class TestAmplitudeEncoder:
    def test_encode_normalised(self):
        enc = AmplitudeEncoder(n_qutrits=1)
        state = enc.encode(np.array([1.0, 2.0, 3.0]))
        assert np.isclose(np.linalg.norm(state), 1.0)

    def test_encode_padding(self):
        enc = AmplitudeEncoder(n_qutrits=2)
        state = enc.encode(np.array([1.0, 2.0]))
        assert len(state) == 9
        assert np.isclose(np.linalg.norm(state), 1.0)


class TestBasisEncoder:
    def test_encode(self):
        enc = BasisEncoder(n_qutrits=1)
        state = enc.encode(2)
        assert np.isclose(state[2], 1.0)
        assert np.isclose(np.linalg.norm(state), 1.0)


class TestPreprocessor:
    def test_minmax(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        pp = Preprocessor(method="minmax", feature_range=(0, np.pi))
        x_scaled = pp.fit_transform(x)
        assert x_scaled.min() >= 0
        assert x_scaled.max() <= np.pi + 1e-10

    def test_standard(self):
        x = np.random.rand(100, 4)
        pp = Preprocessor(method="standard")
        x_scaled = pp.fit_transform(x)
        assert np.allclose(x_scaled.mean(axis=0), 0.0, atol=1e-10)
        assert np.allclose(x_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_inverse(self):
        x = np.random.rand(50, 3)
        pp = Preprocessor(method="minmax")
        x_scaled = pp.fit_transform(x)
        x_back = pp.inverse_transform(x_scaled)
        assert np.allclose(x, x_back, atol=1e-10)

    def test_one_hot(self):
        labels = np.array([0, 1, 2, 0])
        oh = Preprocessor.one_hot(labels, num_classes=3)
        assert oh.shape == (4, 3)
        assert np.allclose(oh[0], [1, 0, 0])
        assert np.allclose(oh[2], [0, 0, 1])

    def test_pca(self):
        x = np.random.rand(100, 10)
        x_red, components = Preprocessor.pca_reduce(x, n_components=3)
        assert x_red.shape == (100, 3)
        assert components.shape == (3, 10)
