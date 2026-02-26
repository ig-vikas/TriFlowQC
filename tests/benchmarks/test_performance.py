"""
Benchmark tests for TriFlowQC performance.

These tests are marked with @pytest.mark.benchmark and are not run by default.
Run with: pytest tests/benchmarks/ -v
"""

import numpy as np
import pytest
import time

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator
from triflowqc.simulator.density_sim import DensityMatrixSimulator
from triflowqc.simulator.noise_model import NoiseModel, depolarising_channel
from triflowqc.simulator.noisy_simulator import NoisySimulator
from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.nn.layers.embedding import AngleEmbedding
from triflowqc.nn.layers.dense import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement import MeasurementLayer
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.losses import MSELoss
from triflowqc.optim.sgd import SGD
from triflowqc.optim.parameter_shift import parameter_shift_gradient
from triflowqc.math.su3 import su3_generators, su3_random_element
from triflowqc.math.algebra import tensor_product, partial_trace_mixed


@pytest.mark.benchmark
class TestSimulatorBenchmarks:
    """Benchmark statevector simulator scaling."""

    @pytest.mark.parametrize("n_qutrits", [1, 2, 3, 4, 5])
    def test_statevector_scaling(self, n_qutrits):
        """Measure statevector simulator time vs number of qutrits."""
        qc = QutritCircuit(n_qutrits)
        for i in range(n_qutrits):
            qc.h3(i)
        for i in range(n_qutrits - 1):
            qc.csum(i, i + 1)

        sim = StatevectorSimulator()
        start = time.perf_counter()
        result = sim.run(qc)
        elapsed = time.perf_counter() - start

        assert result.statevector is not None
        print(f"\n  {n_qutrits} qutrits (dim={3**n_qutrits}): {elapsed:.4f}s")

    @pytest.mark.parametrize("n_qutrits", [1, 2, 3, 4])
    def test_density_matrix_scaling(self, n_qutrits):
        """Measure density matrix simulator time vs number of qutrits."""
        qc = QutritCircuit(n_qutrits)
        for i in range(n_qutrits):
            qc.h3(i)
        for i in range(n_qutrits - 1):
            qc.csum(i, i + 1)

        sim = DensityMatrixSimulator()
        start = time.perf_counter()
        result = sim.run(qc)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n  {n_qutrits} qutrits (dim={3**n_qutrits}): {elapsed:.4f}s")

    @pytest.mark.parametrize("n_qutrits", [1, 2, 3])
    def test_noisy_simulator_scaling(self, n_qutrits):
        """Measure noisy simulator scaling."""
        qc = QutritCircuit(n_qutrits)
        for i in range(n_qutrits):
            qc.h3(i)
        for i in range(n_qutrits - 1):
            qc.csum(i, i + 1)

        noise = NoiseModel()
        noise.add_gate_error('H3', depolarising_channel(0.01))
        if n_qutrits > 1:
            noise.add_gate_error('CSUM', depolarising_channel(0.02))

        sim = NoisySimulator(noise)
        start = time.perf_counter()
        result = sim.run(qc, shots=1000)
        elapsed = time.perf_counter() - start

        assert result is not None
        print(f"\n  {n_qutrits} qutrits noisy: {elapsed:.4f}s")


@pytest.mark.benchmark
class TestCircuitBenchmarks:
    """Benchmark circuit construction and analysis."""

    @pytest.mark.parametrize("n_gates", [10, 50, 100, 500])
    def test_circuit_construction_speed(self, n_gates):
        """Measure time to construct circuits of various sizes."""
        start = time.perf_counter()
        qc = QutritCircuit(3)
        for i in range(n_gates):
            qc.rx3(0.1 * i, i % 3)
            if i % 5 == 0 and i > 0:
                qc.csum(0, 1)
        elapsed = time.perf_counter() - start
        print(f"\n  {n_gates} gates: {elapsed:.4f}s")
        assert elapsed < 10.0  # Should be fast

    def test_circuit_depth_computation(self):
        """Measure depth computation time for a large circuit."""
        qc = QutritCircuit(5)
        for _ in range(100):
            for i in range(5):
                qc.rx3(0.1, i)
            for i in range(4):
                qc.csum(i, i + 1)

        start = time.perf_counter()
        depth = qc.depth
        elapsed = time.perf_counter() - start
        print(f"\n  Depth computation for 500 gates: {elapsed:.4f}s, depth={depth}")
        assert depth > 0


@pytest.mark.benchmark
class TestCompilerBenchmarks:
    """Benchmark compiler performance."""

    @pytest.mark.parametrize("n_qutrits", [1, 2, 3])
    def test_compilation_speed(self, n_qutrits):
        """Measure compilation time vs circuit size."""
        qc = QutritCircuit(n_qutrits)
        for i in range(n_qutrits):
            qc.h3(i)
            qc.rx3(0.5, i)
        for i in range(n_qutrits - 1):
            qc.csum(i, i + 1)

        compiler = QutritToQubitCompiler()
        start = time.perf_counter()
        qubit_circuit = compiler.compile(qc)
        elapsed = time.perf_counter() - start

        print(f"\n  {n_qutrits} qutrits: {elapsed:.4f}s, "
              f"qubit gates: {len(qubit_circuit)}")
        assert qubit_circuit is not None


@pytest.mark.benchmark
class TestGradientBenchmarks:
    """Benchmark gradient computation."""

    @pytest.mark.parametrize("n_params", [1, 2, 4, 8])
    def test_parameter_shift_gradient_speed(self, n_params):
        """Measure gradient computation time vs number of parameters."""
        def circuit_fn(params):
            qc = QutritCircuit(max(1, n_params // 2))
            for i, p in enumerate(params):
                qc.rx3(p, i % max(1, n_params // 2))
            sim = StatevectorSimulator()
            result = sim.run(qc)
            sv = result.statevector
            return np.abs(sv[0]) ** 2

        params = np.random.randn(n_params) * 0.5
        start = time.perf_counter()
        grads = parameter_shift_gradient(circuit_fn, params)
        elapsed = time.perf_counter() - start

        print(f"\n  {n_params} params: {elapsed:.4f}s "
              f"({elapsed/n_params:.4f}s per param)")
        assert len(grads) == n_params


@pytest.mark.benchmark
class TestTrainingBenchmarks:
    """Benchmark training loop performance."""

    def test_single_epoch_time(self):
        """Measure time for a single training epoch."""
        np.random.seed(42)
        model = Sequential(layers=[
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1),
            EntanglingLayer(num_qutrits=2, pattern='linear'),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])
        optimizer = SGD(lr=0.1)
        loss_fn = MSELoss()
        model.compile(optimizer=optimizer, loss=loss_fn)

        X = np.random.randn(4, 2)
        Y = np.eye(3)[np.random.randint(0, 3, 4)]

        start = time.perf_counter()
        history = model.fit(X, Y, epochs=1, verbose=False)
        elapsed = time.perf_counter() - start

        print(f"\n  1 epoch, 4 samples: {elapsed:.4f}s")
        assert len(history['loss']) == 1


@pytest.mark.benchmark
class TestMathBenchmarks:
    """Benchmark mathematical operations."""

    @pytest.mark.parametrize("n_qutrits", [1, 2, 3, 4])
    def test_tensor_product_speed(self, n_qutrits):
        """Measure tensor product computation time."""
        matrices = [np.random.randn(3, 3) + 1j * np.random.randn(3, 3)
                     for _ in range(n_qutrits)]

        start = time.perf_counter()
        result = matrices[0]
        for m in matrices[1:]:
            result = tensor_product(result, m)
        elapsed = time.perf_counter() - start

        expected_dim = 3 ** n_qutrits
        assert result.shape == (expected_dim, expected_dim)
        print(f"\n  {n_qutrits} qutrits (dim={expected_dim}): {elapsed:.6f}s")

    @pytest.mark.parametrize("dim", [3, 9, 27])
    def test_partial_trace_speed(self, dim):
        """Measure partial trace computation time."""
        # Random density matrix
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        n_qutrits = int(np.round(np.log(dim) / np.log(3)))
        if n_qutrits < 2:
            pytest.skip("Need at least 2 qutrits for partial trace")

        start = time.perf_counter()
        rho_reduced = partial_trace_mixed(rho, [0], [3] * n_qutrits)
        elapsed = time.perf_counter() - start

        print(f"\n  Trace over 1 of {n_qutrits} qutrits (dim={dim}): {elapsed:.6f}s")

    def test_su3_random_generation_speed(self):
        """Measure SU(3) random element generation time."""
        n_samples = 100
        start = time.perf_counter()
        for _ in range(n_samples):
            U = su3_random_element()
        elapsed = time.perf_counter() - start

        print(f"\n  {n_samples} random SU(3) elements: {elapsed:.4f}s "
              f"({elapsed/n_samples*1000:.2f}ms each)")
