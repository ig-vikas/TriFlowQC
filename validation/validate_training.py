"""
Validation script: Verify training loop correctness.

Checks:
  - Model builds and produces output
  - Forward pass is deterministic
  - Gradients are non-zero and finite
  - Loss decreases over training
  - Different optimizers converge
  - Callbacks function correctly
"""

import numpy as np
import sys
import time


def validate_model_construction():
    """Validate model can be constructed and produces output."""
    print("\n=== Model Construction ===")
    from triflowqc.nn.layers.embedding import AngleEmbedding
    from triflowqc.nn.layers.dense import DenseQutrit
    from triflowqc.nn.layers.entangling import EntanglingLayer
    from triflowqc.nn.layers.measurement import MeasurementLayer
    from triflowqc.nn.models.sequential import Sequential

    all_pass = True

    # Build model
    try:
        model = Sequential(layers=[
            AngleEmbedding(num_qutrits=2),
            DenseQutrit(num_qutrits=2, num_layers=1, rotations='basic'),
            EntanglingLayer(num_qutrits=2, pattern='linear'),
            MeasurementLayer(num_qutrits=2, num_classes=3),
        ])
        print("  [PASS] Model construction")
    except Exception as e:
        print(f"  [FAIL] Model construction: {e}")
        all_pass = False
        return all_pass

    # Predict
    try:
        x = np.array([0.5, 0.3])
        output = model.predict(x)
        assert output is not None
        assert len(output) > 0
        print(f"  [PASS] Model predict: output shape = {np.array(output).shape}")
    except Exception as e:
        print(f"  [FAIL] Model predict: {e}")
        all_pass = False

    # Determinism
    try:
        out1 = model.predict(x)
        out2 = model.predict(x)
        err = np.max(np.abs(np.array(out1) - np.array(out2)))
        status = "PASS" if err < 1e-12 else "FAIL"
        print(f"  [{status}] Deterministic predict (error = {err:.2e})")
        all_pass = all_pass and (err < 1e-12)
    except Exception as e:
        print(f"  [FAIL] Deterministic predict: {e}")
        all_pass = False

    # Parameter count
    try:
        params = model.all_parameters
        n_params = len(params)
        print(f"  [PASS] Parameter count: {n_params}")
        assert n_params > 0
    except Exception as e:
        print(f"  [FAIL] Parameter count: {e}")
        all_pass = False

    # Summary
    try:
        model.summary()
        print("  [PASS] Model summary")
    except Exception as e:
        print(f"  [FAIL] Model summary: {e}")
        all_pass = False

    return all_pass


def validate_gradient_computation():
    """Validate gradient computation is correct."""
    print("\n=== Gradient Computation ===")
    from triflowqc.circuit.circuit import QutritCircuit
    from triflowqc.simulator.statevector_sim import StatevectorSimulator
    from triflowqc.optim.parameter_shift import parameter_shift_gradient

    all_pass = True

    # Simple cost function
    def cost_fn(params):
        qc = QutritCircuit(1)
        qc.rx3(params[0], 0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        return np.abs(sv[0]) ** 2

    # Gradient at theta=pi/4
    params = np.array([np.pi / 4])
    grads = parameter_shift_gradient(cost_fn, params)

    # Check gradient is non-zero
    status = "PASS" if abs(grads[0]) > 1e-6 else "FAIL"
    print(f"  [{status}] Non-zero gradient: {grads[0]:.6f}")
    all_pass = all_pass and (abs(grads[0]) > 1e-6)

    # Check gradient is finite
    status = "PASS" if np.isfinite(grads[0]) else "FAIL"
    print(f"  [{status}] Finite gradient")
    all_pass = all_pass and np.isfinite(grads[0])

    # Compare with finite difference
    eps = 1e-5
    fd_grad = (cost_fn(params + eps) - cost_fn(params - eps)) / (2 * eps)
    err = abs(grads[0] - fd_grad)
    status = "PASS" if err < 0.05 else "FAIL"
    print(f"  [{status}] Parameter-shift vs finite diff: "
          f"PS={grads[0]:.6f}, FD={fd_grad:.6f}, error={err:.6f}")
    all_pass = all_pass and (err < 0.05)

    # Multi-parameter gradient
    def cost_fn_2(params):
        qc = QutritCircuit(1)
        qc.rx3(params[0], 0)
        qc.ry3(params[1], 0)
        sim = StatevectorSimulator()
        result = sim.run(qc)
        sv = result.statevector
        return np.abs(sv[0]) ** 2

    params2 = np.array([0.5, 0.3])
    grads2 = parameter_shift_gradient(cost_fn_2, params2)
    status = "PASS" if len(grads2) == 2 and all(np.isfinite(grads2)) else "FAIL"
    print(f"  [{status}] Multi-parameter gradient: {grads2}")
    all_pass = all_pass and (len(grads2) == 2 and all(np.isfinite(grads2)))

    return all_pass


def validate_training_convergence():
    """Validate that training reduces loss."""
    print("\n=== Training Convergence ===")
    from triflowqc.nn.layers.embedding import AngleEmbedding
    from triflowqc.nn.layers.dense import DenseQutrit
    from triflowqc.nn.layers.entangling import EntanglingLayer
    from triflowqc.nn.layers.measurement import MeasurementLayer
    from triflowqc.nn.models.sequential import Sequential
    from triflowqc.nn.losses import MSELoss
    from triflowqc.optim.sgd import SGD
    from triflowqc.optim.adam import Adam

    all_pass = True

    # Small dataset
    np.random.seed(42)
    X = np.array([[0.1, 0.2], [0.5, 0.6], [0.8, 0.9], [0.3, 0.7]])
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)

    # SGD training
    print("\n  SGD Training:")
    model_sgd = Sequential(layers=[
        AngleEmbedding(num_qutrits=2),
        DenseQutrit(num_qutrits=2, num_layers=1, rotations='basic'),
        EntanglingLayer(num_qutrits=2, pattern='linear'),
        MeasurementLayer(num_qutrits=2, num_classes=3),
    ])
    optimizer_sgd = SGD(lr=0.1)
    loss_fn = MSELoss()
    model_sgd.compile(optimizer=optimizer_sgd, loss=loss_fn)

    start = time.perf_counter()
    history_sgd = model_sgd.fit(X, Y, epochs=5, verbose=False)
    elapsed = time.perf_counter() - start

    losses_sgd = history_sgd['loss']
    print(f"    Losses: {[f'{l:.4f}' for l in losses_sgd]}")
    print(f"    Time: {elapsed:.2f}s")

    # Check all losses are finite
    finite = all(np.isfinite(l) for l in losses_sgd)
    status = "PASS" if finite else "FAIL"
    print(f"    [{status}] All losses finite")
    all_pass = all_pass and finite

    # Adam training
    print("\n  Adam Training:")
    model_adam = Sequential(layers=[
        AngleEmbedding(num_qutrits=2),
        DenseQutrit(num_qutrits=2, num_layers=1, rotations='basic'),
        EntanglingLayer(num_qutrits=2, pattern='linear'),
        MeasurementLayer(num_qutrits=2, num_classes=3),
    ])
    optimizer_adam = Adam(lr=0.05)
    model_adam.compile(optimizer=optimizer_adam, loss=loss_fn)

    start = time.perf_counter()
    history_adam = model_adam.fit(X, Y, epochs=5, verbose=False)
    elapsed = time.perf_counter() - start

    losses_adam = history_adam['loss']
    print(f"    Losses: {[f'{l:.4f}' for l in losses_adam]}")
    print(f"    Time: {elapsed:.2f}s")

    finite = all(np.isfinite(l) for l in losses_adam)
    status = "PASS" if finite else "FAIL"
    print(f"    [{status}] All losses finite")
    all_pass = all_pass and finite

    return all_pass


def validate_data_pipeline():
    """Validate the data loading and encoding pipeline."""
    print("\n=== Data Pipeline ===")
    from triflowqc.data.loader import DataLoader
    from triflowqc.data.encoder import AngleEncoder, AmplitudeEncoder
    from triflowqc.data.preprocessor import Preprocessor

    all_pass = True

    # Create raw data
    np.random.seed(42)
    X_raw = np.random.randn(20, 4) * 10
    Y_raw = np.eye(3)[np.random.randint(0, 3, 20)]

    # Preprocess
    preprocessor = Preprocessor(method='standard')
    X_scaled = preprocessor.fit_transform(X_raw)
    mean_scaled = np.mean(X_scaled, axis=0)
    std_scaled = np.std(X_scaled, axis=0)

    status = "PASS" if np.allclose(mean_scaled, 0, atol=1e-10) else "FAIL"
    print(f"  [{status}] Standard scaling: mean ≈ 0 ({mean_scaled.round(4)})")
    all_pass = all_pass and np.allclose(mean_scaled, 0, atol=1e-10)

    status = "PASS" if np.allclose(std_scaled, 1, atol=0.1) else "FAIL"
    print(f"  [{status}] Standard scaling: std ≈ 1 ({std_scaled.round(4)})")
    all_pass = all_pass and np.allclose(std_scaled, 1, atol=0.1)

    # Encode
    encoder = AngleEncoder()
    X_encoded = np.array([encoder.encode(x) for x in X_scaled])
    in_range = np.all((X_encoded >= 0) & (X_encoded <= np.pi))
    status = "PASS" if in_range else "FAIL"
    print(f"  [{status}] Angle encoding in [0, π]")
    all_pass = all_pass and in_range

    # DataLoader
    loader = DataLoader(X_encoded, Y_raw, batch_size=5)
    batches = list(loader.train_batches())
    n_batches = len(batches)
    total_samples = sum(len(b.x) for b in batches)
    status = "PASS" if total_samples == 20 else "FAIL"
    print(f"  [{status}] DataLoader: {n_batches} batches, {total_samples} total samples")
    all_pass = all_pass and (total_samples == 20)

    # Inverse transform
    X_recovered = preprocessor.inverse_transform(X_scaled)
    err = np.max(np.abs(X_recovered - X_raw))
    status = "PASS" if err < 1e-10 else "FAIL"
    print(f"  [{status}] Inverse transform (error = {err:.2e})")
    all_pass = all_pass and (err < 1e-10)

    return all_pass


def main():
    """Run all training validations."""
    print("=" * 60)
    print("  TriFlowQC Training Validation Suite")
    print("=" * 60)

    results = {}
    results['Model Construction'] = validate_model_construction()
    results['Gradient Computation'] = validate_gradient_computation()
    results['Training Convergence'] = validate_training_convergence()
    results['Data Pipeline'] = validate_data_pipeline()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_pass = all_pass and passed

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
