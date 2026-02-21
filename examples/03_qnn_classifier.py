"""
Example 3: Qutrit Neural Network Classifier

Demonstrates:
  - Building a QNN for ternary classification
  - Data preprocessing and encoding
  - Training with SGD and Adam
  - Evaluating accuracy
  - Plotting loss curve
"""

import numpy as np
from triflowqc.nn.layers.embedding import AngleEmbedding
from triflowqc.nn.layers.dense import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement import MeasurementLayer
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.losses import MSELoss, CrossEntropyLoss
from triflowqc.nn.metrics import Accuracy
from triflowqc.optim.adam import Adam
from triflowqc.optim.sgd import SGD
from triflowqc.data.loader import DataLoader
from triflowqc.data.encoder import AngleEncoder
from triflowqc.data.preprocessor import Preprocessor

print("=" * 50)
print("  Example 3: Qutrit Neural Network Classifier")
print("=" * 50)

# ── Generate Synthetic Data ──────────────────────────
print("\n1. Generating Synthetic Data")
np.random.seed(42)
n_samples = 30
n_features = 3
n_classes = 3

# Simple cluster data
X = np.zeros((n_samples, n_features))
Y = np.zeros((n_samples, n_classes))
for i in range(n_samples):
    cls = i % n_classes
    X[i] = np.random.randn(n_features) * 0.3 + cls
    Y[i, cls] = 1.0

print(f"  Samples: {n_samples}")
print(f"  Features: {n_features}")
print(f"  Classes: {n_classes}")
print(f"  X shape: {X.shape}")
print(f"  Y shape: {Y.shape}")

# ── Preprocess ───────────────────────────────────────
print("\n2. Preprocessing")
preprocessor = Preprocessor(method='standard')
X_scaled = preprocessor.fit_transform(X)
print(f"  Scaled mean: {X_scaled.mean(axis=0).round(4)}")
print(f"  Scaled std:  {X_scaled.std(axis=0).round(4)}")

# ── Encode ───────────────────────────────────────────
print("\n3. Encoding for Quantum Circuit")
encoder = AngleEncoder()
X_encoded = np.array([encoder.encode(x) for x in X_scaled])
print(f"  Encoded range: [{X_encoded.min():.4f}, {X_encoded.max():.4f}]")

# ── Train/Test Split ─────────────────────────────────
split = int(0.8 * n_samples)
X_train, X_test = X_encoded[:split], X_encoded[split:]
Y_train, Y_test = Y[:split], Y[split:]
print(f"\n4. Train/Test Split")
print(f"  Training: {len(X_train)} samples")
print(f"  Testing:  {len(X_test)} samples")

# ── Build Model ──────────────────────────────────────
print("\n5. Building QNN Model")
n_qutrits = n_features  # 1 qutrit per feature

model = Sequential([
    AngleEmbedding(n_qutrits=n_qutrits),
    DenseQutrit(n_qutrits=n_qutrits, n_layers=2, rotation_type='basic'),
    EntanglingLayer(n_qutrits=n_qutrits, pattern='linear'),
    DenseQutrit(n_qutrits=n_qutrits, n_layers=1, rotation_type='basic'),
    MeasurementLayer(n_qutrits=n_qutrits, n_classes=n_classes),
])

model.summary()
print(f"  Total parameters: {len(model.parameters())}")

# ── Compile and Train ────────────────────────────────
print("\n6. Training")
optimizer = Adam(learning_rate=0.05)
loss_fn = MSELoss()
model.compile(optimizer=optimizer, loss=loss_fn)

history = model.fit(
    X_train, Y_train,
    epochs=10,
    verbose=True,
)

# ── Evaluate ─────────────────────────────────────────
print("\n7. Evaluation")
correct = 0
total = len(X_test)
for x, y_true in zip(X_test, Y_test):
    y_pred = model.predict(x)
    pred_class = np.argmax(y_pred)
    true_class = np.argmax(y_true)
    if pred_class == true_class:
        correct += 1
    print(f"  True: {true_class}, Pred: {pred_class}, "
          f"Probs: [{', '.join(f'{p:.3f}' for p in y_pred)}]")

accuracy = correct / total if total > 0 else 0
print(f"\n  Test Accuracy: {accuracy:.2%} ({correct}/{total})")

# ── Loss History ─────────────────────────────────────
print("\n8. Training History")
print(f"  Epoch | Loss")
print(f"  ------+----------")
for epoch, loss in enumerate(history['loss'], 1):
    bar = '█' * int(loss * 20)
    print(f"  {epoch:5d} | {loss:.6f} {bar}")

print("\n  Final loss: {:.6f}".format(history['loss'][-1]))
if len(history['loss']) > 1:
    improvement = history['loss'][0] - history['loss'][-1]
    print(f"  Total improvement: {improvement:.6f}")

print("\n" + "=" * 50)
print("  Example complete!")
print("=" * 50)
