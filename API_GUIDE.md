# TriFlowQC

**A Complete Python Framework for Qutrit Quantum Neural Networks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TriFlowQC is a full-stack framework for building, training, and running quantum neural networks based on **qutrits** (3-state quantum systems). It provides TensorFlow/Keras-style APIs for quantum machine learning in the SU(3) domain, with a compiler that maps qutrit circuits to real IBM Quantum hardware.

## Key Features

- **Qutrit-native**: Built from the ground up for 3-level quantum systems (SU(3))
- **Keras-like API**: `Sequential` model with `compile()`, `fit()`, `predict()`
- **Full gate library**: Gell-Mann rotations, X3, Z3, H3, CSUM, SWAP3, Toffoli3, parametric gates
- **Multiple simulators**: Statevector, density matrix, and noisy simulation
- **Qutrit-to-qubit compiler**: Run qutrit circuits on real IBM Quantum hardware
- **Leakage handling**: Detect and mitigate |11⟩ forbidden states in qubit encoding
- **Complete math library**: SU(3) algebra, quantum information, entanglement measures
- **Hardware integration**: IBM Quantum via qiskit-ibm-runtime

## Installation

```bash
# Basic installation
pip install -e .

# With IBM Quantum hardware support
pip install -e ".[ibm]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Qutrit Circuit

```python
from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.simulator.statevector_sim import StatevectorSimulator

# Create a Bell state
qc = QutritCircuit(2)
qc.h(0)          # Hadamard on qutrit 0
qc.csum(0, 1)    # Controlled-SUM: |a⟩|b⟩ → |a⟩|(a+b) mod 3⟩

# Simulate
sim = StatevectorSimulator()
result = sim.run(qc, shots=10000)
print(result.probabilities)
# {'00': 0.333, '11': 0.333, '22': 0.333}
```

### 2. Quantum Neural Network

```python
from triflowqc.nn.layers.embedding import AngleEmbedding
from triflowqc.nn.layers.dense import DenseQutrit
from triflowqc.nn.layers.entangling import EntanglingLayer
from triflowqc.nn.layers.measurement import MeasurementLayer
from triflowqc.nn.models.sequential import Sequential
from triflowqc.nn.losses import MSELoss
from triflowqc.optim.adam import Adam

# Build model
model = Sequential([
    AngleEmbedding(n_qutrits=3),
    DenseQutrit(n_qutrits=3, n_layers=2),
    EntanglingLayer(n_qutrits=3, pattern='linear'),
    MeasurementLayer(n_qutrits=3, n_classes=3),
])

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.05), loss=MSELoss())
history = model.fit(X_train, Y_train, epochs=20)
```

### 3. Run on IBM Quantum Hardware

```python
from triflowqc.hardware.hardware_runner import HardwareRunner
from triflowqc.hardware.ibm_backend import IBMBackend

# Connect to IBM Quantum
backend = IBMBackend(token='YOUR_TOKEN', backend_name='ibm_brisbane')
runner = HardwareRunner(backend=backend, n_qutrits=2)

# Run qutrit circuit on real hardware
result = runner.run(qc, shots=4096)
print(result.probabilities)
```

## Architecture

```
triflowqc/
├── core/           # Qutrit, Register, StateVector, DensityMatrix, Variable
├── gates/          # Gell-Mann, single/two/three-qutrit, parametric, custom
├── circuit/        # QutritCircuit, measurement, visualization, transpiler
├── nn/             # Neural network layers, models, losses, metrics, callbacks
│   ├── layers/     # Embedding, Dense, Entangling, Measurement, Pooling, Hybrid
│   └── models/     # Sequential, Functional, Hybrid
├── optim/          # SGD, Adam, parameter-shift, SPSA, natural gradient
├── simulator/      # Statevector, density matrix, noise model, noisy simulator
├── compiler/       # Qutrit-to-qubit compiler, gate decomposer, leakage handler
├── hardware/       # IBM backend, job manager, hardware runner
├── data/           # DataLoader, encoders, preprocessor
├── math/           # SU(3), algebra, information theory, decompositions
└── utils/          # Logging, serialization, visualization, profiling
```

## Qutrit-to-Qubit Encoding

Each qutrit is encoded in 2 physical qubits:

| Qutrit State | Qubit Encoding |
|:---:|:---:|
| \|0⟩ | \|00⟩ |
| \|1⟩ | \|01⟩ |
| \|2⟩ | \|10⟩ |
| — | \|11⟩ **FORBIDDEN** |

The compiler automatically:
1. Embeds qutrit gates into the 4D qubit Hilbert space
2. Decomposes into hardware basis gates via Qiskit
3. Detects and handles leakage into the |11⟩ subspace
4. Maps measurement results back to the trit basis

## Mathematical Foundation

- **SU(3) generators**: 8 Gell-Mann matrices λ₁–λ₈
- **Parametric gates**: R_k(θ) = exp(-iθλ_k/2)
- **Parameter-shift rule**: ∂f/∂θ = [f(θ+2π/3) - f(θ-2π/3)] / (2·sin(2π/3))
- **Qutrit entanglement**: CSUM gate, Schmidt rank up to 3, generalized Bell states

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Math validation
pytest tests/math_validation/ -v

# Integration tests
pytest tests/integration/ -v

# All tests
pytest -v

# With coverage
pytest --cov=triflowqc --cov-report=html

# Benchmarks
pytest tests/benchmarks/ -v -m benchmark

# Validation scripts
python validation/validate_gates.py
python validation/validate_encoding.py
python validation/validate_training.py
```

## Examples

| Example | Description |
|---------|-------------|
| [01_basic_qutrits.py](examples/01_basic_qutrits.py) | Qutrit creation, gates, measurement |
| [02_circuits_simulation.py](examples/02_circuits_simulation.py) | Circuit building and simulation |
| [03_qnn_classifier.py](examples/03_qnn_classifier.py) | QNN ternary classifier |
| [04_noise_simulation.py](examples/04_noise_simulation.py) | Noise models and noisy simulation |
| [05_compilation.py](examples/05_compilation.py) | Qutrit-to-qubit compilation |
| [06_su3_math.py](examples/06_su3_math.py) | SU(3) algebra and quantum information |
| [07_hardware_run.py](examples/07_hardware_run.py) | IBM Quantum hardware pipeline |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_tutorial_introduction.ipynb](notebooks/01_tutorial_introduction.ipynb) | Introduction to qutrit computing |
| [02_qnn_training.ipynb](notebooks/02_qnn_training.ipynb) | QNN training tutorial |

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- Matplotlib ≥ 3.7

Optional:
- Qiskit ≥ 1.0 (for IBM hardware)
- qiskit-ibm-runtime ≥ 0.20 (for IBM hardware)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use TriFlowQC in your research, please cite:

```bibtex
@software{triflowqc2024,
  title={TriFlowQC: A Python Framework for Qutrit Quantum Neural Networks},
  year={2024},
  url={https://github.com/yourusername/TriFlowQC}
}
```
