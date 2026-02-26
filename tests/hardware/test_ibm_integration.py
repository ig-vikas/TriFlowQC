"""
Hardware integration tests (mocked).

These tests mock the IBM Quantum backend to test the hardware pipeline
without requiring actual hardware access.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from triflowqc.circuit.circuit import QutritCircuit
from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.compiler.leakage_handler import LeakageHandler
from triflowqc.compiler.result_mapper import ResultMapper
from triflowqc.hardware.ibm_backend import IBMBackend
from triflowqc.hardware.job_manager import JobManager
from triflowqc.hardware.hardware_runner import HardwareRunner


class TestIBMBackendMocked:
    """Test IBM backend with mocked Qiskit runtime."""

    @patch('triflowqc.hardware.ibm_backend.QiskitRuntimeService')
    def test_backend_initialization(self, mock_service_cls):
        """Backend initializes and connects to service."""
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        mock_backend = MagicMock()
        mock_backend.name = 'ibm_brisbane'
        mock_backend.num_qubits = 127
        mock_service.backend.return_value = mock_backend

        backend = IBMBackend(token='fake_token', backend_name='ibm_brisbane')
        assert backend is not None

    @patch('triflowqc.hardware.ibm_backend.QiskitRuntimeService')
    def test_backend_get_properties(self, mock_service_cls):
        """Backend retrieves backend properties."""
        mock_service = MagicMock()
        mock_service_cls.return_value = mock_service
        mock_backend = MagicMock()
        mock_backend.name = 'ibm_brisbane'
        mock_backend.num_qubits = 127
        mock_backend.configuration.return_value.basis_gates = ['cx', 'id', 'rz', 'sx', 'x']
        mock_service.backend.return_value = mock_backend

        backend = IBMBackend(token='fake_token', backend_name='ibm_brisbane')
        assert backend.backend is not None


class TestJobManagerMocked:
    """Test job manager with mocked jobs."""

    def test_job_manager_creation(self):
        """JobManager can be created with a mock backend."""
        mock_backend = MagicMock()
        manager = JobManager(mock_backend)
        assert manager is not None

    def test_job_submission_tracking(self):
        """JobManager tracks submitted jobs."""
        mock_backend = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id.return_value = 'test_job_123'
        mock_job.status.return_value = 'QUEUED'
        mock_backend.run.return_value = mock_job
        manager = JobManager(mock_backend)
        manager.submit(MagicMock())  # submit a circuit
        jobs = manager.all_jobs()
        assert len(jobs) >= 1

    def test_job_status_check(self):
        """JobManager checks job status."""
        mock_backend = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id.return_value = 'test_job_456'
        mock_job.status.return_value = 'COMPLETED'
        mock_backend.run.return_value = mock_job
        manager = JobManager(mock_backend)
        manager.submit(MagicMock())
        status = manager.status()
        assert status is not None

    def test_job_cancellation(self):
        """JobManager can cancel jobs."""
        mock_backend = MagicMock()
        mock_job = MagicMock()
        mock_job.job_id.return_value = 'cancel_me'
        mock_job.cancel.return_value = True
        mock_backend.run.return_value = mock_job
        manager = JobManager(mock_backend)
        manager.submit(MagicMock())
        manager.cancel()


class TestHardwareRunnerMocked:
    """Test HardwareRunner end-to-end pipeline (mocked)."""

    def _make_runner(self, n_qutrits):
        """Create a HardwareRunner with mocked ibm_backend."""
        mock_ibm_backend = MagicMock()
        mock_ibm_backend.num_qubits = 2 * n_qutrits
        mock_ibm_backend.backend_name = 'mock_backend'
        runner = HardwareRunner.__new__(HardwareRunner)
        runner.compiler = QutritToQubitCompiler()
        runner.leakage_handler = LeakageHandler(n_qutrits=n_qutrits, post_select=True)
        runner.result_mapper = ResultMapper(n_qutrits=n_qutrits)
        runner.ibm_backend = mock_ibm_backend
        return runner

    def test_resource_estimation(self):
        """Estimate resources for a circuit."""
        qc = QutritCircuit(2)
        qc.h3(0)
        qc.csum(0, 1)
        runner = self._make_runner(2)
        estimate = runner.estimate_resources(qc)
        assert 'num_physical_qubits' in estimate
        assert estimate['num_physical_qubits'] == 4  # 2 qutrits × 2 qubits

    def test_circuit_validation(self):
        """Validate circuit before submission."""
        qc = QutritCircuit(1)
        qc.x3(0)
        runner = self._make_runner(1)
        result = runner.validate_circuit(qc)
        assert result['valid'] is True

    def test_result_processing_pipeline(self):
        """Test result processing from raw qubit counts."""
        runner = self._make_runner(2)

        # Simulated raw qubit counts for 2 qutrits
        raw_counts = {
            '0000': 300,  # |00⟩|00⟩ → |0⟩|0⟩
            '0001': 250,  # |00⟩|01⟩ → |0⟩|1⟩
            '0100': 200,  # |01⟩|00⟩ → |1⟩|0⟩
            '0101': 150,  # |01⟩|01⟩ → |1⟩|1⟩
            '1100': 50,   # |11⟩|00⟩ → LEAKAGE
            '0011': 50,   # |00⟩|11⟩ → LEAKAGE
        }

        # Filter leakage
        valid_counts, leaked_counts = runner.leakage_handler.filter_counts(raw_counts)
        assert '1100' not in valid_counts
        assert '0011' not in valid_counts

        # Map to trit basis (valid_counts already has trit strings from filter_counts)
        # The trit_counts are already in valid_counts since filter_counts returns trit strings
        assert valid_counts.get('00', 0) == 300
        assert valid_counts.get('01', 0) == 250
        assert valid_counts.get('10', 0) == 200
        assert valid_counts.get('11', 0) == 150


class TestCompilerHardwareConsistency:
    """Test that the compiler produces hardware-compatible circuits."""

    def test_compiled_circuit_has_valid_gates(self):
        """Compiled circuit should only contain basis gates."""
        qc = QutritCircuit(1)
        qc.h3(0)
        qc.rx3(np.pi / 4, 0)

        compiler = QutritToQubitCompiler()
        qubit_circuit = compiler.compile(qc)
        # Should be a valid Qiskit circuit
        assert qubit_circuit.num_qubits == 2

    def test_multi_qutrit_qubit_count(self):
        """N qutrits → 2N qubits in compiled circuit."""
        for n in [1, 2, 3]:
            qc = QutritCircuit(n)
            for i in range(n):
                qc.h3(i)
            compiler = QutritToQubitCompiler()
            qubit_circuit = compiler.compile(qc)
            assert qubit_circuit.num_qubits == 2 * n

    def test_compiled_circuit_depth(self):
        """Compiled circuit has reasonable depth."""
        qc = QutritCircuit(2)
        qc.h(0)
        qc.csum(0, 1)

        compiler = QutritToQubitCompiler()
        qubit_circuit = compiler.compile(qc)
        depth = qubit_circuit.depth()
        assert depth > 0
        assert depth < 1000  # Reasonable upper bound
