"""
End-to-end hardware runner.

Combines the compiler, IBM backend, job manager, and result mapper into
a single high-level interface for running qutrit circuits on real hardware.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, List, Any, TYPE_CHECKING

from triflowqc.compiler.qutrit_to_qubit import QutritToQubitCompiler
from triflowqc.compiler.result_mapper import ResultMapper
from triflowqc.circuit.measurement import MeasurementResult

if TYPE_CHECKING:
    from triflowqc.circuit.circuit import QutritCircuit
    from triflowqc.hardware.ibm_backend import IBMBackend


class HardwareRunner:
    """
    High-level interface for running qutrit circuits on IBM Quantum hardware.

    Handles the full pipeline:
    1. Compile qutrit circuit → qubit circuit.
    2. Submit to IBM Quantum backend.
    3. Monitor job and retrieve results.
    4. Map qubit results back to qutrit space.

    Parameters
    ----------
    ibm_backend : IBMBackend
        IBM backend wrapper.
    optimization_level : int
        Compiler optimization level (0–3).
    leakage_post_select : bool
        Whether to post-select against leakage.
    shots : int
        Default number of measurement shots.
    """

    def __init__(
        self,
        ibm_backend: "IBMBackend",
        optimization_level: int = 1,
        leakage_post_select: bool = True,
        shots: int = 4096,
    ):
        self.ibm_backend = ibm_backend
        self.shots = shots
        self.leakage_post_select = leakage_post_select

        self.compiler = QutritToQubitCompiler(
            leakage_post_select=leakage_post_select,
            optimization_level=optimization_level,
        )
        self._result_mappers: Dict[int, ResultMapper] = {}

    def _get_result_mapper(self, n_qutrits: int) -> ResultMapper:
        if n_qutrits not in self._result_mappers:
            self._result_mappers[n_qutrits] = ResultMapper(
                n_qutrits=n_qutrits,
                post_select=self.leakage_post_select,
            )
        return self._result_mappers[n_qutrits]

    def run(
        self,
        circuit: "QutritCircuit",
        shots: Optional[int] = None,
        parameters: Optional[NDArray[np.floating]] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> MeasurementResult:
        """
        Run a qutrit circuit on IBM Quantum hardware.

        Parameters
        ----------
        circuit : QutritCircuit
        shots : int or None
        parameters : ndarray or None
            Parameter values to bind before compilation.
        wait : bool
            If True, block until the job completes.
        timeout : float or None
            Maximum wait time in seconds.

        Returns
        -------
        MeasurementResult
            Results mapped back to qutrit space.
        """
        from triflowqc.hardware.job_manager import JobManager

        shots = shots or self.shots

        # Compile
        compiled = self.compiler.compile_and_bind(circuit, parameters)

        # Submit
        manager = JobManager(self.ibm_backend)
        job_id = manager.submit(compiled, shots=shots)

        if wait:
            manager.wait(job_id, timeout=timeout)
            qubit_counts = manager.result(job_id)

            # Map back to qutrit space
            mapper = self._get_result_mapper(circuit.num_qutrits)
            return mapper.map_to_measurement_result(qubit_counts, shots=shots)
        else:
            # Return a placeholder; user can poll later
            return MeasurementResult(
                counts={"pending": 1},
                probabilities=np.zeros(3 ** circuit.num_qutrits),
                statevector=None,
                shots=shots,
                num_qutrits=circuit.num_qutrits,
                metadata={"job_id": job_id, "status": "SUBMITTED"},
            )

    def run_batch(
        self,
        circuits: List["QutritCircuit"],
        shots: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[MeasurementResult]:
        """
        Run multiple qutrit circuits on hardware.

        Parameters
        ----------
        circuits : list of QutritCircuit
        shots : int or None
        timeout : float or None

        Returns
        -------
        list of MeasurementResult
        """
        results = []
        for circ in circuits:
            results.append(self.run(circ, shots=shots, timeout=timeout))
        return results

    def estimate_resources(
        self, circuit: "QutritCircuit"
    ) -> Dict[str, Any]:
        """
        Estimate hardware resources needed.

        Returns
        -------
        dict
            Resource estimates including qubit/gate counts and
            compatibility with the current backend.
        """
        resources = self.compiler.resource_estimate(circuit)
        n_physical = resources["num_physical_qubits"]
        n_available = self.ibm_backend.num_qubits

        resources["backend_name"] = self.ibm_backend.backend_name
        resources["backend_qubits"] = n_available
        resources["fits_on_backend"] = n_physical <= n_available

        return resources

    def validate_circuit(self, circuit: "QutritCircuit") -> Dict[str, Any]:
        """
        Check if a circuit is compatible with the current backend.

        Returns
        -------
        dict
            Validation report.
        """
        resources = self.estimate_resources(circuit)
        issues: List[str] = []

        if not resources["fits_on_backend"]:
            issues.append(
                f"Circuit requires {resources['num_physical_qubits']} qubits "
                f"but backend has {resources['backend_qubits']}"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "resources": resources,
        }

    def __repr__(self) -> str:
        return (
            f"HardwareRunner(backend={self.ibm_backend.backend_name}, "
            f"shots={self.shots})"
        )
