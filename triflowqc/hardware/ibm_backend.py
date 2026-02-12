"""
IBM Quantum backend wrapper.

Thin abstraction over qiskit-ibm-runtime that handles service
initialisation, backend selection, and session management.
"""

from __future__ import annotations

import warnings
from typing import Optional, List, Dict, Any

try:
    from qiskit_ibm_runtime import (
        QiskitRuntimeService,
        Session,
        SamplerV2 as Sampler,
        EstimatorV2 as Estimator,
    )
    from qiskit_ibm_runtime.fake_provider import FakeManila
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

try:
    from qiskit.circuit import QuantumCircuit as QiskitCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


class IBMBackend:
    """
    Wrapper around IBM Quantum Runtime services.

    Parameters
    ----------
    channel : str
        IBM Quantum channel ('ibm_quantum' or 'ibm_cloud').
    instance : str or None
        Hub/Group/Project instance string (e.g. 'ibm-q/open/main').
    token : str or None
        IBM Quantum API token. If None, uses saved credentials.
    backend_name : str or None
        Specific backend name, e.g. 'ibm_brisbane'. If None, uses
        least-busy available backend.
    """

    def __init__(
        self,
        channel: str = "ibm_quantum",
        instance: Optional[str] = None,
        token: Optional[str] = None,
        backend_name: Optional[str] = None,
    ):
        if not HAS_RUNTIME:
            raise ImportError(
                "qiskit-ibm-runtime is required. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        self._channel = channel
        self._instance = instance
        self._backend_name = backend_name
        self._service: Optional[QiskitRuntimeService] = None
        self._backend: Any = None
        self._session: Optional[Session] = None

        # Initialise service
        kwargs: Dict[str, Any] = {"channel": channel}
        if token is not None:
            kwargs["token"] = token
        if instance is not None:
            kwargs["instance"] = instance

        self._service = QiskitRuntimeService(**kwargs)

        # Select backend
        if backend_name:
            self._backend = self._service.backend(backend_name)
        else:
            self._backend = self._service.least_busy(
                simulator=False, operational=True, min_num_qubits=10
            )

    @property
    def backend(self):
        """Return the underlying Qiskit backend object."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Return the name of the selected backend."""
        return self._backend.name

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits on the backend."""
        return self._backend.num_qubits

    @property
    def max_qutrits(self) -> int:
        """Maximum number of qutrits (= num_qubits // 2)."""
        return self.num_qubits // 2

    def status(self) -> Dict[str, Any]:
        """Get backend status."""
        st = self._backend.status()
        return {
            "backend_name": st.backend_name,
            "operational": st.operational,
            "pending_jobs": st.pending_jobs,
            "status_msg": st.status_msg,
        }

    def configuration_summary(self) -> Dict[str, Any]:
        """Return a summary of the backend configuration."""
        config = self._backend.configuration()
        return {
            "backend_name": config.backend_name,
            "n_qubits": config.n_qubits,
            "basis_gates": config.basis_gates,
            "max_shots": config.max_shots,
            "max_experiments": config.max_experiments,
            "simulator": config.simulator,
        }

    def open_session(self, max_time: Optional[int] = None) -> Session:
        """
        Open a runtime session for batching multiple jobs.

        Parameters
        ----------
        max_time : int or None
            Maximum session time in seconds.

        Returns
        -------
        Session
        """
        kwargs: Dict[str, Any] = {
            "service": self._service,
            "backend": self._backend,
        }
        if max_time is not None:
            kwargs["max_time"] = max_time
        self._session = Session(**kwargs)
        return self._session

    def close_session(self) -> None:
        """Close the current session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def get_sampler(self, **options) -> "Sampler":
        """
        Create a Sampler primitive.

        Parameters
        ----------
        **options
            Options passed to the Sampler.

        Returns
        -------
        Sampler
        """
        if self._session is not None:
            return Sampler(session=self._session, options=options)
        return Sampler(backend=self._backend, options=options)

    def get_estimator(self, **options) -> "Estimator":
        """Create an Estimator primitive."""
        if self._session is not None:
            return Estimator(session=self._session, options=options)
        return Estimator(backend=self._backend, options=options)

    def available_backends(self, min_qubits: int = 0) -> List[str]:
        """List available backends with at least min_qubits qubits."""
        backends = self._service.backends(
            min_num_qubits=min_qubits, operational=True
        )
        return [b.name for b in backends]

    @staticmethod
    def save_account(token: str, channel: str = "ibm_quantum", **kwargs) -> None:
        """Save IBM Quantum credentials for future use."""
        if not HAS_RUNTIME:
            raise ImportError("qiskit-ibm-runtime required")
        QiskitRuntimeService.save_account(
            channel=channel, token=token, overwrite=True, **kwargs
        )

    def __repr__(self) -> str:
        return f"IBMBackend(backend={self.backend_name}, qubits={self.num_qubits})"

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, *args):
        self.close_session()
