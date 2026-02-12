"""
Job manager for IBM Quantum hardware jobs.

Handles job submission, monitoring, retrieval, and result caching.
"""

from __future__ import annotations

import time
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QiskitCircuit
    from triflowqc.hardware.ibm_backend import IBMBackend

try:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False


@dataclass
class JobRecord:
    """Record of a submitted job."""
    job_id: str
    backend_name: str
    circuit_name: str
    shots: int
    submitted_at: float
    status: str = "QUEUED"
    result: Optional[Dict] = None
    error: Optional[str] = None


class JobManager:
    """
    Manage IBM Quantum hardware jobs: submit, monitor, retrieve results.

    Parameters
    ----------
    ibm_backend : IBMBackend
        The IBM backend wrapper.
    cache_dir : str or Path or None
        Directory for caching results. None = no caching.
    max_concurrent : int
        Maximum concurrent jobs.
    """

    def __init__(
        self,
        ibm_backend: "IBMBackend",
        cache_dir: Optional[str] = None,
        max_concurrent: int = 5,
    ):
        self.ibm_backend = ibm_backend
        self.max_concurrent = max_concurrent
        self._jobs: Dict[str, JobRecord] = {}
        self._runtime_jobs: Dict[str, Any] = {}

        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

    def submit(
        self,
        circuit: "QiskitCircuit",
        shots: int = 4096,
        name: Optional[str] = None,
        **sampler_options,
    ) -> str:
        """
        Submit a circuit for execution.

        Parameters
        ----------
        circuit : QiskitCircuit
        shots : int
        name : str or None
        **sampler_options
            Additional Sampler options.

        Returns
        -------
        str
            Job ID.
        """
        if not HAS_RUNTIME:
            raise ImportError("qiskit-ibm-runtime required")

        # Check concurrent limit
        active = sum(
            1 for j in self._jobs.values()
            if j.status in ("QUEUED", "RUNNING")
        )
        if active >= self.max_concurrent:
            warnings.warn(
                f"Already {active} active jobs. Consider waiting before submitting more."
            )

        sampler = self.ibm_backend.get_sampler(**sampler_options)
        job = sampler.run([circuit], shots=shots)
        job_id = job.job_id()

        record = JobRecord(
            job_id=job_id,
            backend_name=self.ibm_backend.backend_name,
            circuit_name=name or circuit.name or "unnamed",
            shots=shots,
            submitted_at=time.time(),
        )
        self._jobs[job_id] = record
        self._runtime_jobs[job_id] = job
        return job_id

    def submit_batch(
        self,
        circuits: List["QiskitCircuit"],
        shots: int = 4096,
        **sampler_options,
    ) -> List[str]:
        """
        Submit multiple circuits as a batch.

        Returns
        -------
        list of str
            Job IDs.
        """
        job_ids = []
        for i, circ in enumerate(circuits):
            jid = self.submit(
                circ, shots=shots, name=f"batch_{i}", **sampler_options
            )
            job_ids.append(jid)
        return job_ids

    def status(self, job_id: str) -> str:
        """
        Get the status of a job.

        Returns
        -------
        str
            One of: QUEUED, RUNNING, DONE, ERROR, CANCELLED.
        """
        if job_id not in self._runtime_jobs:
            return self._jobs.get(job_id, JobRecord("?", "", "", 0, 0)).status

        job = self._runtime_jobs[job_id]
        try:
            st = job.status().name
        except Exception:
            st = "UNKNOWN"

        self._jobs[job_id].status = st
        return st

    def wait(
        self,
        job_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 5.0,
    ) -> str:
        """
        Wait for a job to complete.

        Parameters
        ----------
        job_id : str
        timeout : float or None
            Max wait time in seconds.
        poll_interval : float
            Polling interval in seconds.

        Returns
        -------
        str
            Final status.
        """
        start = time.time()
        while True:
            st = self.status(job_id)
            if st in ("DONE", "ERROR", "CANCELLED"):
                return st
            if timeout and (time.time() - start) > timeout:
                return st
            time.sleep(poll_interval)

    def result(self, job_id: str) -> Dict[str, int]:
        """
        Retrieve the measurement counts for a completed job.

        Returns
        -------
        dict
            {bitstring: count}
        """
        if job_id not in self._runtime_jobs:
            raise KeyError(f"Job {job_id} not found")

        job = self._runtime_jobs[job_id]
        result = job.result()

        # Extract counts from SamplerV2 result
        pub_result = result[0]
        counts = {}
        if hasattr(pub_result, 'data'):
            data = pub_result.data
            if hasattr(data, 'meas'):
                counts = data.meas.get_counts()
            elif hasattr(data, 'c'):
                counts = data.c.get_counts()
        else:
            # Fallback for older format
            counts = pub_result.get_counts() if hasattr(pub_result, 'get_counts') else {}

        self._jobs[job_id].result = counts
        self._jobs[job_id].status = "DONE"

        # Cache result
        if self._cache_dir is not None:
            self._cache_result(job_id, counts)

        return counts

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        if job_id in self._runtime_jobs:
            try:
                self._runtime_jobs[job_id].cancel()
                self._jobs[job_id].status = "CANCELLED"
                return True
            except Exception:
                return False
        return False

    def all_jobs(self) -> List[JobRecord]:
        """Return all job records."""
        return list(self._jobs.values())

    def active_jobs(self) -> List[JobRecord]:
        """Return active (queued/running) job records."""
        return [
            j for j in self._jobs.values() if j.status in ("QUEUED", "RUNNING")
        ]

    def _cache_result(self, job_id: str, counts: Dict[str, int]) -> None:
        """Save result to disk cache."""
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{job_id}.json"
        record = {
            "job_id": job_id,
            "backend_name": self._jobs[job_id].backend_name,
            "shots": self._jobs[job_id].shots,
            "counts": counts,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def load_cached_result(self, job_id: str) -> Optional[Dict[str, int]]:
        """Load result from disk cache."""
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{job_id}.json"
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("counts")
        return None

    def __repr__(self) -> str:
        n_total = len(self._jobs)
        n_active = len(self.active_jobs())
        return f"JobManager(total={n_total}, active={n_active})"
