"""Process-scoped GPU memory sampler shared by the JAX and Torch workers.

Polls ``nvidia-smi --query-compute-apps`` for this process and records the
high-water mark. ``nvidia-smi`` reports the driver-level footprint, i.e. what
another job on the card actually loses: for JAX that is the BFC pool
high-water mark (with ``XLA_PYTHON_CLIENT_PREALLOCATE=false``), for Torch the
caching-allocator reserved size plus the CUDA context.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


def _query_process_memory_mib(pid: int) -> float | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    total = 0.0
    found = False
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) != pid:
                continue
            total += float(fields[1])
            found = True
        except ValueError:
            continue
    return total if found else None


class ProcessGpuMemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        self._pid = os.getpid()
        self._interval = float(interval_seconds)
        self._stop = threading.Event()
        self._peak_mib = 0.0
        self._samples = 0
        self._last_mib = 0.0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            self._record(_query_process_memory_mib(self._pid))
            time.sleep(self._interval)

    def _record(self, value):
        if value is None:
            return
        with self._lock:
            self._samples += 1
            self._last_mib = value
            self._peak_mib = max(self._peak_mib, value)

    def sample_now(self) -> float | None:
        """Synchronously read the current footprint and fold it into the peak."""
        value = _query_process_memory_mib(self._pid)
        self._record(value)
        return value

    def reset_peak(self) -> float:
        """Return the peak so far and restart peak tracking from the current value."""
        self.sample_now()
        with self._lock:
            peak = self._peak_mib
            self._peak_mib = self._last_mib
        return peak

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=10.0)
        # One final synchronous read so the last phase is not missed.
        self.sample_now()
        return {
            "nvidia_smi_peak_mib": self._peak_mib,
            "nvidia_smi_last_mib": self._last_mib,
            "nvidia_smi_samples": self._samples,
        }

    @property
    def peak_mib(self) -> float:
        return self._peak_mib
