"""Observe owned evaluator children without retrying or extending their budget."""

from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


_IMPORT_TIMING = re.compile(
    r"^import time:\s+(\d{1,12})\s+\|\s+(\d{1,12})\s+\|\s+([\w.]+)$"
)


def _import_timings(stderr):
    rows = deque(maxlen=12)
    for line in stderr.splitlines():
        match = _IMPORT_TIMING.fullmatch(line)
        if match is None:
            continue
        module = match[3].split(".", 1)[0]
        known = getattr(sys, "stdlib_module_names", ())
        module = module if module in known or module == "coverage" else "external"
        rows.append({"self_us": int(match[1]), "cumulative_us": int(match[2]), "module": module})
    return list(rows)


def run_evaluator(argv, *, trace=None):
    """Keep the original capture/text/three-second contract and failure type."""
    enabled = os.environ.get("SCITEX_DATASET_EVALUATOR_IMPORT_TRACE") == "1"
    enabled = enabled if trace is None else trace
    command = [argv[0], "-X", "importtime", *argv[1:]] if enabled else argv
    started = time.monotonic()
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        pressure = {}
        for name in ("cpu", "io", "memory"):
            try:
                pressure[name] = Path("/proc/pressure", name).read_text()[:4096]
            except OSError:
                pressure[name] = "unavailable"
        evidence = {
            "timeout_seconds": error.timeout,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "worker_pid": os.getpid(),
            "import_trace_enabled": enabled,
            "last_import_timings": _import_timings(stderr),
            "pressure": pressure,
        }
        # No command arguments, paths, submissions or arbitrary stderr content.
        print("EVALUATOR_TIMEOUT " + json.dumps(evidence), file=sys.stderr, flush=True)
        raise
