"""Run the complete release suite inside the job's verified store context."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", required=True)
    requested = parser.parse_args().python_version
    expected = tuple(map(int, requested.split(".")))
    if len(expected) != 2 or sys.version_info[:2] != expected:
        raise RuntimeError("Release interpreter does not match the requested version")
    if not Path("tests").is_dir():
        raise RuntimeError("Release tests directory is missing")
    expected_dsn = os.environ.get("SCITEX_STORE_DSN")
    if not expected_dsn:
        raise RuntimeError("Release requires the explicit job-owned store")

    from scitex_dev.store.testing import writable_dsn

    with writable_dsn() as observed_dsn:
        if observed_dsn != expected_dsn:
            raise RuntimeError("Resolved store is not the job-owned service")
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-x"])
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
