#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/general/reference.py

"""Registry of well-known reference files, fetched with provenance.

Each entry pins ONE file by URL + sha256 and carries its license and
citation, so :func:`fetch_reference` returns a complete
:class:`~scitex_dataset._fetch_file.FetchRecord` (where it came from, what
it is, how to cite it) — not just a path.

Add a dataset by adding one dict to :data:`REFERENCES`. Verify the sha256
by downloading the file once before committing it; never copy a digest
unchecked.

Example
-------
>>> from scitex_dataset import fetch_reference
>>> rec = fetch_reference("mnist", "./data")          # doctest: +SKIP
>>> rec.sha256, rec.cached, rec.license               # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from .._fetch_file import FetchRecord, fetch_file

REFERENCES: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "name": "MNIST handwritten digits",
        "description": (
            "70,000 28x28 grayscale digit images as one NumPy archive: "
            "x_train (60000, 28, 28) uint8, y_train (60000,) uint8, "
            "x_test (10000, 28, 28) uint8, y_test (10000,) uint8."
        ),
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "filename": "mnist.npz",
        # Verified 2026-11-19 by downloading the URL above (11,490,434 bytes).
        "sha256": "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
        "size_bytes": 11_490_434,
        "format": "npz",
        "license": "CC BY-SA 3.0 (Yann LeCun and Corinna Cortes)",
        "citation": (
            "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "
            "Gradient-based learning applied to document recognition. "
            "Proceedings of the IEEE, 86(11), 2278-2324. "
            "doi:10.1109/5.726791"
        ),
        "homepage": "http://yann.lecun.com/exdb/mnist/",
    },
}


def list_references() -> List[Dict[str, Any]]:
    """Return every registered reference file as ``{"id": ..., **entry}`` dicts."""
    return [{"id": key, **entry} for key, entry in REFERENCES.items()]


def fetch_reference(
    name: str,
    dest_dir: Union[str, Path],
    *,
    timeout: float = 60.0,
    replace: bool = False,
) -> FetchRecord:
    """Fetch a registered reference file into ``dest_dir`` with provenance.

    Parameters
    ----------
    name : str
        Registry id (case-insensitive), e.g. ``"mnist"``. See
        :func:`list_references`.
    dest_dir : str or Path
        Directory to keep the file in (created if missing). A verified copy
        already there is reused (``cached=True``).
    timeout : float
        HTTP socket timeout in seconds.
    replace : bool
        Replace an existing file whose sha256 does not match. Default False
        (raise instead).

    Returns
    -------
    FetchRecord
        Including the entry's ``license`` and ``citation``.

    Raises
    ------
    KeyError
        Unknown ``name``.
    scitex_dataset.FetchError
        Checksum mismatch / truncated download (see :func:`fetch_file`).
    """
    key = name.strip().lower()
    if key not in REFERENCES:
        raise KeyError(
            f"fetch_reference: unknown reference {name!r}; known: {sorted(REFERENCES)}"
        )
    entry = REFERENCES[key]
    return fetch_file(
        entry["url"],
        entry["sha256"],
        dest_dir,
        entry["filename"],
        timeout=timeout,
        replace=replace,
        license=entry.get("license"),
        citation=entry.get("citation"),
    )


__all__ = ["REFERENCES", "fetch_reference", "list_references"]

# EOF
