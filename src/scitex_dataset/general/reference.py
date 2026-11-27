#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/general/reference.py

"""Registry of well-known reference files, fetched with provenance.

Each entry pins its file(s) by URL + sha256 and carries its license and
citation, so :func:`fetch_reference` returns a complete
:class:`~scitex_dataset._fetch_file.FetchRecord` (where it came from, what
it is, how to cite it) — not just a path. A single-file entry gives
``url`` / ``filename`` / ``sha256`` / ``size_bytes``; a multi-file entry gives
a ``files`` list of the same keys (``path`` for the name) and
:func:`fetch_reference` returns one record per file.

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
    "mnist-idx": {
        "name": "MNIST handwritten digits (original IDX files)",
        "description": (
            "The four gzipped IDX files of the original MNIST release: "
            "60,000 training and 10,000 test 28x28 grayscale digit images "
            "with their labels, as served by the CVDF mirror."
        ),
        # Verified 2026-11-27 by downloading every URL below; sha256 and size
        # measured, md5 equal to the officially published digests.
        "files": [
            {
                "path": "train-images-idx3-ubyte.gz",
                "url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                "sha256": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
                "size_bytes": 9_912_422,
                "md5": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            },
            {
                "path": "train-labels-idx1-ubyte.gz",
                "url": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
                "sha256": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
                "size_bytes": 28_881,
                "md5": "d53e105ee54ea40749a09fcbcd1e9432",
            },
            {
                "path": "t10k-images-idx3-ubyte.gz",
                "url": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                "sha256": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
                "size_bytes": 1_648_877,
                "md5": "9fb629c4189551a2d022fa330f9573f3",
            },
            {
                "path": "t10k-labels-idx1-ubyte.gz",
                "url": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
                "sha256": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
                "size_bytes": 4_542,
                "md5": "ec29112dd5afa0611ce80d1b7f02629c",
            },
        ],
        "format": "idx (gzip)",
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


def reference_files(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The pinned files of one entry, as ``{path, url, sha256, size_bytes}``."""
    if entry.get("files"):
        return [dict(item) for item in entry["files"]]
    return [
        {
            "path": entry["filename"],
            "url": entry["url"],
            "sha256": entry["sha256"],
            "size_bytes": entry.get("size_bytes"),
        }
    ]


def match_reference(sha256s) -> List[str]:
    """Ids of the entries whose EVERY pinned file is among ``sha256s``."""
    wanted = {str(s).lower() for s in sha256s if s}
    return [
        key
        for key, entry in REFERENCES.items()
        if all(f["sha256"] in wanted for f in reference_files(entry))
    ]


def list_references() -> List[Dict[str, Any]]:
    """Return every registered reference file as ``{"id": ..., **entry}`` dicts."""
    return [{"id": key, **entry} for key, entry in REFERENCES.items()]


def fetch_reference(
    name: str,
    dest_dir: Union[str, Path],
    *,
    timeout: float = 60.0,
    replace: bool = False,
) -> Union[FetchRecord, List[FetchRecord]]:
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
    FetchRecord or list of FetchRecord
        Including the entry's ``license`` and ``citation``; a list, one per
        file in entry order, for a multi-file entry (``files``).

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
    records = [
        fetch_file(
            item["url"],
            item["sha256"],
            dest_dir,
            item["path"],
            timeout=timeout,
            replace=replace,
            license=entry.get("license"),
            citation=entry.get("citation"),
        )
        for item in reference_files(entry)
    ]
    return records if entry.get("files") else records[0]


__all__ = [
    "REFERENCES",
    "fetch_reference",
    "list_references",
    "match_reference",
    "reference_files",
]

# EOF
