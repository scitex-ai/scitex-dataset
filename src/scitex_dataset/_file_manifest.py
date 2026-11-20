"""Portable known-file manifests using the checksum-verified fetch primitive."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import replace as replace_record
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlparse

from scitex_dataset._fetch_file import FetchError, FetchRecord, fetch_file

_SCHEMA = "scitex-file-manifest/v1"


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Manifest path must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or any(part in ("..", ".") for part in value.split("/"))
        or not path.name
    ):
        raise ValueError(f"Manifest path must remain relative: {value!r}")
    if any(not part for part in value.split("/")):
        raise ValueError("Manifest paths cannot have empty components")
    return path.as_posix()


def validate_file_manifest(manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate every entry before replay; preserve caller provenance fields.

    Hashes identify bytes, not publisher authenticity. A checksum_basis field
    is preserved unchanged when supplied. No checksum is learned from a download.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    if not isinstance(manifest, Mapping) or manifest.get("schema") != _SCHEMA:
        raise ValueError(f"Expected manifest schema {_SCHEMA!r}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Manifest files must be a list")
    result = json.loads(json.dumps(dict(manifest), allow_nan=False))
    seen = set()
    for entry in result["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Each manifest entry must be an object")
        path = _relative_path(entry.get("path"))
        if path in seen:
            raise ValueError(f"Duplicate manifest destination: {path}")
        seen.add(path)
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", entry["sha256"]
        ):
            raise ValueError("Every entry requires a pinned SHA256")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Every entry requires nonnegative integer size_bytes")
        url = entry.get("url")
        if not isinstance(url, str) or urlparse(url).scheme not in (
            "http",
            "https",
            "file",
        ):
            raise ValueError(
                "File URLs must use http, https or an explicit file mirror"
            )
        parsed = urlparse(url)
        if (parsed.scheme in ("http", "https") and not parsed.netloc) or (
            parsed.scheme == "file" and not parsed.path
        ):
            raise ValueError("Source URL must identify a host/file path")
        entry["path"], entry["sha256"] = path, entry["sha256"].lower()
    return result


def create_file_manifest(
    records: Iterable[FetchRecord | Mapping[str, Any]], root: str | Path
) -> dict[str, Any]:
    """Export fetch records with paths relative to root, without modifying them.

    Extra JSON metadata (license, citation, checksum_basis, subject IDs, etc.)
    is retained. No data are uploaded, downloaded or indexed in a database.
    """
    root = Path(root).expanduser().resolve()
    files = []
    for record in records:
        item = record.to_dict() if isinstance(record, FetchRecord) else dict(record)
        path = Path(item["path"]).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            item["path"] = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("Fetch record is outside the source root") from exc
        files.append(item)
    return validate_file_manifest({"schema": _SCHEMA, "files": files})


def replay_file_manifest(
    manifest: Mapping[str, Any] | str | Path,
    dest_root: str | Path,
    *,
    offline: bool = False,
    timeout: float = 60.0,
    replace: bool = False,
) -> list[dict[str, Any]]:
    """Materialize pinned files under a new root, returning verified fetch records.

    All paths/entries are preflighted before writes. Existing mismatched files
    are preserved unless replace=True. New/replacement files are size-checked
    in a temporary directory before atomic installation. Offline mode performs
    only local hash/size verification and never repairs or downloads files.
    """
    manifest = validate_file_manifest(manifest)
    root = Path(dest_root).expanduser().resolve()
    if timeout <= 0 or (offline and replace):
        raise ValueError(
            "timeout must be positive; offline replay cannot replace files"
        )
    targets = []
    seen = set()
    for entry in manifest["files"]:
        target = (root / entry["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Manifest destination escapes root through a symlink")
        if target in seen:
            raise ValueError("Manifest destinations resolve to the same path")
        seen.add(target)
        if offline and not target.is_file():
            raise FileNotFoundError(target)
        targets.append(target)
    records = []
    for entry, target in zip(manifest["files"], targets):
        existing = None
        if target.exists():
            try:
                existing = fetch_file(
                    entry["url"],
                    entry["sha256"],
                    target.parent,
                    filename=target.name,
                    timeout=timeout,
                    license=entry.get("license"),
                    citation=entry.get("citation"),
                )
            except FetchError:
                if not replace:
                    raise
        if existing is not None:
            if existing.size_bytes != entry["size_bytes"]:
                raise FetchError(f"Manifest size mismatch for {entry['path']}")
            verified = existing
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".scitex-manifest-", dir=target.parent
            ) as scratch:
                fetched = fetch_file(
                    entry["url"],
                    entry["sha256"],
                    scratch,
                    filename=target.name,
                    timeout=timeout,
                    license=entry.get("license"),
                    citation=entry.get("citation"),
                )
                if fetched.size_bytes != entry["size_bytes"]:
                    raise FetchError(f"Manifest size mismatch for {entry['path']}")
                if replace:
                    os.replace(fetched.path, target)
                else:
                    try:
                        os.link(fetched.path, target)
                    except FileExistsError as exc:
                        raise FetchError(
                            "Destination appeared during replay; existing file preserved"
                        ) from exc
                verified = replace_record(fetched, path=str(target))
        records.append({**entry, **verified.to_dict()})
    return records


__all__ = ["create_file_manifest", "validate_file_manifest", "replay_file_manifest"]
