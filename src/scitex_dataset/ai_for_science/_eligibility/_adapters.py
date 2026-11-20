#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_adapters.py

"""Per-benchmark adapters: they supply FACTS, never decide eligibility.

Each adapter turns one benchmark's upstream layout into the same unit
records (see :data:`._config.FACTS`):

- ``units()`` — one record per screening unit (the archive an agent would
  get) with the cheap METADATA facts, read from the benchmark's own
  manifest. Only question text / keys and descriptive fields are read;
  no reference value is ever touched.
- ``resolve(unit, fact)`` — a metadata fact that costs I/O
  (``archive_bytes``: local file, else an HTTP ``HEAD`` / HF listing).
- ``archive_facts(unit, fetch=...)`` — the ARCHIVE facts, from the unit's
  archive via the dataset-blind :func:`._facts.analyze_tree`. A missing
  archive is fetched with the benchmark's own selective ``download`` when
  ``fetch`` is true, else :class:`ArchiveUnavailableError`.
- ``source_info()`` — sha256 of every manifest read, plus the pinned
  revision, so a frozen result names exactly what it screened.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

from ._archive import read_archive_tree
from ._code import is_figure_question
from ._facts import analyze_tree


class ArchiveUnavailableError(FileNotFoundError):
    """A unit's archive is needed for an archive-stage rule but is not staged."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Adapter:
    benchmark = ""
    entry_scripts: tuple[str, ...] | None = None

    def __init__(self, raw_dir: Path, source: dict | None = None):
        self.raw_dir = Path(raw_dir)
        self.source = dict(source or {})

    def _facts_from(self, archive: Path, member_prefix: str | None = None) -> dict:
        names, texts = read_archive_tree(archive, member_prefix=member_prefix)
        return analyze_tree(names, texts, entry_scripts=self.entry_scripts)

    def stage_manifests(self, *, fetch: bool) -> None:
        """Make sure the benchmark manifest is staged (fetch it when allowed)."""
        raise NotImplementedError

    def _missing(self, unit: dict, path: Path) -> ArchiveUnavailableError:
        return ArchiveUnavailableError(
            f"{self.benchmark}: archive of {unit['unit_id']} not staged at {path}; "
            "pass fetch=True (CLI --fetch) or stage it first"
        )


class CoreBenchAdapter(_Adapter):
    """CORE-Bench: one unit per capsule; questions are the reference-run keys."""

    benchmark = "corebench"
    entry_scripts = ("code/run", "code/run.sh")
    # Upstream marks figure-reading questions with a leading ``fig`` key token.
    _FIG_MARKER = re.compile(r"fig\b", re.I)

    def _oracle_paths(self) -> list[tuple[str, Path]]:
        from .._corebench_download import _ORACLE_TEST_RELPATH, _ORACLE_TRAIN_RELPATH

        pairs = [("train", self.raw_dir.joinpath(*_ORACLE_TRAIN_RELPATH)),
                 ("test", self.raw_dir.joinpath(*_ORACLE_TEST_RELPATH))]  # fmt: skip
        missing = [str(p) for _, p in pairs if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"corebench: manifests not staged: {missing}. Run "
                "corebench.bootstrap_oracle(raw_dir) (or `corebench download`) first."
            )
        return pairs

    def stage_manifests(self, *, fetch: bool) -> None:
        from .._corebench_download import bootstrap_oracle, oracle_staged

        if fetch and not oracle_staged(self.raw_dir):
            bootstrap_oracle(self.raw_dir)  # train + decrypted test manifests

    def units(self) -> list[dict]:
        units: dict[str, dict] = {}
        for split, path in self._oracle_paths():
            for rec in json.loads(path.read_text(encoding="utf-8")):
                cid = rec["capsule_id"]
                if cid in units:
                    raise ValueError(f"corebench: capsule {cid} listed twice")
                keys = sorted({k for run in rec.get("results") or [] for k in run})
                units[cid] = {
                    "unit_id": cid,
                    "label": cid,
                    "split": split,
                    "declared_language": rec.get("language"),
                    "n_questions": len(keys),
                    "n_figure_questions": sum(
                        1 for k in keys if self._FIG_MARKER.match(k)
                    ),
                    "extra": {"field": rec.get("field")},
                    "fact_sources": {"n_figure_questions": "upstream fig key marker"},
                }
        return list(units.values())

    def _archive(self, unit: dict) -> Path:
        return self.raw_dir / "capsules" / f"{unit['unit_id']}.tar.gz"

    def resolve(self, unit: dict, fact: str):
        if fact != "archive_bytes":
            raise KeyError(fact)
        path = self._archive(unit)
        if path.is_file():
            unit["fact_sources"]["archive_bytes"] = "local archive size"
            return path.stat().st_size
        from .._corebench_download import SOURCE_URL, _capsule_url

        req = urllib.request.Request(
            _capsule_url(SOURCE_URL, unit["unit_id"]), method="HEAD"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                length = resp.headers.get("Content-Length")
        except OSError as exc:
            unit["fact_sources"]["archive_bytes"] = f"unknown (HEAD failed: {exc})"
            return None
        unit["fact_sources"]["archive_bytes"] = "HTTP HEAD Content-Length"
        return int(length) if length is not None else None

    def archive_facts(self, unit: dict, *, fetch: bool) -> dict:
        path = self._archive(unit)
        if not path.is_file():
            if not fetch:
                raise self._missing(unit, path)
            from .._corebench_download import IncompleteDownloadError, download

            res = download(raw_dir=self.raw_dir, capsule_ids=[unit["unit_id"]])
            if res["n_failed"]:
                raise IncompleteDownloadError(f"corebench: {res['failures']}")
        return self._facts_from(path)

    def source_info(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "manifests": {p.name: _sha256(p) for _, p in self._oracle_paths()},
        }


class BixBenchAdapter(_Adapter):
    """BixBench: one unit per ``data_folder`` archive; its questions are the
    records sharing it. ``source.revision`` pins the HF snapshot."""

    benchmark = "bixbench"
    entry_scripts = None  # archives ship notebooks/scripts, no run script

    def _manifest(self) -> Path:
        from ..bixbench import ORACLE_MANIFEST_NAME

        path = self.raw_dir / ORACLE_MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"bixbench: {path} not staged; run bixbench.download(raw_dir, "
                "capsule_ids=[...]) (fetches the manifest) first."
            )
        return path

    def stage_manifests(self, *, fetch: bool) -> None:
        from ..bixbench import HF_REPO_ID, HF_REPO_TYPE, ORACLE_MANIFEST_NAME

        if not fetch or (self.raw_dir / ORACLE_MANIFEST_NAME).is_file():
            return
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "bixbench: fetching the manifest needs scitex-dataset[huggingface]"
            ) from exc
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        revision = self.source.get("revision")
        snapshot_download(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                          local_dir=str(self.raw_dir), allow_patterns=[ORACLE_MANIFEST_NAME],
                          **({"revision": revision} if revision else {}))  # fmt: skip

    def units(self) -> list[dict]:
        from .._capsule_layout import _native_capsule_id

        groups: dict[str, dict] = {}
        for line in self._manifest().read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            folder = rec.get("data_folder")
            if not folder:
                continue
            unit = groups.setdefault(folder, {
                "unit_id": _native_capsule_id(f"./{folder}"),
                "split": None, "declared_language": None,
                "n_questions": 0, "n_figure_questions": 0,
                "extra": {"data_folder": folder, "short_ids": set()},
                "fact_sources": {"n_figure_questions": "question-text keyword screen"},
            })  # fmt: skip
            unit["n_questions"] += 1
            unit["n_figure_questions"] += int(
                is_figure_question(rec.get("question", ""))
            )
            if rec.get("short_id"):
                unit["extra"]["short_ids"].add(rec["short_id"])
        units = []
        for unit in groups.values():
            ids = sorted(unit["extra"]["short_ids"])
            unit["extra"]["short_ids"] = ids
            unit["label"] = ",".join(ids) or unit["unit_id"]
            units.append(unit)
        return units

    def _archive(self, unit: dict) -> Path:
        return self.raw_dir / unit["extra"]["data_folder"]

    def _hf_sizes(self) -> dict[str, int] | None:
        """Archive sizes from the HF listing at the pinned revision; ``None``
        without the optional ``huggingface_hub`` (the size is then unknown)."""
        if not hasattr(self, "_sizes"):
            try:
                from huggingface_hub import HfApi
            except ImportError:  # extra [huggingface] not installed
                return None
            from ..bixbench import HF_REPO_ID, HF_REPO_TYPE

            items = HfApi().list_repo_tree(HF_REPO_ID, repo_type=HF_REPO_TYPE,
                                           revision=self.source.get("revision"))  # fmt: skip
            self._sizes = {i.path: i.size for i in items if getattr(i, "size", None)}
        return self._sizes

    def resolve(self, unit: dict, fact: str):
        if fact != "archive_bytes":
            raise KeyError(fact)
        path = self._archive(unit)
        if path.is_file():
            unit["fact_sources"]["archive_bytes"] = "local archive size"
            return path.stat().st_size
        try:
            sizes = self._hf_sizes()
        except OSError as exc:
            unit["fact_sources"]["archive_bytes"] = (
                f"unknown (HF listing failed: {exc})"
            )
            return None
        if sizes is None:
            unit["fact_sources"]["archive_bytes"] = (
                "unknown (huggingface_hub not installed)"
            )
            return None
        size = sizes.get(unit["extra"]["data_folder"])
        unit["fact_sources"]["archive_bytes"] = "HF repo listing"
        return size

    def archive_facts(self, unit: dict, *, fetch: bool) -> dict:
        path = self._archive(unit)
        if not path.is_file():
            if not fetch:
                raise self._missing(unit, path)
            from ..bixbench import download

            download(raw_dir=self.raw_dir, capsule_ids=[unit["unit_id"]],
                     revision=self.source.get("revision"))  # fmt: skip
        return self._facts_from(path)

    def source_info(self) -> dict:
        manifest = self._manifest()
        return {"benchmark": self.benchmark, "revision": self.source.get("revision"),
                "manifests": {manifest.name: _sha256(manifest)}}  # fmt: skip


class BioMysteryBenchAdapter(_Adapter):
    """BioMysteryBench: one unit per problem (one question each). Problems
    ship raw data only, so code facts come out empty; the preview packs
    every problem in ``data.zip`` (``<id>/...``), the full set per-problem
    ``data/<id>.zip``. ``source.variant`` is ``preview`` (default) or ``full``."""

    benchmark = "biomysterybench"
    entry_scripts = None

    def __init__(self, raw_dir: Path, source: dict | None = None):
        super().__init__(raw_dir, source)
        self.variant = self.source.get("variant", "preview")
        if self.variant not in ("preview", "full"):
            raise ValueError(f"biomysterybench: unknown variant {self.variant!r}")
        self.snapshot = (
            self.raw_dir / "variants" / "full"
            if self.variant == "full"
            else self.raw_dir
        )

    def stage_manifests(self, *, fetch: bool) -> None:
        if not fetch or (self.snapshot / "problems.csv").is_file():
            return
        if self.variant == "full":
            raise ArchiveUnavailableError(
                "biomysterybench: the gated full set has no selective download; "
                f"stage it under {self.snapshot} first"
            )
        from ..biomysterybench import download

        download(raw_dir=self.raw_dir, revision=self.source.get("revision"))

    def _problems(self) -> Path:
        path = self.snapshot / "problems.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"biomysterybench: {path} not staged; download first."
            )
        return path

    def units(self) -> list[dict]:
        units = []
        with self._problems().open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):  # answer_rubric is never read
                units.append({
                    "unit_id": row["id"], "label": row["id"], "split": None,
                    "declared_language": None, "n_questions": 1,
                    "n_figure_questions": int(is_figure_question(row.get("question", ""))),
                    "extra": {"human_solvable": row.get("human_solvable")},
                    "fact_sources": {"n_figure_questions": "question-text keyword screen"},
                })  # fmt: skip
        return units

    def _locate(self, unit: dict) -> tuple[Path, str | None] | None:
        per_problem = self.snapshot / "data" / f"{unit['unit_id']}.zip"
        if per_problem.is_file():
            return per_problem, None
        packed = self.snapshot / "data.zip"
        if packed.is_file():
            with zipfile.ZipFile(packed) as zf:
                if any(n.startswith(f"{unit['unit_id']}/") for n in zf.namelist()):
                    return packed, unit["unit_id"]
        return None

    def resolve(self, unit: dict, fact: str):
        if fact != "archive_bytes":
            raise KeyError(fact)
        found = self._locate(unit)
        if found is None:
            unit["fact_sources"]["archive_bytes"] = "unknown (no staged data)"
            return None
        path, prefix = found
        if prefix is None:
            unit["fact_sources"]["archive_bytes"] = "local archive size"
            return path.stat().st_size
        with zipfile.ZipFile(path) as zf:
            size = sum(i.compress_size for i in zf.infolist()
                       if i.filename.startswith(f"{prefix}/"))  # fmt: skip
        unit["fact_sources"]["archive_bytes"] = (
            "compressed bytes of its members in data.zip"
        )
        return size

    def archive_facts(self, unit: dict, *, fetch: bool) -> dict:
        found = self._locate(unit)
        if found is None:
            if not fetch or self.variant == "full":
                raise self._missing(unit, self.snapshot / "data")
            from ..biomysterybench import download

            download(raw_dir=self.raw_dir, revision=self.source.get("revision"))
            found = self._locate(unit)
            if found is None:
                raise self._missing(unit, self.snapshot / "data")
        return self._facts_from(found[0], member_prefix=found[1])

    def source_info(self) -> dict:
        problems = self._problems()
        return {"benchmark": self.benchmark, "variant": self.variant,
                "revision": self.source.get("revision"),
                "manifests": {problems.name: _sha256(problems)}}  # fmt: skip


ADAPTERS = {
    a.benchmark: a for a in (CoreBenchAdapter, BixBenchAdapter, BioMysteryBenchAdapter)
}

__all__ = ["ADAPTERS", "ArchiveUnavailableError"]

# EOF
