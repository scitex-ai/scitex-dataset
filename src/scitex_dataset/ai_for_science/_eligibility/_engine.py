#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_engine.py

"""The screening walk: config + adapter facts -> an auditable unit list.

Semantics (fixed, so a frozen result can be re-derived):

1. Every unit gets its METADATA rules checked in declared order; the first
   failing rule excludes it (cheap rules short-circuit, so a unit already
   out never costs a download or ``HEAD``).
2. The surviving candidates are ordered by ``order.keys`` (``null`` last,
   strings in natural order), then by a seeded random draw when
   ``order.seed`` is set, then by natural ``unit_id``.
3. Candidates are walked in that order. While fewer than ``target`` are
   selected, a candidate's archive facts are read (fetched if allowed) and
   ALL its archive rules evaluated; it is ``selected`` when every rule
   holds, else ``excluded`` with the first failing rule as its reason and
   every failing rule listed (near misses stay visible). Once the target is
   met, the rest are ``not_screened`` (or ``eligible_not_needed`` when the
   config has no archive rule, i.e. they are already fully decided).

Every unit therefore carries a verdict, a reason and the facts behind it.
"""

from __future__ import annotations

import functools
import random
import re
from collections import Counter
from pathlib import Path

from ._adapters import ADAPTERS
from ._config import config_sha256, evaluate, get_fact, load_config

RESULT_SCHEMA = "scitex-dataset/eligibility-result-v1"


def _natural(text) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


def _compare(a, b, descending: bool) -> int:
    if a is None or b is None:
        return (a is None) - (b is None)  # null last, whatever the direction
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        a, b = _natural(a), _natural(b)
    result = (a > b) - (a < b)
    return -result if descending else result


def _order(units: list[dict], order: dict) -> list[dict]:
    draws: dict[str, float] = {}
    if order["seed"] is not None:
        rng = random.Random(order["seed"])
        for unit in sorted(units, key=lambda u: _natural(u["unit_id"])):
            draws[unit["unit_id"]] = rng.random()

    def cmp(x, y):
        for key in order["keys"]:
            c = _compare(
                get_fact(x, key["fact"]), get_fact(y, key["fact"]), key["descending"]
            )
            if c:
                return c
        if draws:
            c = _compare(draws[x["unit_id"]], draws[y["unit_id"]], False)
            if c:
                return c
        return _compare(x["unit_id"], y["unit_id"], False)

    return sorted(units, key=functools.cmp_to_key(cmp))


def _rule_outcome(rule: dict, facts: dict, on_unknown: str) -> dict | None:
    """``None`` when the rule holds, else the failure record."""
    ok, checks = evaluate(rule, facts)
    if ok:
        return None
    if ok is None and on_unknown == "error":
        raise ValueError(
            f"rule {rule['id']}: a fact is unknown for {facts['unit_id']}: {checks}"
        )
    return {"rule": rule["id"], "outcome": "unknown" if ok is None else "fail",
            "checks": checks}  # fmt: skip


def _ensure(adapter, unit: dict, facts_needed: list[str]) -> None:
    for name in facts_needed:
        if name == "archive_bytes" and name not in unit:
            unit[name] = adapter.resolve(unit, name)


def _facts_of(pred: dict) -> list[str]:
    for combo in ("all", "any"):
        if combo in pred:
            return [f for sub in pred[combo] for f in _facts_of(sub)]
    return [pred["fact"]]


def screen_eligibility(
    config,
    *,
    dataset_root: Path | str | None = None,
    raw_dir: Path | str | None = None,
    fetch: bool = False,
) -> dict:
    """Screen one benchmark's units against ``config``; return the result.

    ``raw_dir`` defaults to ``<dataset-root>/ai-for-science/<benchmark>/raw``
    (see :func:`..resolve_paths`). With ``fetch`` a missing manifest and each
    missing archive a rule needs are downloaded selectively (verified) into
    ``raw_dir`` by the benchmark's own downloader; without it a missing
    manifest raises ``FileNotFoundError`` and a needed archive that is not
    staged raises :class:`ArchiveUnavailableError`.
    The result is deterministic for the same config, manifests and archives.
    """
    cfg = load_config(config)
    if cfg["benchmark"] not in ADAPTERS:
        raise ValueError(
            f"unknown benchmark {cfg['benchmark']!r}; known: {sorted(ADAPTERS)}"
        )
    if raw_dir is None:
        from .._base import resolve_paths

        raw_dir = resolve_paths(cfg["benchmark"], dataset_root=dataset_root).raw_dir
    adapter = ADAPTERS[cfg["benchmark"]](Path(raw_dir), cfg["source"])
    adapter.stage_manifests(fetch=fetch)
    meta_rules = [r for r in cfg["rules"] if r["stage"] == "metadata"]
    archive_rules = [r for r in cfg["rules"] if r["stage"] == "archive"]

    records: dict[str, dict] = {}
    candidates: list[dict] = []
    for unit in adapter.units():
        rec = {"unit_id": unit["unit_id"], "label": unit["label"], "position": None,
               "verdict": None, "reason": None, "failed_rules": [], "facts": unit}  # fmt: skip
        records[unit["unit_id"]] = rec
        for rule in meta_rules:
            _ensure(adapter, unit, _facts_of(rule))
            failure = _rule_outcome(rule, unit, cfg["on_unknown"])
            if failure:
                rec.update(
                    verdict="excluded", reason=failure, failed_rules=[failure["rule"]]
                )
                break
        else:
            candidates.append(unit)

    for unit in candidates:
        _ensure(adapter, unit, [k["fact"] for k in cfg["order"]["keys"]])
    ordered = _order(candidates, cfg["order"])
    selected: list[str] = []
    for position, unit in enumerate(ordered, 1):
        rec = records[unit["unit_id"]]
        rec["position"] = position
        if cfg["target"] is not None and len(selected) >= cfg["target"]:
            rec["verdict"] = "not_screened" if archive_rules else "eligible_not_needed"
            continue
        if archive_rules:
            unit.update(adapter.archive_facts(unit, fetch=fetch))
            failures = [f for f in (_rule_outcome(r, unit, cfg["on_unknown"])
                                    for r in archive_rules) if f]  # fmt: skip
            if failures:
                rec.update(verdict="excluded", reason=failures[0],
                           failed_rules=[f["rule"] for f in failures])  # fmt: skip
                continue
        rec["verdict"] = "selected"
        selected.append(unit["unit_id"])

    in_order = [records[u["unit_id"]] for u in ordered]
    rest = sorted((r for r in records.values() if r["position"] is None),
                  key=lambda r: _natural(r["unit_id"]))  # fmt: skip
    units = in_order + rest
    return {
        "schema": RESULT_SCHEMA,
        "scitex_dataset_version": _version(),
        "benchmark": cfg["benchmark"],
        "config": cfg,
        "config_sha256": config_sha256(cfg),
        "source": adapter.source_info(),
        "target": cfg["target"],
        "n_units": len(units),
        "n_candidates": len(candidates),
        "selected": selected,
        "selected_labels": [records[u]["label"] for u in selected],
        "shortfall": max(0, cfg["target"] - len(selected)) if cfg["target"] else 0,
        "counts": dict(sorted(Counter(r["verdict"] for r in units).items())),
        "units": units,
    }


def _version() -> str:
    from ... import __version__

    return __version__


def write_result(result: dict, path: Path | str) -> Path:
    """Write a screening result as canonical JSON (sorted keys, stable)."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")  # fmt: skip
    return path


__all__ = ["RESULT_SCHEMA", "screen_eligibility", "write_result"]

# EOF
