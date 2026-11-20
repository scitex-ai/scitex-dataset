#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/_config.py

"""Eligibility criteria as data: the config schema, its validation, and the
predicate evaluator.

A config (YAML file, JSON file or dict)::

    schema: scitex-dataset/eligibility-config-v1   # optional, checked if given
    benchmark: corebench                           # corebench | bixbench | biomysterybench
    source: {revision: <hf sha>, variant: preview} # optional, adapter-specific
    rules:                                         # ALL must hold; order = reporting order
      - id: 1_split
        fact: split
        in: [train]
      - id: 8_light_deps
        all:
          - {fact: dl_frameworks, none_of: [tensorflow, torch, keras, jax]}
          - {fact: code_languages, none_of: [r]}
    order:                                         # candidate order (metadata facts only)
      keys:
        - {fact: n_questions, descending: true}
        - {fact: archive_bytes, descending: true}
        - {fact: label}
      seed: null            # int: seeded random tie-break (or a full shuffle without keys)
    target: 4               # take the first N survivors; null = screen every candidate
    on_unknown: exclude     # a rule on an unknown (null) fact excludes (default) or errors

Operators: ``equals``, ``in``, ``not_in``, ``min``, ``max`` (scalar facts);
``any_of``, ``none_of``, ``subset_of`` (list facts); ``all`` / ``any``
combine predicates. ``extra.<key>`` reaches an adapter's own metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_SCHEMA = "scitex-dataset/eligibility-config-v1"

# fact -> (stage, description). Metadata facts come from the benchmark's
# manifest (cheap); archive facts need the unit's archive.
FACTS: dict[str, tuple[str, str]] = {
    "unit_id": ("metadata", "native unit id (what prepare(only=...) accepts)"),
    "label": ("metadata", "human-facing name (CORE-Bench capsule id, BixBench short_id)"),
    "split": ("metadata", "upstream split (train/test) or null when none"),
    "declared_language": ("metadata", "language the benchmark manifest declares"),
    "n_questions": ("metadata", "number of questions (tasks) of the unit"),
    "n_figure_questions": ("metadata", "questions that read a figure/plot/image"),
    "archive_bytes": ("metadata", "size of the unit's archive in bytes"),
    "file_languages": ("archive", "languages of code files by extension"),
    "kernel_languages": ("archive", "languages of notebook kernels"),
    "magic_languages": ("archive", "languages run from Python: %%R / %R magics, rpy2"),
    "toolchain_languages": ("archive", "languages the environment installs or the entry invokes"),
    "code_languages": ("archive", "union of the four language facts"),
    "dl_frameworks": ("archive", "deep-learning frameworks installed or imported"),
    "gpu_in_environment": ("archive", "the environment spec asks for a GPU (CUDA base image, -gpu wheels)"),
    "gpu_in_code": ("archive", "code uses a GPU (cupy, .cuda(), 'cuda' device)"),
    "gpu_required": ("archive", "gpu_in_environment or gpu_in_code"),
    "entry_kinds": ("archive", "what the entry executes: python, notebook, r, other"),
    "n_entry_steps": ("archive", "scripts/notebooks/modules the entry executes"),
    "entry_notebook_code_cells": ("archive", "max non-empty code cells of an entry notebook"),
    "entry_notebook_local_imports": ("archive", "max project-local modules an entry notebook imports"),
}  # fmt: skip

SCALAR_OPS = ("equals", "in", "not_in", "min", "max")
LIST_OPS = ("any_of", "none_of", "subset_of")
_TOP_KEYS = {"schema", "benchmark", "description", "source", "rules", "order",
             "target", "on_unknown"}  # fmt: skip


class EligibilityConfigError(ValueError):
    """The eligibility config is malformed (unknown key, fact or operator)."""


def fact_stage(fact: str) -> str:
    if fact.startswith("extra."):
        return "metadata"
    if fact not in FACTS:
        raise EligibilityConfigError(f"unknown fact {fact!r}; known: {sorted(FACTS)}")
    return FACTS[fact][0]


def _check_predicate(pred, where: str) -> list[str]:
    """Validate one predicate; return the facts it reads."""
    if not isinstance(pred, dict):
        raise EligibilityConfigError(f"{where}: predicate must be a mapping")
    combos = [k for k in ("all", "any") if k in pred]
    if combos:
        if (
            len(pred) != 1
            or not isinstance(pred[combos[0]], list)
            or not pred[combos[0]]
        ):
            raise EligibilityConfigError(
                f"{where}: '{combos[0]}' takes a non-empty list alone"
            )
        facts: list[str] = []
        for i, sub in enumerate(pred[combos[0]]):
            facts += _check_predicate(sub, f"{where}.{combos[0]}[{i}]")
        return facts
    ops = [k for k in pred if k != "fact"]
    if "fact" not in pred or len(ops) != 1:
        raise EligibilityConfigError(f"{where}: needs 'fact' and exactly one operator")
    op = ops[0]
    if op not in SCALAR_OPS + LIST_OPS:
        raise EligibilityConfigError(f"{where}: unknown operator {op!r}")
    if op in ("in", "not_in") + LIST_OPS and not isinstance(pred[op], list):
        raise EligibilityConfigError(f"{where}: '{op}' takes a list")
    if op in ("min", "max") and (
        isinstance(pred[op], bool) or not isinstance(pred[op], (int, float))
    ):
        raise EligibilityConfigError(f"{where}: '{op}' takes a number")
    fact_stage(pred["fact"])
    return [pred["fact"]]


def load_config(config) -> dict:
    """Validate a config (dict, or a ``.yaml``/``.yml``/``.json`` path) and
    return it normalized (defaults filled in, rule stages attached)."""
    if isinstance(config, (str, Path)):
        path = Path(config)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            config = json.loads(text)
        else:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise EligibilityConfigError(
                    f"{path.name}: reading YAML needs pyyaml; install it or pass "
                    "the config as JSON / a dict"
                ) from exc
            config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise EligibilityConfigError("config must be a mapping")
    unknown = set(config) - _TOP_KEYS
    if unknown:
        raise EligibilityConfigError(f"unknown config keys: {sorted(unknown)}")
    if config.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise EligibilityConfigError(f"schema must be {CONFIG_SCHEMA!r}")
    if not isinstance(config.get("benchmark"), str):
        raise EligibilityConfigError("'benchmark' is required")
    rules = config.get("rules")
    if not isinstance(rules, list) or not rules:
        raise EligibilityConfigError("'rules' must be a non-empty list")
    seen: set[str] = set()
    norm_rules = []
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
            raise EligibilityConfigError(f"rules[{i}]: needs a string 'id'")
        if rule["id"] in seen:
            raise EligibilityConfigError(f"rules[{i}]: duplicate id {rule['id']!r}")
        seen.add(rule["id"])
        # ``stage`` is derived (an already-normalized config loads unchanged).
        rule = {k: v for k, v in rule.items() if k != "stage"}
        pred = {k: v for k, v in rule.items() if k not in ("id", "description")}
        facts = _check_predicate(pred, f"rule {rule['id']}")
        stage = (
            "archive" if any(fact_stage(f) == "archive" for f in facts) else "metadata"
        )
        norm_rules.append({**rule, "stage": stage})
    order = config.get("order") or {}
    if set(order) - {"keys", "seed"}:
        raise EligibilityConfigError("order takes only 'keys' and 'seed'")
    keys = []
    for key in order.get("keys") or []:
        if (
            not isinstance(key, dict)
            or set(key) - {"fact", "descending"}
            or "fact" not in key
        ):
            raise EligibilityConfigError(
                f"order key {key!r}: needs 'fact' (+ 'descending')"
            )
        if fact_stage(key["fact"]) != "metadata":
            raise EligibilityConfigError(
                f"order key {key['fact']!r}: metadata facts only"
            )
        keys.append(
            {"fact": key["fact"], "descending": bool(key.get("descending", False))}
        )
    seed = order.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise EligibilityConfigError("order.seed must be an integer or null")
    target = config.get("target")
    if target is not None and (
        isinstance(target, bool) or not isinstance(target, int) or target < 1
    ):
        raise EligibilityConfigError("target must be a positive integer or null")
    on_unknown = config.get("on_unknown", "exclude")
    if on_unknown not in ("exclude", "error"):
        raise EligibilityConfigError("on_unknown must be 'exclude' or 'error'")
    return {
        "schema": CONFIG_SCHEMA,
        "benchmark": config["benchmark"],
        "description": config.get("description"),
        "source": dict(config.get("source") or {}),
        "rules": norm_rules,
        "order": {"keys": keys, "seed": seed},
        "target": target,
        "on_unknown": on_unknown,
    }


def config_sha256(config: dict) -> str:
    """Hash of the normalized config (canonical JSON)."""
    body = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def get_fact(facts: dict, name: str):
    if name.startswith("extra."):
        return (facts.get("extra") or {}).get(name[len("extra.") :])
    return facts.get(name)


def evaluate(pred: dict, facts: dict) -> tuple[bool | None, list[dict]]:
    """Evaluate one predicate: ``(True|False|None-for-unknown, checks)``."""
    for combo in ("all", "any"):
        if combo in pred:
            results = [evaluate(sub, facts) for sub in pred[combo]]
            outcomes = [r[0] for r in results]
            checks = [c for r in results for c in r[1]]
            decisive = False if combo == "all" else True
            if decisive in outcomes:
                return decisive, checks
            return (None if None in outcomes else not decisive), checks
    op = next(k for k in pred if k not in ("fact", "id", "description", "stage"))
    expected = pred[op]
    value = get_fact(facts, pred["fact"])
    if value is None:
        ok = None
    elif op == "equals":
        ok = value == expected
    elif op == "in":
        ok = value in expected
    elif op == "not_in":
        ok = value not in expected
    elif op == "min":
        ok = value >= expected
    elif op == "max":
        ok = value <= expected
    else:
        items = set(value if isinstance(value, (list, tuple, set)) else [value])
        if op == "any_of":
            ok = bool(items & set(expected))
        elif op == "none_of":
            ok = not items & set(expected)
        else:
            ok = items <= set(expected)
    check = {
        "fact": pred["fact"],
        "op": op,
        "expected": expected,
        "value": value,
        "ok": ok,
    }
    return ok, [check]


__all__ = [
    "CONFIG_SCHEMA",
    "FACTS",
    "EligibilityConfigError",
    "config_sha256",
    "evaluate",
    "load_config",
]

# EOF
