#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_gate_plugin.py

"""Thin scitex-dev submission-gate provider (entry-point target).

Registered under the ``scitex_dev.gate.checks`` entry-point group. The
``scitex_dev.gate`` import is DEFERRED into the functions so this module
imports cleanly even when ``scitex_dev`` is not installed — all the real
logic lives in the agnostic :mod:`.ai_for_science._gate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .ai_for_science._gate import CHECK_ID, build_gate_result


def _run(workdir: Path, config: Mapping):
    from scitex_dev.gate import Finding, GateResult  # pragma: no cover

    r = build_gate_result(workdir, config)  # pragma: no cover
    return GateResult(  # pragma: no cover
        passed=r["passed"], findings=tuple(Finding(**f) for f in r["findings"])
    )


def provide():
    from scitex_dev.gate import GateCheck  # pragma: no cover

    return [  # pragma: no cover
        GateCheck(
            id=CHECK_ID,
            stage="pre-submission",
            run=_run,
            requires="",
            description=(
                "scitex-dataset: structural, oracle-free submission-format "
                "validation for the bound capsule (validate_submission)."
            ),
        )
    ]


__all__ = ["provide"]

# EOF
