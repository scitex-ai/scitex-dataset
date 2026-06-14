#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_ai_for_science.py

"""MCP tools for the ``ai-for-science`` benchmarks.

One ``dataset_<bench>_standardize`` tool per benchmark — the read-only,
network-free pure-Python standardize is the variant safe to expose over
MCP. ``download`` and ``prepare`` are side-effecting (multi-GB pulls,
writes to operator-private directories), so they stay CLI-only on
purpose: the right call site for the heavy verbs is a ``sbatch`` job,
not an MCP client.

Each ``standardize`` tool reads the operator-private ``raw_dir``
snapshot and writes the agent-visible ``for_solver`` view (uniform
``tasks.jsonl`` + submission schema) plus the operator ``eval`` view
(``answers.jsonl`` + ``evaluate.py``). All three default to the
canonical ``<dataset-root>/ai-for-science/<benchmark>/{raw,for_solver,eval}``
layout when not given.
"""

from typing import Any, Dict, Optional


def register_ai_for_science_tools(mcp) -> None:
    """Register ``dataset_<corebench|bixbench|biomysterybench>_standardize`` tools."""

    @mcp.tool()
    def corebench_standardize(
        raw_dir: Optional[str] = None,
        for_solver_dir: Optional[str] = None,
        eval_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Split CORE-Bench oracle JSONs into the agent for_solver/ view + operator eval/ view - use whenever the user asks to prepare CORE-Bench for an agentic-reproducibility experiment, build the uniform answer-free tasks.jsonl the SAC capsule binds at /for_solver:ro, or emit the eval/answers.jsonl + evaluate.py scorer. Replaces the legacy mask step (raw -> for_solver/eval)."""
        from pathlib import Path

        from ...ai_for_science import corebench, resolve_paths

        if raw_dir is None or for_solver_dir is None or eval_dir is None:
            paths = resolve_paths(corebench.BENCHMARK)
            raw_dir = raw_dir or str(paths.raw_dir)
            for_solver_dir = for_solver_dir or str(paths.for_solver_dir)
            eval_dir = eval_dir or str(paths.eval_dir)
        return corebench.standardize(
            raw_dir=Path(raw_dir),
            for_solver_dir=Path(for_solver_dir),
            eval_dir=Path(eval_dir),
        )

    @mcp.tool()
    def bixbench_standardize(
        raw_dir: Optional[str] = None,
        for_solver_dir: Optional[str] = None,
        eval_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Split BixBench oracle JSONL into the agent for_solver/ view + operator eval/ view - use whenever the user asks to prepare BixBench for an agentic-bioinformatics experiment, build the uniform answer-free tasks.jsonl, or emit the eval/answers.jsonl + evaluate.py scorer. Replaces the legacy mask step (raw -> for_solver/eval)."""
        from pathlib import Path

        from ...ai_for_science import bixbench, resolve_paths

        if raw_dir is None or for_solver_dir is None or eval_dir is None:
            paths = resolve_paths(bixbench.BENCHMARK)
            raw_dir = raw_dir or str(paths.raw_dir)
            for_solver_dir = for_solver_dir or str(paths.for_solver_dir)
            eval_dir = eval_dir or str(paths.eval_dir)
        return bixbench.standardize(
            raw_dir=Path(raw_dir),
            for_solver_dir=Path(for_solver_dir),
            eval_dir=Path(eval_dir),
        )

    @mcp.tool()
    def biomysterybench_standardize(
        raw_dir: Optional[str] = None,
        for_solver_dir: Optional[str] = None,
        eval_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Split BioMysteryBench problems.csv into the agent for_solver/ view + operator eval/ view - use whenever the user asks to prepare BioMysteryBench (preview or full) for an agentic-biology experiment, build the uniform answer-free tasks.jsonl, or emit the rubric-bearing eval/answers.jsonl + evaluate.py scorer. Replaces the legacy mask step (raw -> for_solver/eval)."""
        from pathlib import Path

        from ...ai_for_science import biomysterybench, resolve_paths

        if raw_dir is None or for_solver_dir is None or eval_dir is None:
            paths = resolve_paths(biomysterybench.BENCHMARK)
            raw_dir = raw_dir or str(paths.raw_dir)
            for_solver_dir = for_solver_dir or str(paths.for_solver_dir)
            eval_dir = eval_dir or str(paths.eval_dir)
        return biomysterybench.standardize(
            raw_dir=Path(raw_dir),
            for_solver_dir=Path(for_solver_dir),
            eval_dir=Path(eval_dir),
        )


__all__ = ["register_ai_for_science_tools"]

# EOF
