#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_ai_for_science.py

"""MCP tools for the ``ai-for-science`` benchmark cohorts.

One ``dataset_<bench>_mask`` tool per benchmark — the read-only,
network-free pure-Python mask is the variant safe to expose over MCP.
``download`` and ``prepare`` are side-effecting (multi-GB pulls, writes
to operator-private directories), so they stay CLI-only on purpose:
the right call site for the heavy verbs is a ``sbatch`` job, not an MCP
client.

Each ``mask`` tool reads the operator-staged oracle artifact under
``oracle_dir`` and writes the masked agent-visible questions file under
``benchmark_dir``. Both directories default to the canonical
``$SCITEX_ORACLES_ROOT / $SCITEX_DATASET_ROOT`` layout when not given.
"""

from typing import Any, Dict, Optional


def register_ai_for_science_tools(mcp) -> None:
    """Register ``dataset_<corebench|bixbench|biomysterybench>_mask`` tools."""

    @mcp.tool()
    def corebench_mask(
        oracle_dir: Optional[str] = None,
        benchmark_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mask CORE-Bench oracle JSONs and emit the agent-visible questions.json - use whenever the user asks to prepare CORE-Bench for an agentic-reproducibility experiment, null out CodeOcean answer values + paper-identity (capsule_title / capsule_doi), or rebuild the 90-record merged train+test questions file the SAC capsule binds at /questions:ro. Drop-in replacement for the legacy paper-scitex-clew/scripts/cohorts/a_corebench/dataset/mask_questions.py."""
        from pathlib import Path

        from ...ai_for_science import corebench, resolve_paths

        if oracle_dir is None or benchmark_dir is None:
            paths = resolve_paths(corebench.COHORT_DIR)
            oracle_dir = oracle_dir or str(paths.oracle_dir)
            benchmark_dir = benchmark_dir or str(paths.benchmark_dir)
        return corebench.mask(
            oracle_dir=Path(oracle_dir), benchmark_dir=Path(benchmark_dir)
        )

    @mcp.tool()
    def bixbench_mask(
        oracle_dir: Optional[str] = None,
        benchmark_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mask BixBench oracle JSONL and emit the agent-visible questions.jsonl - use whenever the user asks to prepare BixBench for an agentic-bioinformatics experiment, null out answer/ideal/result/distractors/paper while preserving hypothesis + canary, or rebuild the 205-record agent-visible questions file. Drop-in replacement for the legacy paper-scitex-clew/scripts/cohorts/b_bixbench/dataset/mask_questions.py."""
        from pathlib import Path

        from ...ai_for_science import bixbench, resolve_paths

        if oracle_dir is None or benchmark_dir is None:
            paths = resolve_paths(bixbench.COHORT_DIR)
            oracle_dir = oracle_dir or str(paths.oracle_dir)
            benchmark_dir = benchmark_dir or str(paths.benchmark_dir)
        return bixbench.mask(
            oracle_dir=Path(oracle_dir), benchmark_dir=Path(benchmark_dir)
        )

    @mcp.tool()
    def biomysterybench_mask(
        oracle_dir: Optional[str] = None,
        benchmark_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Mask BioMysteryBench problems.csv and emit the agent-visible questions.jsonl - use whenever the user asks to prepare BioMysteryBench (preview or full) for an agentic-biology experiment, null out the answer_rubric column while preserving id/question/allowed_domains/human_solvable, or rebuild the masked questions file the SAC capsule reads. Drop-in replacement for the legacy paper-scitex-clew/scripts/cohorts/c_biomysterybench/dataset/mask_questions.py."""
        from pathlib import Path

        from ...ai_for_science import biomysterybench, resolve_paths

        if oracle_dir is None or benchmark_dir is None:
            paths = resolve_paths(biomysterybench.COHORT_DIR)
            oracle_dir = oracle_dir or str(paths.oracle_dir)
            benchmark_dir = benchmark_dir or str(paths.benchmark_dir)
        return biomysterybench.mask(
            oracle_dir=Path(oracle_dir), benchmark_dir=Path(benchmark_dir)
        )


__all__ = ["register_ai_for_science_tools"]

# EOF
