#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_agentic_solver_project.py

"""``ai-for-science <bench> build-solver-project`` — the guarded assembly
of one prepared capsule into a neutral agent project
(:mod:`..ai_for_science._solver_project`). Kept apart from
:mod:`._agentic` (the preparation/validation verbs) so each file owns one
responsibility.
"""

from __future__ import annotations

import click


def make_solver_project_command(
    source: str, module, common_options, emit
) -> click.Command:
    """Build the ``build-solver-project`` command for one benchmark.

    ``common_options`` / ``emit`` are :mod:`._agentic`'s shared
    ``--dataset-root``/``--json`` decorator and result renderer.
    """

    @click.command("build-solver-project")
    @common_options
    @click.argument("capsule")
    @click.argument("out", type=click.Path())
    @click.option(
        "--mapping",
        required=True,
        type=click.Path(),
        help="Where to write the q<n> -> task_id map (operator side; must lie "
        "outside OUT).",
    )
    @click.option(
        "--dry-run",
        is_flag=True,
        help="Resolve the capsule and report what would be written; write nothing.",
    )
    @click.option(
        "-y",
        "--yes",
        is_flag=True,
        help="Non-interactive (the command never prompts; accepted for scripts).",
    )
    def _solver_project_cmd(dataset_root, as_json, capsule, out, mapping, dry_run, yes):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._solver_project import (
            SolverProjectLeakError,
            prepare_solver_project,
        )

        try:
            result = prepare_solver_project(
                module.BENCHMARK,
                capsule,
                out,
                mapping=mapping,
                dataset_root=dataset_root,
                dry_run=dry_run,
            )
        except SolverProjectLeakError as exc:
            for v in exc.violations:
                click.echo(f"Error: {v['kind']}: {v['path']}", err=True)
            raise SystemExit(2)
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        emit(
            as_json,
            {**result, "output": out},
            f"Solver project for {source} {capsule}.",
        )

    _solver_project_cmd.help = (
        f"Assemble a neutral, guarded solver project from one prepared "
        f"{source} capsule: input/ + questions.json (q1..qN). The task-id map "
        f"goes to --mapping (operator side). Refuses (exit 2) and removes OUT "
        f"if any operator file, eval copy, outside symlink, task identifier or "
        f"answer value is found.\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} build-solver-project "
        f"capsule-001 ./p1 --mapping ./sealed/p1.json\n"
        f"  $ scitex-dataset ai-for-science {source} build-solver-project "
        f"capsule-001 ./p1 --mapping ./sealed/p1.json --dry-run"
    )
    return _solver_project_cmd


__all__ = ["make_solver_project_command"]

# EOF
