#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_cli/_agentic_screen.py

"""``ai-for-science <bench> screen`` — declarative eligibility screening
(:mod:`..ai_for_science._eligibility`). Kept apart from :mod:`._agentic`
so each file owns one responsibility.
"""

from __future__ import annotations

import json

import click


def make_screen_command(source: str, module, common_options) -> click.Command:
    """Build the ``screen`` command for one benchmark."""

    @click.command("screen")
    @common_options
    @click.option(
        "--config",
        "config_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False),
        help="Eligibility criteria (YAML or JSON): rules, order, target.",
    )
    @click.option(
        "--raw-dir",
        type=click.Path(file_okay=False),
        default=None,
        help="Benchmark raw/ dir (default: <dataset-root>/ai-for-science/<bench>/raw).",
    )
    @click.option(
        "--fetch",
        is_flag=True,
        help="Download a missing unit archive selectively when an archive rule "
        "needs it (default: refuse and say which archive is missing).",
    )
    @click.option(
        "--out",
        type=click.Path(dir_okay=False),
        default=None,
        help="Write the full result (every unit, verdict, reason, facts) as JSON.",
    )
    def _screen_cmd(dataset_root, as_json, config_path, raw_dir, fetch, out):
        """Placeholder docstring (overwritten below with the per-source example)."""
        from ..ai_for_science._eligibility import (
            load_config,
            screen_eligibility,
            write_result,
        )

        try:
            cfg = load_config(config_path)
            if cfg["benchmark"] != module.BENCHMARK:
                raise ValueError(
                    f"config is for {cfg['benchmark']!r}, not {module.BENCHMARK!r}"
                )
            result = screen_eligibility(
                cfg, dataset_root=dataset_root, raw_dir=raw_dir, fetch=fetch
            )
        except Exception as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1)
        if out:
            write_result(result, out)
        if as_json:
            click.echo(json.dumps(result, indent=2, sort_keys=True, default=str))
            return
        click.echo(
            f"Screened {source}: {result['n_units']} units, "
            f"{result['n_candidates']} candidates, selected "
            f"{len(result['selected'])}"
            + (f" of {result['target']}" if result["target"] else "")
        )
        for rec in result["units"]:
            if rec["position"] is None:
                continue
            reason = rec["reason"]["rule"] if rec["reason"] else ""
            click.echo(
                f"  {rec['position']:>3} {rec['label']:<24} {rec['verdict']:<20} {reason}"
            )
        click.echo(f"  counts: {result['counts']}")
        if out:
            click.echo(f"  output: {out}")

    _screen_cmd.help = (
        f"Screen {source} units against declarative eligibility criteria. "
        f"Every unit gets a verdict (selected / excluded / not_screened / "
        f"eligible_not_needed) and, when excluded, the failing rule and fact "
        f"values. Deterministic for the same config, manifests and archives."
        f"\n\n\b\nExample:\n"
        f"  $ scitex-dataset ai-for-science {source} screen "
        f"--config criteria.yaml --out eligibility.json\n"
        f"  $ scitex-dataset ai-for-science {source} screen "
        f"--config criteria.yaml --fetch --json"
    )
    return _screen_cmd


__all__ = ["make_screen_command"]

# EOF
