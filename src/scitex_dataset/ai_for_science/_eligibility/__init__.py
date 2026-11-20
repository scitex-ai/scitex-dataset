#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/ai_for_science/_eligibility/__init__.py

"""Eligibility screening: declarative criteria over dataset-blind unit facts.

Which capsules/tasks qualify for an experiment is a CONFIG (see
:mod:`._config`), not code. Per-benchmark adapters (:mod:`._adapters`)
only supply facts — split, declared language, question counts, archive
size, and archive-derived languages, frameworks, GPU need and entry — and
the engine (:mod:`._engine`) applies the config the same way for every
benchmark, returning a deterministic, ordered list in which every unit has
a verdict and, when excluded, the rule and fact values behind it.

    from scitex_dataset.ai_for_science import screen_eligibility, write_result
    result = screen_eligibility("criteria.yaml", raw_dir=".../corebench/raw", fetch=True)
    write_result(result, "eligibility.json")
    result["selected"]          # ordered native ids -> prepare(only=...)
"""

from ._adapters import ADAPTERS, ArchiveUnavailableError
from ._archive import read_archive_tree
from ._code import FIGURE_QUESTION_PATTERN, is_figure_question
from ._config import CONFIG_SCHEMA, FACTS, EligibilityConfigError, load_config
from ._engine import RESULT_SCHEMA, screen_eligibility, write_result
from ._facts import analyze_tree

__all__ = [
    "ADAPTERS",
    "CONFIG_SCHEMA",
    "FACTS",
    "FIGURE_QUESTION_PATTERN",
    "RESULT_SCHEMA",
    "ArchiveUnavailableError",
    "EligibilityConfigError",
    "analyze_tree",
    "is_figure_question",
    "load_config",
    "read_archive_tree",
    "screen_eligibility",
    "write_result",
]

# EOF
