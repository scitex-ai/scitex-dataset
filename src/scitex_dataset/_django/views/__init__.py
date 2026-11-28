#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/views/__init__.py

"""Views of the Dataset app — thin: authorize through the host, call the logic.

What the app is for: FIND openly available datasets, see how to cite and use
one, and USE it in a project with its version, checksums and licence pinned.
See :mod:`._host` for the authorization rules.
"""

from ._curated import cache_page, curated_detail, curated_list
from ._project import project_page, use_in_project
from ._public import (
    detail,
    legacy_redirect,
    search,
)

index = search

__all__ = [
    "cache_page",
    "curated_detail",
    "curated_list",
    "detail",
    "index",
    "legacy_redirect",
    "project_page",
    "search",
    "use_in_project",
]

# EOF
