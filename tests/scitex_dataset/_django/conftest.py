#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/conftest.py

"""Configure Django as a SciTeX host (hub mode) for the Dataset app tests.

Skipped as a whole when the ``server`` extra (Django + scitex-sdk) is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("django")
pytest.importorskip("scitex_sdk.host")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import django  # noqa: E402
from django.conf import settings  # noqa: E402

if not settings.configured:
    settings.configure(
        DEBUG=False,
        SECRET_KEY="dataset-app-tests",  # pragma: allowlist secret
        ALLOWED_HOSTS=["testserver"],
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.staticfiles",
            "scitex_sdk.ui",
            "scitex_sdk.app",
            "scitex_dataset._django.apps.DatasetAppConfig",
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "_dataset_test_host.HeaderUserMiddleware",
        ],
        ROOT_URLCONF="_dataset_test_urls",
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.request",
                        "scitex_sdk.app._app_scope.app_scope_context",
                    ]
                },
            }
        ],
        DATABASES={},
        STATIC_URL="/static/",
        USE_TZ=True,
        SCITEX_APP_MODE="hub",
        SCITEX_PROJECT_PROVIDER="_dataset_test_host.RoleTableProjectProvider",
        SCITEX_PROJECT_STORAGE="_dataset_test_host.RoleTableProjectStorage",
    )
    django.setup()

import _dataset_test_host as host  # noqa: E402


@pytest.fixture()
def hub(tmp_path):
    """One project ``owner/alpha`` with an owner, a maintainer and a read-only member."""
    host.reset()
    root = tmp_path / "alpha"
    (root / "data").mkdir(parents=True)
    (root / "data" / "a.csv").write_bytes(b"x,y\n1,2\n")
    host.PROJECTS["owner/alpha"] = {
        "root": root,
        "roles": {"owner": "write", "maint": "write", "reader": "read"},
    }
    yield root
    host.reset()


# EOF
