#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/conftest.py

"""Configure Django as a SciTeX host (hub mode) for the Dataset app tests.

Skipped as a whole when the ``server`` extra (Django + scitex-sdk >= 0.3.26
with the storage gate) is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("django")
_host = pytest.importorskip("scitex_sdk.host")
import importlib.util  # noqa: E402

if not hasattr(_host, "require_storage_write") or not importlib.util.find_spec(
    "scitex_sdk.app.api_serve"
):
    pytest.skip(
        "scitex-sdk without the storage gate / API operation layer (>= 0.3.26)",
        allow_module_level=True,
    )

sys.path.insert(0, str(Path(__file__).resolve().parent))

import django  # noqa: E402
from django.conf import settings  # noqa: E402

#: A store target nothing listens on: the index is UNAVAILABLE, for real.
UNREACHABLE_INDEX = "_dataset_test_host_index.unreachable_index"

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
        LOGIN_URL="/accounts/login/",
        SCITEX_APP_MODE="hub",
        SCITEX_PROJECT_PROVIDER="_dataset_test_host.RoleTableProjectProvider",
        SCITEX_PROJECT_STORAGE="_dataset_test_host.RoleTableProjectStorage",
        SCITEX_STORAGE_GATE="_dataset_test_host.QuotaStorageGate",
        SCITEX_DATASET_INDEX_STORE=UNREACHABLE_INDEX,
        SCITEX_API_CALLER="_dataset_test_host.api_caller",
        SCITEX_API_IDEMPOTENCY_STORE="_dataset_test_host.IDEMPOTENCY",
    )
    django.setup()

import _dataset_test_host as host  # noqa: E402

@pytest.fixture()
def hub(tmp_path, settings):
    """Project ``owner/alpha``: owner, maintainer, read-only member; op is staff.

    ``data/results/metrics.json`` is a project OUTPUT and must never be listed.
    """
    host.reset()
    root = tmp_path / "alpha"
    (root / "data" / "results").mkdir(parents=True)
    (root / "data" / "results" / "metrics.json").write_text('{"acc": 0.9}')
    host.PROJECTS["owner/alpha"] = {
        "root": root,
        "roles": {"owner": "write", "maint": "write", "reader": "read"},
    }
    host.STAFF.add("op")
    host.GATE.update(used=0, quota=10**9, blocked=False)
    settings.SCITEX_DATASET_CACHE_DIR = str(tmp_path / "cache")
    settings.SCITEX_DATASET_ROOT = str(tmp_path / "dataset-root")
    yield root
    host.reset()


@pytest.fixture()
def fixture_reference(loopback_server, loopback_payload):
    """A two-file reference ``fixture-set`` served by the real loopback server.

    Registered in the real registry dict and restored afterwards (no mocks).
    """
    import hashlib

    from scitex_dataset.general import reference as ref

    base, hits = loopback_server
    files = [
        {
            "path": "payload.bin",
            "url": f"{base}/payload.bin",
            "sha256": hashlib.sha256(loopback_payload).hexdigest(),
            "size_bytes": len(loopback_payload),
        }
    ]
    entry = {
        "name": "Fixture payload set",
        "description": "A loopback-served public payload for tests.",
        "files": files,
        "license": "CC0-1.0",
        "citation": "Fixture, T. (2026). A payload. Test Repository.",
    }
    saved = dict(ref.REFERENCES)
    ref.REFERENCES["fixture-set"] = entry
    try:
        yield {"key": "reference:fixture-set", "files": files, "hits": hits}
    finally:
        ref.REFERENCES.clear()
        ref.REFERENCES.update(saved)


def get(client, url, user=None, **params):
    headers = {"HTTP_X_TEST_USER": user} if user else {}
    return client.get(url, params, **headers)


def post(client, url, data, user=None):
    headers = {"HTTP_X_TEST_USER": user} if user else {}
    return client.post(url, data, **headers)


# EOF


def api_get(client, path, token=None, user=None, **params):
    headers = {}
    if token:
        headers["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    if user:
        headers["HTTP_X_TEST_USER"] = user
    return client.get(f"/api/v1/dataset/{path}", params, **headers)


def api_post(client, path, data, token=None, user=None, key="k-1"):
    import json

    headers = {"HTTP_IDEMPOTENCY_KEY": key} if key else {}
    if token:
        headers["HTTP_AUTHORIZATION"] = f"Bearer {token}"
    if user:
        headers["HTTP_X_TEST_USER"] = user
    return client.post(
        f"/api/v1/dataset/{path}", json.dumps(data), content_type="application/json", **headers
    )
