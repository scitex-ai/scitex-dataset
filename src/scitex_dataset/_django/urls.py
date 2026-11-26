#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/urls.py

"""URL patterns of the Dataset app; a host mounts them at ``apps/dataset/``.

Dataset keys are ``<repository>:<accession>`` and an accession may contain
``/`` (``huggingface:org/name``), hence the ``path`` converter.
"""

try:
    from django.urls import path
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.urls needs Django, which is not installed. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc

from . import views

app_name = "scitex_dataset"

urlpatterns = [
    path("", views.index, name="index"),
    path("search/", views.search, name="search"),
    path("project/", views.project_page, name="project"),
    path("curated/", views.curated_list, name="curated"),
    path("curated/<slug:benchmark>/", views.curated_detail, name="curated-detail"),
    path("cache/", views.cache_page, name="cache"),
    path("d/<path:key>", views.detail, name="detail"),
    path("use", views.use_in_project, name="use"),
    # Programs use the REST API at /api/v1/dataset/ (scitex_dataset._rest),
    # served by the host through scitex_sdk.app.api_serve. The 0.15 JSON
    # endpoints answer 410 with that pointer.
    path("api/upload", views.legacy_redirect, name="upload"),
    path("api/register", views.legacy_redirect, name="register"),
    path("api/datasets", views.legacy_redirect, name="datasets-legacy"),
]

# EOF
