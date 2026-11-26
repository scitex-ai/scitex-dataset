#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/urls.py

"""URL patterns of the Dataset app; a host mounts them at ``apps/dataset/``."""

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
    path("api/datasets", views.datasets, name="datasets"),
    path("api/upload", views.upload, name="upload"),
    path("api/register", views.register, name="register"),
]

# EOF
