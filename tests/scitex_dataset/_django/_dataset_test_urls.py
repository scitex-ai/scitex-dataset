#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/_dataset_test_urls.py

"""The test host's urlconf: the app at the same route the Hub mounts it."""

from django.urls import include, path

from scitex_sdk.app.api_plugin import discover_api_plugins
from scitex_sdk.app.api_serve import api_urlpatterns

# The app where the Hub mounts it, plus EVERY installed scitex.apis plugin
# from a SciTeX distribution, mounted generically at api/v1/<id>/ — the same
# zero-leaf-code line a host adds.
urlpatterns = [
    path("apps/dataset/", include("scitex_dataset._django.urls")),
    *api_urlpatterns(
        [ref for ref in discover_api_plugins() if ref.distribution.startswith("scitex-")]
    ),
]

# EOF
