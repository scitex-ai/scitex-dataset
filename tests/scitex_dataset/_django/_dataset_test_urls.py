#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/_dataset_test_urls.py

"""The test host's urlconf: the app at the same route the Hub mounts it."""

from django.urls import include, path

urlpatterns = [path("apps/dataset/", include("scitex_dataset._django.urls"))]

# EOF
