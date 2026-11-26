#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_django/apps.py

"""Django AppConfig for the Dataset app.

The SDK base is named through the package attribute path
(``app.embed.ScitexAppConfig``, the scitex-stats / writer reference shape) so
a host's source-only plugin certifier can prove the base. The guard is LOUD:
a fallback to ``django.apps.AppConfig`` would silently drop the manifest and
every SDK contract with it.
"""

try:
    from scitex_sdk import app as _sdk_app
except ImportError as exc:
    raise ImportError(
        "scitex_dataset._django.apps needs scitex-sdk, which is not installed. "
        "Install the optional stack: pip install 'scitex-dataset[server]'"
    ) from exc


class DatasetAppConfig(_sdk_app.embed.ScitexAppConfig):
    name = "scitex_dataset._django"
    label = "scitex_dataset"
    verbose_name = "SciTeX Dataset"
    default = True


# EOF
