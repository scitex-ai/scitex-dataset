#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/_dataset_test_host_index.py

"""The test host's catalogue index: a REAL store target nobody listens on.

Opening it fails the way an unbuilt or unreachable host index fails, so the
app's "index unavailable" path is exercised for real, quickly (connection
refused on a loopback port), without touching any shared store.
"""

from __future__ import annotations


def unreachable_index():
    from scitex_dev.store import Store, StoreTarget, WriterPolicy

    from scitex_dataset._index_schema import PKG, SCHEMA

    return Store(
        StoreTarget.postgres(
            "postgresql://nobody@127.0.0.1:9/none?connect_timeout=1",
            pkg=PKG,
            name="index",
        ),
        SCHEMA,
        node="pytest",
        writer_policy=WriterPolicy.MULTI_WRITER,
    )


# EOF
