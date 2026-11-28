#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_rest/test_api_contract.py

"""The pilot leaf passes the SciTeX API contract v1 test (scitex-sdk >= 0.4.0).

Static rules, MCP 1:1, OpenAPI validity, and every operation probed over the
real SDK serving path (401 / 403 / 405 / 400 problem+json). No handler runs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scitex_sdk.testing.api_contract")

from scitex_sdk.testing.api_contract import check_plugin, run_contract_suite  # noqa: E402

from scitex_dataset._rest import PLUGIN  # noqa: E402

TARGET = "scitex_dataset._rest:PLUGIN"


def test_the_dataset_declaration_passes_the_static_contract():
    # Arrange
    plugin = PLUGIN
    # Act
    found = [str(v) for v in check_plugin(plugin)]
    # Assert
    assert found == []


def test_every_dataset_operation_answers_the_contract_envelope():
    # Arrange
    plugin = PLUGIN
    # Act
    found = [str(v) for v in run_contract_suite(plugin, target=TARGET)]
    # Assert
    assert found == []
