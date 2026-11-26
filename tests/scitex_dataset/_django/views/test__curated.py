#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: tests/scitex_dataset/_django/test_views_curated.py

"""Curated AI-for-science datasets: the masking boundary, per role.

The held-out answer below must never appear in any response, for anyone;
held-out FILE NAMES appear for operators only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import get

SECRET = "HELD-OUT-ANSWER-7d41"
URL = "/apps/dataset/curated/bixbench/"


@pytest.fixture()
def curated_tree(hub, settings):
    from scitex_dataset.ai_for_science import resolve_paths, write_manifest

    paths = resolve_paths("bixbench", dataset_root=settings.SCITEX_DATASET_ROOT)
    for directory in (paths.raw_dir, paths.for_solver_dir, paths.eval_dir):
        directory.mkdir(parents=True)
    (paths.raw_dir / "upstream.jsonl").write_text(f'{{"q": "x", "a": "{SECRET}"}}\n')
    tasks = paths.for_solver_dir / "tasks.jsonl"
    tasks.write_text('{"task_id": "t1", "prompt": "x"}\n')
    (paths.eval_dir / "answers.jsonl").write_text(
        f'{{"task_id": "t1", "a": "{SECRET}"}}\n'
    )
    write_manifest(
        manifest_dir=paths.manifest_dir,
        id="bixbench",
        name="BixBench",
        version="v1.5",
        source_url="https://huggingface.co/datasets/futurehouse/BixBench",
        benchmark="bixbench",
        tracked_paths=[tasks],
        tracked_root=paths.root,
        prepared_at="2026-11-27T00:00:00Z",
    )
    return paths


def test_anonymous_curated_page_redirects_to_login(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url)
    # Assert
    assert response.status_code == 302


def test_member_sees_the_source_version(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "reader")
    # Assert
    assert b"v1.5" in response.content


def test_member_sees_the_masking_policies(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "reader")
    # Assert
    assert b"precision-aware-values-v1" in response.content


def test_member_never_sees_held_out_file_names(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "reader")
    # Assert
    assert b"answers.jsonl" not in response.content


def test_member_never_sees_held_out_content(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "reader")
    # Assert
    assert SECRET.encode() not in response.content


def test_member_api_omits_the_held_out_listing(client, curated_tree):
    # Arrange
    url = "/api/v1/dataset/curated-datasets/bixbench"
    # Act
    response = get(client, url, "reader")
    # Assert
    assert response.json()["data"]["held_out"]["listing"] is None


def test_operator_sees_held_out_file_names(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "op")
    # Assert
    assert b"answers.jsonl" in response.content


def test_operator_never_sees_held_out_content(client, curated_tree):
    # Arrange
    url = URL
    # Act
    response = get(client, url, "op")
    # Assert
    assert SECRET.encode() not in response.content


def test_operator_api_never_carries_content(client, curated_tree):
    # Arrange
    url = "/api/v1/dataset/curated-datasets/bixbench"
    # Act
    response = get(client, url, "op")
    # Assert
    assert SECRET.encode() not in response.content


def test_no_route_serves_a_held_out_file(client, curated_tree):
    # Arrange
    url = URL + "eval/answers.jsonl"
    # Act
    response = get(client, url, "op")
    # Assert
    assert response.status_code == 404


def test_agent_view_files_are_pinned_by_sha256(client, curated_tree):
    # Arrange
    url = "/api/v1/dataset/curated-datasets/bixbench"
    # Act
    response = get(client, url, "reader")
    # Assert
    assert [
        f["path"] for f in response.json()["data"]["agent_visible"]["tracked"]
    ] == ["for_solver/tasks.jsonl"]


def test_unprepared_curated_dataset_is_404(client, hub):
    # Arrange
    url = "/apps/dataset/curated/corebench/"
    # Act
    response = get(client, url, "reader")
    # Assert
    assert response.status_code == 404


# --- shared cache administration ---------------------------------------------


def test_non_operator_cache_page_is_403(client, hub):
    # Arrange
    url = "/apps/dataset/cache/"
    # Act
    response = get(client, url, "owner")
    # Assert
    assert response.status_code == 403


def test_operator_sees_the_cache_page(client, hub):
    # Arrange
    url = "/apps/dataset/cache/"
    # Act
    response = get(client, url, "op")
    # Assert
    assert b"Shared dataset cache" in response.content


def test_anonymous_cache_page_redirects_to_login(client, hub):
    # Arrange
    url = "/apps/dataset/cache/"
    # Act
    response = get(client, url)
    # Assert
    assert response.status_code == 302


def test_curated_tree_raw_dir_is_unchanged_by_viewing(client, curated_tree):
    # Arrange
    before = sorted(p.name for p in Path(curated_tree.raw_dir).iterdir())
    # Act
    get(client, URL, "op")
    # Assert
    assert sorted(p.name for p in Path(curated_tree.raw_dir).iterdir()) == before


# EOF
