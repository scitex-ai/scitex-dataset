#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_rest/__init__.py

"""The dataset REST API and MCP tools, declared once (scitex-sdk ``ApiPlugin``).

Published under the ``scitex.apis`` entry point as ``dataset``; any SciTeX
host serves it at ``/api/v1/dataset/`` through ``scitex_sdk.app.api_serve``
with no dataset-specific host code, and :func:`register_mcp` turns the same
operations into MCP tools named ``dataset_<resource>_<verb>``.

=========================  ======================================  =====================
operation                  REST                                    scope
=========================  ======================================  =====================
dataset_datasets_list      GET  /api/v1/dataset/datasets           dataset:datasets:read
dataset_datasets_get       GET  /api/v1/dataset/datasets/{id}      dataset:datasets:read
dataset_cache_entries_get  GET  /api/v1/dataset/cache-entries/{id} dataset:cache-entries:read
dataset_project_datasets_list    GET  .../project-datasets         dataset:project-datasets:read
dataset_project_datasets_create  POST .../project-datasets         dataset:project-datasets:write
dataset_project_datasets_fetch   POST .../project-datasets/{id}/fetch (202 job)
dataset_curated_datasets_list / _get   .../curated-datasets[/{id}]   dataset:curated-datasets:read
dataset_jobs_get           GET  /api/v1/dataset/jobs/{id}          dataset:jobs:read
=========================  ======================================  =====================

A dataset id is its key ``<repository>:<accession>`` with any ``/`` written
as ``~`` (``huggingface:org~name``), so it is one path segment. Project ids
are ``owner/slug`` and travel as the ``project`` parameter.
"""

from __future__ import annotations

from scitex_sdk.app.api_ops import ApiOperation, build_plugin
from scitex_sdk.app.api_plugin import ApiError, ApiField, ApiSchema

JOB = ApiSchema(
    "Job",
    (
        ApiField("id"),
        ApiField("state"),
        ApiField("percent", "number", False),
        ApiField("message", "string", False),
    ),
)
NOT_FOUND = ApiError("dataset.not_found", 404, "No such dataset")

OPERATIONS = (
    ApiOperation(
        "dataset",
        "datasets",
        "list",
        "scitex_dataset._rest._handlers:search_datasets",
        summary="Search openly available datasets across every supported repository",
        params=(
            ApiField("q", "string", False, "Free-text query, e.g. 'eeg motor imagery'"),
            ApiField("repository", "string", False, "Repository id, e.g. openneuro"),
            ApiField("domain", "string", False, "Domain, e.g. neuroscience"),
            ApiField("modality", "string", False, "Modality, e.g. eeg"),
            ApiField("license", "string", False, "Exact licence string"),
            ApiField("min_size_bytes", "integer", False, "Minimum size in bytes"),
            ApiField("max_size_bytes", "integer", False, "Maximum size in bytes"),
        ),
        paginated=True,
        default_limit=20,
        max_limit=100,
        rate_class="search",
    ),
    ApiOperation(
        "dataset",
        "datasets",
        "get",
        "scitex_dataset._rest._handlers:get_dataset",
        summary="One public dataset: metadata, files, licence, citation, code snippet",
        errors=(NOT_FOUND,),
    ),
    ApiOperation(
        "dataset",
        "cache-entries",
        "get",
        "scitex_dataset._rest._handlers:get_cache_entry",
        summary="Shared-cache state of a dataset: cached, partial or not cached",
        errors=(NOT_FOUND,),
    ),
    ApiOperation(
        "dataset",
        "project-datasets",
        "list",
        "scitex_dataset._rest._handlers:list_project_datasets",
        summary="Public datasets a project uses, with fetch state and the scripts that read them",
        project_scoped=True,
        paginated=True,
        default_limit=50,
        max_limit=200,
    ),
    ApiOperation(
        "dataset",
        "project-datasets",
        "create",
        "scitex_dataset._rest._handlers:record_project_dataset",
        summary="Use a public dataset in a project: pin version, DOI, licence and checksums",
        params=(
            ApiField("dataset", "string", True, "Dataset id, e.g. reference:mnist-idx"),
            ApiField("local_dir", "string", False, "Project folder under data/"),
        ),
        project_scoped=True,
        errors=(NOT_FOUND,),
    ),
    ApiOperation(
        "dataset",
        "project-datasets",
        "fetch",
        "scitex_dataset._rest._handlers:fetch_project_dataset",
        summary="Fetch a dataset into project storage through the shared cache (background job)",
        item=True,
        params=(ApiField("local_dir", "string", False, "Project folder under data/"),),
        response=JOB,
        project_scoped=True,
        errors=(NOT_FOUND,),
        compute_cost="heavy",
    ),
    ApiOperation(
        "dataset",
        "curated-datasets",
        "list",
        "scitex_dataset._rest._handlers:list_curated",
        summary="Curated AI-for-science datasets prepared on this host",
        paginated=True,
        default_limit=20,
        max_limit=100,
    ),
    ApiOperation(
        "dataset",
        "curated-datasets",
        "get",
        "scitex_dataset._rest._handlers:get_curated",
        summary=(
            "A curated dataset: source and version, derivation chain, masking "
            "policies, and counts of held-out content (never the content)"
        ),
        errors=(NOT_FOUND,),
    ),
    ApiOperation(
        "dataset",
        "jobs",
        "get",
        "scitex_dataset._rest._handlers:get_job",
        summary="Progress of a fetch job",
        project_scoped=True,
    ),
)

PLUGIN = build_plugin("dataset", "SciTeX Dataset API", "1", OPERATIONS)


def register_mcp(mcp) -> list:
    """Register every operation as an MCP tool (``dataset_<resource>_<verb>``)."""
    from scitex_sdk.app.api_mcp import register_fastmcp

    return register_fastmcp(mcp, PLUGIN)


__all__ = ["OPERATIONS", "PLUGIN", "register_mcp"]

# EOF
