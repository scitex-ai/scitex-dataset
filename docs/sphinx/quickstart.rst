Quickstart
==========

This guide gets you to a first dataset fetch in under a minute, then
shows the local index and the HuggingFace flow.

Install
-------

.. code-block:: bash

    pip install scitex-dataset
    # Optional extras:
    pip install "scitex-dataset[huggingface]"   # HF Hub access
    pip install "scitex-dataset[mcp]"           # MCP server

Python API
----------

The package exposes per-source ``<src>_fetch`` aliases plus a unified
``filter_results`` helper. Every alias has a matching MCP tool of the
same name (under the ``dataset`` namespace).

.. code-block:: python

    from scitex_dataset import (
        openneuro_fetch, dandi_fetch, hf_search,
        filter_results, list_sources,
        db_build, db_search,
    )

    # 1) Fetch from a catalog source.
    records = openneuro_fetch(max_datasets=200)

    # 2) Filter + rank in memory.
    top = filter_results(
        records, modality="eeg", min_subjects=20,
        sort_by="downloads", limit=10,
    )

    # 3) Search HuggingFace Hub directly.
    hf_hits = hf_search("biology", limit=20)

    # 4) Build the local SQLite + FTS5 index for offline queries.
    db_build()
    db_search("Alzheimer EEG")

    # 5) Inspect the supported sources.
    list_sources()["count"]   # 11

CLI
---

The CLI uses the SciTeX 3-level grammar
``scitex-dataset <domain> <dataset> <action>``:

.. code-block:: bash

    # Catalog fetch
    scitex-dataset neuroscience openneuro fetch -n 50 -o openneuro.json
    scitex-dataset general zenodo fetch -q "neuroscience" -n 20

    # HuggingFace
    scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
    scitex-dataset general huggingface search "biology" -n 10 --json

    # Local index
    scitex-dataset db build
    scitex-dataset db search "Alzheimer EEG"
    scitex-dataset db show-stats

See :doc:`cli_reference` for the full subtree, or run
``scitex-dataset --help-recursive``.

Configuration
-------------

Project scope wins over user scope, per the SciTeX local-state layout:

::

    --config <path>
    $SCITEX_DATASET_CONFIG
    <project>/.scitex/dataset/config.yaml
    $SCITEX_DIR/dataset/config.yaml   (default ~/.scitex/dataset/config.yaml)

Runtime artifacts (the SQLite index, snapshots) live under
``<scope-root>/runtime/``.

Next steps
----------

- :doc:`cli_reference` — full CLI grammar and per-action flags.
- :doc:`sources` — every supported repository.
- :doc:`api/scitex_dataset` — generated Python API reference.
