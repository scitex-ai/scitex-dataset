CLI Reference
=============

The ``scitex-dataset`` command-line interface follows the SciTeX
3-level noun-verb grammar:

.. code-block:: text

    scitex-dataset <domain> <dataset> <action> [OPTIONS]

See ``general/03_interface_02_cli/02_subcommand-structure-noun-verb`` in
the SciTeX skills for the underlying convention.

Global Options
--------------

.. code-block:: bash

    scitex-dataset --help
    scitex-dataset --version
    scitex-dataset --help-recursive       # Show help for every subcommand
    scitex-dataset --json                 # Emit machine-readable JSON

Domains and Datasets
--------------------

.. code-block:: text

    neuroscience:
      - openneuro       BIDS neuroimaging
      - dandi           NWB neurophysiology
      - physionet       EEG / ECG / waveforms
    general:
      - zenodo          general scientific repository
      - figshare        research data sharing
      - openml          ML datasets
      - huggingface     ML datasets/models (on-demand)
    biology:
      - geo             Gene Expression Omnibus
    pharmacology:
      - moleculenet     molecular ML benchmarks
      - chembl          bioactivity database
    medical:
      - clinicaltrials  clinical study registry

Every dataset exposes a ``fetch`` action with the standard flags
``-n / -o / -v`` (and ``-q`` where the upstream API supports a query).
HuggingFace adds ``search`` / ``info`` / ``download-file``.

Common ``fetch`` flags
~~~~~~~~~~~~~~~~~~~~~~

- ``-n, --max-datasets INT``  Max records (``0`` = no limit, default ``0``)
- ``-o, --output PATH``       Write JSON to file (default: stdout summary)
- ``-v, --verbose``           Per-record progress
- ``-q, --query STRING``      Search query (zenodo / figshare / huggingface)
- ``-b, --batch-size INT``    Batch size (openneuro)

Examples
--------

Catalog fetch (10 sources)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    scitex-dataset neuroscience openneuro fetch -n 50 -o openneuro.json
    scitex-dataset neuroscience dandi fetch -v
    scitex-dataset general zenodo fetch -q "neuroscience" -n 20
    scitex-dataset pharmacology chembl fetch
    scitex-dataset medical clinicaltrials fetch -n 100

HuggingFace
~~~~~~~~~~~

.. code-block:: bash

    scitex-dataset general huggingface fetch Anthropic/BioMysteryBench-full
    scitex-dataset general huggingface search "biology" -n 20 --json
    scitex-dataset general huggingface info Anthropic/BioMysteryBench-full
    scitex-dataset general huggingface download-file Anthropic/BioMysteryBench-full README.md

Local SQLite + FTS5 index
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    scitex-dataset db build                    # index all catalog sources
    scitex-dataset db build -s openneuro -s dandi
    scitex-dataset db search "Alzheimer EEG"
    scitex-dataset db show-stats
    scitex-dataset db clear

Skills
~~~~~~

.. code-block:: bash

    scitex-dataset skills list
    scitex-dataset skills get 04_cli-reference
    scitex-dataset skills install --dry-run

Introspection / completion
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    scitex-dataset list-python-apis
    scitex-dataset mcp list-tools
    eval "$(scitex-dataset print-tab-completion --shell bash)"

Configuration precedence
------------------------

Resolution chain (highest first), per ``general/01_ecosystem_06_local-state-directories``:

1. ``--config <path>`` CLI flag
2. ``$SCITEX_DATASET_CONFIG`` env var
3. ``<project>/.scitex/dataset/config.yaml``
4. ``$SCITEX_DIR/dataset/config.yaml`` (default ``~/.scitex/dataset/config.yaml``)

Project scope wins over user scope. ``SCITEX_DIR`` relocates the user
scope atomically. Runtime files (the local SQLite index, snapshots,
logs) live under ``<scope-root>/runtime/``.

Legacy aliases (hidden, exit code 2)
------------------------------------

For users migrating from the pre-3-level CLI:

============================== ==========================================
Old form                       New form
============================== ==========================================
``fetch-<source> [...]``       ``<domain> <source> fetch [...]``
``<source>`` (bare)            ``<domain> <source> fetch``
``hf <verb>``                  ``general huggingface <verb>``
``hf-<verb>``                  ``general huggingface <verb>``
``db stats``                   ``db show-stats``
``completion``                 ``print-tab-completion``
============================== ==========================================

The legacy commands print a redirect message and exit with status 2.

For the full subtree, run ``scitex-dataset --help-recursive``.
