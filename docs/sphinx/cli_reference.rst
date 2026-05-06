CLI Reference
=============

The ``scitex-dataset`` command-line interface follows the SciTeX
3-level noun-verb grammar:

.. code-block:: text

    scitex-dataset <domain> <dataset> <action> [OPTIONS]

See ``general/03_interface_02_cli/02_subcommand-structure-noun-verb`` in
the SciTeX skills for the underlying convention.

Configuration precedence
------------------------

Resolution chain (highest first), per
``general/01_ecosystem_06_local-state-directories``:

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

Live command tree
-----------------

The full subcommand tree below is rendered directly from the click
group via `sphinx-click <https://sphinx-click.readthedocs.io/>`_, so it
cannot drift from ``scitex-dataset --help-recursive``. Per the SciTeX
Sphinx spec (audit code PS128), hand-written subcommand prose is
forbidden here.

.. click:: scitex_dataset._cli:main
   :prog: scitex-dataset
   :nested: full
