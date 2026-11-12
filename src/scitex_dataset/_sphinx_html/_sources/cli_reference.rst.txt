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

The Python helper ``scitex_dataset._config.find_config_path(cli_path=...)``
resolves the following paths (highest first):

1. An explicit ``cli_path`` Python argument
2. ``$SCITEX_DATASET_CONFIG`` env var
3. ``<project>/.scitex/dataset/config.yaml``
4. ``$SCITEX_DIR/dataset/config.yaml`` (default ``~/.scitex/dataset/config.yaml``)

The root CLI does not currently register ``--config``. Catalog commands do not
call ``load_config``. This remains a CLI integration gap: use each command's
explicit flags rather than assuming a config file supplies its options.
The resolver implementation is in ``src/scitex_dataset/_config.py``;
the command registration and catalog caller are in
``src/scitex_dataset/_cli/__init__.py`` and ``_factory.py``.

Project scope wins over user scope. ``SCITEX_DIR`` relocates the user
scope atomically. Runtime files (snapshots, caches, logs) live under
``<scope-root>/runtime/``. The dataset index does not: it lives in the
shared SciTeX store, which ``SCITEX_STORE_DSN`` repoints.

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
============================== ==========================================

The legacy commands print a redirect message and exit with status 2.

Live command tree
-----------------

The full subcommand tree below is rendered directly from the click
group via `sphinx-click <https://sphinx-click.readthedocs.io/>`_. Regenerate
the shipped HTML after CLI changes and check the rendered commands;
a source directive alone does not prove that the bundle is current. Per the SciTeX
Sphinx spec (audit code PS128), hand-written subcommand prose is
forbidden here.

``completion install`` and ``completion status`` are current commands,
not legacy aliases. Their drop-in behavior is covered by
``tests/scitex_dataset/_cli/test__completion.py``.

.. click:: scitex_dataset._cli:main
   :prog: scitex-dataset
   :nested: full
