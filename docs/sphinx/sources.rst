Data Sources
============

scitex-dataset supports **10 catalog sources** (enumerable, indexable
into the local SQLite + FTS5 cache) plus **HuggingFace Hub**
(on-demand fetch by repo_id; not indexed by default).

Every catalog source exposes the same two callables —
``fetch_all_datasets()`` and ``format_dataset(ds)`` — so they plug
uniformly into ``scitex_dataset.database.build()`` and
``scitex_dataset.search.search_datasets()``. The package also
re-exports each as ``<src>_fetch`` at the top level.

Catalog sources
---------------

OpenNeuro — BIDS neuroimaging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`OpenNeuro <https://openneuro.org>`_ — MRI / fMRI / EEG / MEG / iEEG / PET in BIDS.

.. code-block:: bash

    scitex-dataset neuroscience openneuro fetch -n 50

.. code-block:: python

    from scitex_dataset import openneuro_fetch
    records = openneuro_fetch(max_datasets=50)

DANDI Archive — NWB neurophysiology
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`DANDI <https://dandiarchive.org>`_ — electrophysiology + ophys, NWB-format.

.. code-block:: bash

    scitex-dataset neuroscience dandi fetch -v

.. code-block:: python

    from scitex_dataset import dandi_fetch
    records = dandi_fetch(max_datasets=100)

PhysioNet — physiological waveforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`PhysioNet <https://physionet.org>`_ — ECG, EEG, EMG, MIMIC, sleep / arrhythmia studies.

.. code-block:: bash

    scitex-dataset neuroscience physionet fetch

.. code-block:: python

    from scitex_dataset import physionet_fetch
    records = physionet_fetch()

Zenodo — general scientific data (CERN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Zenodo <https://zenodo.org>`_ — research data, software, publications.

.. code-block:: bash

    scitex-dataset general zenodo fetch -q "neural network" -n 20

.. code-block:: python

    from scitex_dataset import zenodo_fetch
    records = zenodo_fetch(query="neural network", max_datasets=20)

Figshare — research data sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Figshare <https://figshare.com>`_.

.. code-block:: bash

    scitex-dataset general figshare fetch -q "biology"

OpenML — ML datasets and benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`OpenML <https://www.openml.org>`_.

.. code-block:: bash

    scitex-dataset general openml fetch -n 100

MoleculeNet — molecular ML benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`MoleculeNet <https://moleculenet.org>`_ — BBBP, ClinTox, Tox21, ESOL, FreeSolv, …

.. code-block:: bash

    scitex-dataset pharmacology moleculenet fetch

GEO — Gene Expression Omnibus (NCBI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`GEO <https://www.ncbi.nlm.nih.gov/geo/>`_ — RNA-seq, microarray, GSE / GDS series.

.. code-block:: bash

    scitex-dataset biology geo fetch

ChEMBL — bioactivity (EBI)
~~~~~~~~~~~~~~~~~~~~~~~~~~

`ChEMBL <https://www.ebi.ac.uk/chembl/>`_ — IC50 / Ki / EC50 assays.

.. code-block:: bash

    scitex-dataset pharmacology chembl fetch

ClinicalTrials.gov — interventional / observational studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`ClinicalTrials.gov <https://clinicaltrials.gov>`_ — NIH study registry.

.. code-block:: bash

    scitex-dataset medical clinicaltrials fetch

On-demand source
----------------

HuggingFace Hub
~~~~~~~~~~~~~~~

`HuggingFace Hub <https://huggingface.co>`_ has no bounded catalog, so
the standard ``fetch_all_datasets()`` adapter calls
``search_hub(query="", limit=1000)``. Three workflows:

.. code-block:: python

    from scitex_dataset import hf_search, hf_info, hf_fetch

    # 1) Search the live catalog.
    results = hf_search("biology", limit=50)

    # 2) Inspect a specific repo.
    info = hf_info("Anthropic/BioMysteryBench-full")

    # 3) Snapshot-download (gated repos auto-resolve HF_TOKEN /
    #    HF_TOKEN_PATH / ~/.bash.d/secrets/access_tokens/huggingface.txt).
    from scitex_dataset.general.huggingface import fetch_dataset
    path = fetch_dataset(
        "Anthropic/BioMysteryBench-full",
        local_dir="/data/gpfs/projects/punim2354/biomysterybench",
    )

CLI mirror::

    scitex-dataset general huggingface (fetch | search | info | download-file)

To opt HuggingFace into the local index explicitly (capped at 1000 items):

.. code-block:: python

    from scitex_dataset import db_build
    db_build(sources=["huggingface"])

Local Database
--------------

.. code-block:: python

    from scitex_dataset import db_build, db_search, db_stats
    db_build()                    # Build / refresh (catalog sources only)
    results = db_search("EEG")    # Offline FTS5 search
    db_stats()                    # Per-source counts, size, last-build time
