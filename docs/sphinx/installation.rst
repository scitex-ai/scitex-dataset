Installation
============

Requirements
------------

- Python 3.10 or higher
- pip (Python package installer)

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

    pip install scitex-dataset

Development Installation
------------------------

Clone and install in development mode:

.. code-block:: bash

    git clone https://github.com/ywatanabe1989/scitex-dataset.git
    cd scitex-dataset
    pip install -e ".[dev]"

Optional Dependencies
---------------------

Install with all optional features:

.. code-block:: bash

    pip install scitex-dataset[all]

MCP server support:

.. code-block:: bash

    pip install scitex-dataset[mcp]

HuggingFace Hub support:

.. code-block:: bash

    pip install scitex-dataset[huggingface]

SciTeX integration:

.. code-block:: bash

    pip install scitex-dataset[scitex]

Verification
------------

Verify the installation:

.. code-block:: bash

    scitex-dataset --version
    scitex-dataset sources
