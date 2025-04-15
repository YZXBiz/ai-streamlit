Installation
============

This guide will help you set up and install the CVS Dagster Project on your system.

Requirements
-----------

* Python 3.10 or later
* `uv` package manager (recommended)

Installation Steps
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/cvs-repo-dagster.git
      cd cvs-repo-dagster

2. Set up a virtual environment and install dependencies:

   .. code-block:: bash

      uv venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate
      uv pip install -e .

3. Set up environment variables:

   .. code-block:: bash

      cp .env.example .env
      # Edit .env with your configuration

4. Verify installation:

   .. code-block:: bash

      python -m src.clustering --help
