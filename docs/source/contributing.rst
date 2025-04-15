Contributing
============

Thank you for your interest in contributing to the CVS Dagster Project. This document outlines the process for contributing.

Development Setup
----------------

Follow the installation instructions, then:

1. Install development dependencies:

   .. code-block:: bash

      uv pip install -e ".[dev]"

2. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Style
----------

This project follows:

* PEP 8 style guide
* Uses type hints (Python 3.10+)
* Uses Ruff for formatting and linting
* Uses Google-style docstrings

Example docstring:

.. code-block:: python

   def my_function(param1: str, param2: int) -> bool:
       """Does something useful with the parameters.

       Args:
           param1: A string parameter description
           param2: An integer parameter description

       Returns:
           A boolean result

       Raises:
           ValueError: If param1 is empty
       """

Testing
-------

Run tests with:

.. code-block:: bash

   pytest
