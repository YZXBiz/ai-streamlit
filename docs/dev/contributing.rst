.. _contributing:

Contributing Guide
=================

This guide outlines the process for contributing to the Data Chat Assistant project.

.. contents:: Contents
   :local:
   :depth: 2

Getting Started
-------------

1. Fork the repository on GitHub
2. Clone your fork: ``git clone https://github.com/YOUR-USERNAME/chatbot-assortment.git``
3. Set up the development environment: ``./setup.sh``
4. Create a new branch for your feature: ``git checkout -b feature-name``

Development Workflow
------------------

1. Make your changes
2. Run the tests: ``make test``
3. Run the code quality checks: ``make check-all``
4. Fix any issues that arise
5. Submit a pull request

Code Style
---------

This project follows the PEP 8 style guide for Python code. We use the following tools to enforce code style:

- **Ruff** for linting and fixing common issues
- **Black** for code formatting
- **MyPy** for static type checking

You can run all checks with: ``make check-all``

Documentation Standards
---------------------

All code must be thoroughly documented using NumPy-style docstrings. Here's an example of how to format your docstrings:

.. code-block:: python

    def function_name(param1: type1, param2: type2) -> return_type:
        """
        Short description of the function.
        
        Parameters
        ----------
        param1 : type1
            Description of param1
        param2 : type2
            Description of param2
            
        Returns
        -------
        return_type
            Description of return value
            
        Notes
        -----
        Additional information about the function
        
        Examples
        --------
        >>> function_name(1, 'test')
        'example output'
        """
        # Function implementation

Type Annotations
--------------

All new code must include type annotations following Python 3.10+ syntax. Use built-in types for collections rather than importing from ``typing`` when possible.

For example, use:

.. code-block:: python

    def process_items(items: list[str]) -> dict[str, int]:
        # Implementation here
        
Instead of:

.. code-block:: python

    from typing import List, Dict
    
    def process_items(items: List[str]) -> Dict[str, int]:
        # Implementation here

Testing
------

All new features should include tests. We use pytest for testing. To run the tests:

.. code-block:: bash

    make test

To add a new test, create a file in the ``tests/`` directory that follows the naming pattern ``test_*.py``.

Building Documentation
--------------------

To build and preview the documentation locally:

.. code-block:: bash

    make docs

This will generate HTML documentation in the ``docs/_build/html`` directory. Open ``docs/_build/html/index.html`` in a browser to view it.

Submitting a Pull Request
-----------------------

1. Ensure all tests pass and code quality checks succeed
2. Update the documentation if you've added new features
3. Push your branch to your fork: ``git push origin feature-name``
4. Submit a pull request to the main repository
5. Describe your changes in the pull request

Code Review Process
-----------------

Pull requests will be reviewed by maintainers. Changes may be requested before merging. The review process checks:

1. Code quality and style
2. Test coverage
3. Documentation quality
4. Type annotations
5. Overall design and implementation 