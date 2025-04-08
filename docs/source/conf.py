# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CVS Dagster Project"
copyright = "2025, Jackson Yang"
author = "Jackson Yang"

version = "0.1"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core extension for API documentation
    "sphinx.ext.viewcode",  # Add links to view source code
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx_autodoc_typehints",  # Better support for type hints
]

# Configure autodoc
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# Configure Napoleon for Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Intersphinx mapping to common Python libraries
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "dagster": ("https://docs.dagster.io/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# Set up sys.path for autodoc to find your modules
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
