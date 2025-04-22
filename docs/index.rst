.. Data Chat Assistant documentation master file

================================
Data Chat Assistant Documentation
================================

A Python-based application for chatting with your data using natural language.

.. image:: https://img.shields.io/badge/version-0.1.0-blue.svg
   :target: https://github.com/example/data-chat-assistant
   :alt: Version 0.1.0

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/installation
   user/quickstart
   user/interface

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/app
   api/dashboard
   api/services

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev/architecture
   dev/contributing

Features
========

* Natural language interface to your data
* Support for multiple data formats (CSV, Excel, JSON)
* Interactive data visualizations
* AI-powered data analysis and insights
* Persistent storage for results and chat history

Installation
===========

Install using Docker (recommended):

.. code-block:: bash

   git clone https://github.com/example/data-chat-assistant.git
   cd data-chat-assistant
   docker-compose up -d

Or install directly:

.. code-block:: bash

   git clone https://github.com/example/data-chat-assistant.git
   cd data-chat-assistant
   ./setup.sh

Quick Start
==========

1. Navigate to the dashboard at ``http://localhost:8501``
2. Upload your data file (CSV, Excel, or JSON)
3. Explore data with automatic visualizations
4. Chat with your data using natural language queries

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 