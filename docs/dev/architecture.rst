.. _architecture:

Architecture Overview
====================

This document provides a high-level overview of the Data Chat Assistant architecture.

.. contents:: Contents
   :local:
   :depth: 2

System Components
---------------

The Data Chat Assistant is built with a modular architecture consisting of the following main components:

1. **Dashboard UI** - Built with Streamlit, provides the user interface
2. **LLM Integration** - Connects to AI services for natural language processing
3. **Data Processing** - Handles data loading, processing, and visualization
4. **Query Service** - Converts natural language to SQL queries
5. **Persistence** - Stores conversation history and data metadata

Component Diagram
---------------

.. code-block::

    ┌─────────────────────┐     ┌─────────────────────┐
    │                     │     │                     │
    │    Dashboard UI     │────▶│    LLM Assistant    │
    │    (Streamlit)      │◀────│                     │
    │                     │     │                     │
    └─────────────────────┘     └─────────────────────┘
             ▲  ▼                        ▲
             │  │                        │
             │  │                        │
             │  │     ┌─────────────────────┐
             │  └────▶│                     │
             │        │   Query Service     │
             │        │                     │
             │        └─────────────────────┘
             │                  ▲
             │                  │
    ┌─────────────────────┐     │
    │                     │     │
    │   Data Service      │◀────┘
    │                     │
    │                     │
    └─────────────────────┘
             ▲
             │
    ┌─────────────────────┐
    │                     │
    │     Persistence     │
    │      Service        │
    │                     │
    └─────────────────────┘

Main Data Flow
------------

The typical data flow for a user interaction is:

1. User uploads data through the **Dashboard UI**
2. Data is processed by the **Data Service** and visualized
3. User enters a natural language query in the chat interface
4. The query is sent to the **LLM Assistant** for processing
5. The **Query Service** converts the natural language to SQL
6. **Data Service** executes the SQL query and returns results
7. Results are displayed to the user in the **Dashboard UI**
8. Conversation and results are saved through the **Persistence Service**

Technology Stack
--------------

The application is built with the following core technologies:

- **Python 3.10+** - Core programming language
- **Streamlit** - Web UI framework
- **Pandas** - Data processing and manipulation
- **Plotly** - Interactive data visualizations
- **SQLite/DuckDB** - Local SQL database for queries
- **MongoDB** (optional) - Persistent storage
- **Docker** - Containerization for deployment

Directory Structure
-----------------

.. code-block::

    chatbot-assortment/
    ├── app/                  # Core application code
    │   ├── components/       # Reusable UI components
    │   ├── models/           # Data models
    │   ├── services/         # Backend services
    │   └── utils/            # Utility functions
    ├── dashboard/            # Streamlit dashboard
    │   ├── components/       # Dashboard UI components
    │   └── data/             # Data handling utilities
    ├── data/                 # Data storage
    ├── docs/                 # Documentation
    ├── tests/                # Test suite
    ├── .env.example          # Environment variables template
    ├── docker-compose.yml    # Docker configuration
    ├── Makefile              # Build and development commands
    └── pyproject.toml        # Project dependencies 