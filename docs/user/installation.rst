.. _installation:

Installation Guide
================

This guide will walk you through installing and setting up the Data Chat Assistant.

.. contents:: Contents
   :local:
   :depth: 2

Prerequisites
-----------

Before installing, ensure you have:

- **Python 3.10** or higher
- **Git** for cloning the repository
- **Docker** (optional, for containerized deployment)
- **7GB** of free disk space

Option 1: Docker Installation (Recommended)
-----------------------------------------

The easiest way to get started is with Docker, which handles all dependencies automatically.

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/example/data-chat-assistant.git
      cd data-chat-assistant

2. Start the application using Docker Compose:

   .. code-block:: bash

      docker-compose up -d

3. Access the dashboard at http://localhost:8501

4. To stop the application:

   .. code-block:: bash

      docker-compose down

Option 2: Local Installation
-------------------------

For local development or if you don't want to use Docker:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/example/data-chat-assistant.git
      cd data-chat-assistant

2. Run the setup script:

   .. code-block:: bash

      ./setup.sh

   This will:
   - Create necessary directories
   - Install the `uv` package manager if not already installed
   - Install dependencies
   - Create a default .env file

3. Start the application:

   .. code-block:: bash

      make dashboard

4. Access the dashboard at http://localhost:8501

Configuration Options
------------------

You can customize the application by editing the `.env` file:

.. code-block:: bash

   # Dashboard Configuration
   DATA_DIR=./data
   ENV=dev

   # Streamlit Configuration
   STREAMLIT_SERVER_PORT=8501
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

Key configuration options:

- **DATA_DIR**: Where data files are stored
- **ENV**: Environment (dev, test, prod)
- **STREAMLIT_SERVER_PORT**: Port for the web interface

Advanced: MongoDB Setup (Optional)
--------------------------------

For persistent storage beyond the browser session, you can connect to MongoDB:

1. Add MongoDB connection string to your `.env` file:

   .. code-block:: bash

      MONGODB_URI=mongodb://username:password@localhost:27017/chatbot_db

2. Restart the application for changes to take effect

Troubleshooting
-------------

Common installation issues:

1. **Port conflicts**: If port 8501 is already in use, change the port in the `.env` file
2. **Python version**: Ensure you're using Python 3.10+
3. **Dependencies**: If you encounter dependency issues, try running:

   .. code-block:: bash

      uv sync --upgrade

4. **Docker issues**: Ensure Docker and Docker Compose are correctly installed and running

Getting Help
----------

If you encounter issues not covered here:

1. Check the project issues on GitHub
2. Join our community Discord channel
3. Contact the maintainers directly 