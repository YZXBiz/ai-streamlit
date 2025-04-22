.. _interface:

User Interface Guide
==================

This guide provides a detailed overview of the Data Chat Assistant user interface.

.. contents:: Contents
   :local:
   :depth: 2

Dashboard Layout
-------------

The dashboard consists of four main sections, accessible via the sidebar navigation:

1. **Home**: Welcome page with overview information
2. **Data Explorer**: Upload and explore data visually
3. **Cluster Analysis**: View clustering results (if available)
4. **AI Chat**: Ask questions about your data

.. image:: ../images/dashboard_layout.png
   :alt: Dashboard layout
   :width: 700px

Home Page
--------

The home page provides:

- A welcome message explaining the application purpose
- Key features of the Data Chat Assistant
- Getting started instructions
- Links to useful resources

Data Explorer
-----------

The Data Explorer page contains three main components:

1. **Data Uploader**: For importing data files
2. **Data Summary**: Statistics and metadata about your dataset
3. **Visualization Tools**: To create charts and graphs

.. image:: ../images/data_explorer.png
   :alt: Data Explorer interface
   :width: 700px

Data Uploader
^^^^^^^^^^^

The data uploader accepts:

- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)

After uploading, you'll see:

- A success message with row and column counts
- A preview of the first 10 rows of data
- Data will be available in other sections of the app

Data Summary
^^^^^^^^^^

The data summary shows:

- **Row count**: Total number of records
- **Column count**: Number of features/variables
- **Missing values**: Count of null/empty cells
- **Duplicate rows**: Number of duplicate records
- **Column information**: Data types, null counts, unique values

This helps you quickly understand your dataset's structure and quality.

Visualization Tools
^^^^^^^^^^^^^^^^

The visualization panel offers:

- **Histogram**: For numeric column distributions
- **Box Plot**: For statistical summaries and outliers
- **Scatter Plot**: For exploring relationships between variables
- **Bar Chart**: For comparing categorical values

Each visualization is interactive - you can hover to see detailed values and use controls to adjust the display.

Cluster Analysis
--------------

This page is available when your data contains a 'CLUSTER' column:

- **Cluster Distribution**: Shows the size of each cluster
- **Cluster Statistics**: Compares metrics across clusters
- **Feature Analysis**: Examines how specific features vary by cluster

.. image:: ../images/cluster_analysis.png
   :alt: Cluster Analysis interface
   :width: 700px

AI Chat Interface
--------------

The AI Chat interface has these components:

- **Chat Input**: Where you type your questions
- **Message History**: Shows the conversation
- **Response Area**: Displays answers and visualizations
- **Clear Button**: Resets the conversation

.. image:: ../images/ai_chat.png
   :alt: AI Chat interface
   :width: 700px

The chat system supports:

- Natural language queries about your data
- Follow-up questions that refer to previous answers
- Questions about trends, patterns, comparisons, and statistics

Example Queries
^^^^^^^^^^^^

You can ask questions like:

- "What is the average value of [column]?"
- "Show me the highest and lowest values in [column]"
- "What's the distribution of [column]?"
- "How does [column1] correlate with [column2]?"
- "Which [category] has the highest [value]?"
- "Show me trends over time for [column]"

Advanced Features
--------------

Keyboard Shortcuts
^^^^^^^^^^^^^^^

- **Enter**: Send message in chat
- **Ctrl+K** or **⌘+K**: Focus on chat input
- **Esc**: Clear current input

Mobile View
^^^^^^^^^

The interface is responsive and works on mobile devices:

- Sidebar collapses into a hamburger menu
- Visualizations resize to fit the screen
- Touch-friendly controls for all interactions

Customizing the Interface
----------------------

You can customize your experience:

1. **Theme Toggle**: Switch between light and dark mode (top-right corner)
2. **Sidebar Width**: Drag the edge to resize
3. **Chart Controls**: Most visualizations have controls to adjust colors, axes, and other properties 