.. _quickstart:

Quickstart Guide
===============

This guide will help you get started with the Data Chat Assistant quickly.

.. contents:: Contents
   :local:
   :depth: 2

Overview
-------

The Data Chat Assistant allows you to:

1. Upload and explore your data
2. Create interactive visualizations
3. Chat with your data using natural language
4. Analyze data with AI-powered insights

Basic Workflow
------------

.. image:: ../images/workflow.png
   :alt: Basic workflow diagram
   :width: 600px

The typical workflow is:

1. **Upload Data**: Start by uploading your data file
2. **Explore Data**: View summaries and basic visualizations
3. **Ask Questions**: Use the chat interface to query your data
4. **Review Results**: Examine the generated visualizations and insights

Uploading Data
------------

1. Navigate to the **Data Explorer** page from the sidebar
2. Click the file upload box or drag and drop your file
3. Supported formats: CSV, Excel (.xlsx/.xls), and JSON
4. After upload, a preview of your data will appear

.. code-block:: none

   Tip: Ensure your data has clean column names without special characters 
   for best results with the natural language queries.

Exploring Your Data
-----------------

Once your data is uploaded, you can:

1. View basic statistics (row count, missing values, etc.)
2. See column information (types, null counts, unique values)
3. Create visualizations using the built-in tools
4. Filter and sort your data

Creating Visualizations
--------------------

The Data Explorer provides several visualization options:

- **Histograms**: View distributions of numeric columns
- **Box Plots**: See statistical summaries and outliers
- **Scatter Plots**: Explore relationships between variables
- **Bar Charts**: Compare categorical values

To create a visualization:

1. Select the visualization type from the dropdown
2. Choose the column(s) to visualize
3. Apply any grouping or filtering options
4. The visualization will automatically update

Chatting with Your Data
--------------------

The AI Chat interface allows you to ask natural language questions about your data:

1. Navigate to the **AI Chat** page from the sidebar
2. Type your question in the chat input box
3. The system will generate an answer, often including visuals
4. You can ask follow-up questions to refine your analysis

Example questions you can ask:

- "What's the average value in the sales column?"
- "Show me the trend of revenue over time"
- "Which region has the highest customer satisfaction?"
- "Compare product categories by profit margin"

Interpreting Results
-----------------

When you ask a question, the system:

1. Converts your question to SQL
2. Executes the query on your data
3. Generates an appropriate visualization
4. Provides a written explanation

You can see the generated SQL by expanding the details section below each answer.

Saving Results
-----------

Currently, your data and chat history are saved in your browser session. 
To preserve important findings:

1. Use browser screenshots to capture visualizations
2. Copy and paste text responses to your preferred notes application

Troubleshooting
-------------

If you encounter issues:

- **Query not understood**: Try rephrasing your question with simpler language
- **Incorrect results**: Check that your data was imported correctly
- **Browser performance issues**: Try with a smaller dataset or refresh the page 