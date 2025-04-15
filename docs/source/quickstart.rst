Quickstart
==========

This guide will help you quickly get started using the CVS Dagster Project.

Running the Pipeline Locally
---------------------------

1. Start the Dagster UI:

   .. code-block:: bash

      dagster dev

2. Open your browser at http://localhost:3000 to access the Dagster UI.

3. Navigate to the "Deployments" tab and select the appropriate pipeline.

4. Click "Launch Run" to execute the pipeline.

Example Code
-----------

Here's a quick example of how to programmatically run an asset:

.. code-block:: python

   from dagster import materialize
   from src.clustering.dagster.assets import sales_by_category

   # Materialize the asset
   result = materialize([sales_by_category])

   # Check the result
   assert result.success
