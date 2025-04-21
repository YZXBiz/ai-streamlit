import dagster as dg
from clustering.pipeline.definitions import defs
from dagster import AssetSelection
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting materialization of internal_assign_clusters")
    
    # Get a subset with just the asset we want to debug and its dependencies
    asset_selection = AssetSelection.keys("internal_assign_clusters")
    
    # Materialize the asset
    result = dg.materialize(
        defs.get_subset_build(asset_selection), 
        raise_on_error=False
    )
    
    # Check for failures
    if result.success:
        logger.info("Asset materialized successfully")
    else:
        logger.error("Asset materialization failed")
        for failure in result.get_failure_events():
            logger.error(f"Failure: {failure.message}")
    
    # Print materialization events
    logger.info("Asset materialization events:")
    for event in result.get_asset_materialization_events():
        logger.info(f"Event: {event}")
        
    # Print logs for debugging
    logger.info("Asset logs:")
    for log in result.get_logs():
        if "internal_assign_clusters" in log.message:
            logger.info(f"Log: {log.message}")
            
except Exception as e:
    logger.error(f"Error running debug script: {e}", exc_info=True)
