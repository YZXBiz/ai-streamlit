"""IO manager for the clustering pipeline."""

from typing import Any

import dagster as dg
from dagster import InputContext, IOManager, OutputContext


class InMemoryIOManager(IOManager):
    """A simple in-memory IO manager that stores all data in memory."""

    def __init__(self):
        """Initialize the in-memory IO manager."""
        self.in_memory_store: dict[str, Any] = {}

    def _get_key(self, context: InputContext | OutputContext) -> str:
        """Get the storage key for an asset.

        Args:
            context: The context object

        Returns:
            A string key for storage
        """
        if context.asset_key:
            return context.asset_key.to_string()
        elif hasattr(context, "step_key") and hasattr(context, "name"):
            return f"{context.step_key}.{context.name}"
        elif hasattr(context, "upstream_output"):
            return f"{context.upstream_output.step_key}.{context.upstream_output.name}"
        else:
            return "default"

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Store output in memory.

        Args:
            context: The output context
            obj: The output value
        """
        key = self._get_key(context)
        self.in_memory_store[key] = obj
        context.log.info(f"Saved output to in-memory store with key {key}")

    def load_input(self, context: InputContext) -> Any:
        """Load input from memory.

        Args:
            context: The input context

        Returns:
            The input value
        """
        key = self._get_key(context)

        if key not in self.in_memory_store:
            raise KeyError(f"Key '{key}' not found in in-memory store")

        context.log.info(f"Loading input from in-memory store with key {key}")
        return self.in_memory_store[key]


@dg.io_manager
def clustering_io_manager() -> IOManager:
    """Factory function for the in-memory IO manager.

    Args:
        _: The context for initializing the resource (unused)

    Returns:
        A configured IO manager
    """
    return InMemoryIOManager()
