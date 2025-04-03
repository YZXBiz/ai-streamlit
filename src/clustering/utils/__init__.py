"""Utility functions for the clustering pipeline."""

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar

from clustering.utils.common import ensure_directory, get_project_root, timer

__all__ = [
    "ensure_directory",
    "get_project_root",
    "timer",
]

T = TypeVar("T")


class ResourceRegistry:
    """Resource registry using the Singleton pattern.

    Provides centralized resource management to avoid resource leaks
    and enable efficient resource sharing.
    """

    _instance: ClassVar[Optional["ResourceRegistry"]] = None

    @classmethod
    def get_instance(cls) -> "ResourceRegistry":
        """Get or create a ResourceRegistry instance.

        Returns:
            A ResourceRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the ResourceRegistry."""
        self._resources: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}

    def register(self, name: str, resource: Any) -> None:
        """Register a resource.

        Args:
            name: Name of the resource
            resource: The resource to register
        """
        self._resources[name] = resource

    def register_factory(self, name: str, factory: callable) -> None:
        """Register a resource factory.

        A factory is a callable that creates a resource when needed.

        Args:
            name: Name of the resource
            factory: Factory function that creates the resource
        """
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Get a resource.

        If the resource doesn't exist but a factory is registered,
        the factory will be called to create the resource.

        Args:
            name: Name of the resource

        Returns:
            The requested resource

        Raises:
            KeyError: If the resource is not found and no factory is registered
        """
        if name in self._resources:
            return self._resources[name]

        if name in self._factories:
            # Create the resource using the factory
            resource = self._factories[name]()
            self._resources[name] = resource
            return resource

        raise KeyError(f"Resource not found: {name}")

    def get_or_create(self, name: str, factory: callable) -> Any:
        """Get a resource or create it if it doesn't exist.

        Args:
            name: Name of the resource
            factory: Factory function to create the resource if needed

        Returns:
            The requested resource
        """
        if name not in self._resources:
            self._resources[name] = factory()
        return self._resources[name]

    def get_by_type(self, resource_type: Type[T]) -> Dict[str, T]:
        """Get all resources of a specified type.

        Args:
            resource_type: The type of resources to return

        Returns:
            A dictionary of resources of the specified type
        """
        return {name: resource for name, resource in self._resources.items() if isinstance(resource, resource_type)}

    def clear(self) -> None:
        """Clear all resources.

        This should be used with caution, as it may disrupt active connections.
        """
        self._resources.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a resource exists.

        Args:
            name: Name of the resource

        Returns:
            True if the resource exists, False otherwise
        """
        return name in self._resources or name in self._factories
