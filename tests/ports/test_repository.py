"""Tests for repository interfaces."""

from abc import abstractmethod

import pytest
from app.ports.repository import (
    ChatSessionRepository,
    DataFileRepository,
    Repository,
    UserRepository,
)


def test_repository_is_abstract():
    """Test that Repository is an abstract base class."""
    # Trying to instantiate Repository should raise TypeError
    with pytest.raises(TypeError):
        Repository()


def test_repository_abstract_methods():
    """Test that Repository has the expected abstract methods."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(Repository)
        if getattr(getattr(Repository, method_name), "__isabstractmethod__", False)
    ]

    # Check that the expected methods are in the abstract methods
    expected_methods = ["get", "create", "update", "delete", "list_items"]
    for method in expected_methods:
        assert method in abstract_methods


def test_user_repository_is_abstract():
    """Test that UserRepository is an abstract base class."""
    # Trying to instantiate UserRepository should raise TypeError
    with pytest.raises(TypeError):
        UserRepository()


def test_user_repository_additional_methods():
    """Test that UserRepository has additional abstract methods beyond Repository."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(UserRepository)
        if getattr(getattr(UserRepository, method_name), "__isabstractmethod__", False)
    ]

    # Check that the additional methods are in the abstract methods
    expected_methods = ["get_by_username", "get_by_email"]
    for method in expected_methods:
        assert method in abstract_methods


def test_data_file_repository_is_abstract():
    """Test that DataFileRepository is an abstract base class."""
    # Trying to instantiate DataFileRepository should raise TypeError
    with pytest.raises(TypeError):
        DataFileRepository()


def test_data_file_repository_additional_methods():
    """Test that DataFileRepository has additional abstract methods beyond Repository."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(DataFileRepository)
        if getattr(getattr(DataFileRepository, method_name), "__isabstractmethod__", False)
    ]

    # Check that the additional methods are in the abstract methods
    expected_methods = ["get_by_user"]
    for method in expected_methods:
        assert method in abstract_methods


def test_chat_session_repository_is_abstract():
    """Test that ChatSessionRepository is an abstract base class."""
    # Trying to instantiate ChatSessionRepository should raise TypeError
    with pytest.raises(TypeError):
        ChatSessionRepository()


def test_chat_session_repository_additional_methods():
    """Test that ChatSessionRepository has additional abstract methods beyond Repository."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(ChatSessionRepository)
        if getattr(getattr(ChatSessionRepository, method_name), "__isabstractmethod__", False)
    ]

    # Check that the additional methods are in the abstract methods
    expected_methods = [
        "get_by_user",
        "get_session_with_file",
        "add_message",
        "get_messages",
    ]
    for method in expected_methods:
        assert method in abstract_methods


class ConcreteRepositoryForTesting(Repository):
    """A concrete implementation of Repository for testing."""

    def get(self, entity_id):
        return None

    def create(self, entity):
        return entity

    def update(self, entity):
        return entity

    def delete(self, entity_id):
        return True

    def list_items(self, skip=0, limit=100):
        return []

    def get_all(self):
        return []


def test_concrete_repository():
    """Test that a concrete implementation of Repository can be instantiated."""
    repo = ConcreteRepositoryForTesting()
    assert isinstance(repo, Repository)
    assert repo.get(1) is None
    assert repo.create("entity") == "entity"
    assert repo.update("entity") == "entity"
    assert repo.delete(1) is True
    assert repo.list_items() == []
    assert repo.get_all() == []
