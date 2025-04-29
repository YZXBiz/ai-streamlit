"""Unit tests for domain models."""

from datetime import datetime

import pandas as pd
import pytest

from backend.app.domain.models.chat_session import ChatSession, Message, MessageSender
from backend.app.domain.models.datafile import DataFile, FileType
from backend.app.domain.models.dataframe import DataFrameCollection
from backend.app.domain.models.user import User


class TestUserModel:
    """Tests for the User domain model."""

    def test_user_creation(self):
        """Test creating a user with default values."""
        user = User(username="testuser", email="test@example.com")

        assert user.id is None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.hashed_password == ""
        assert user.first_name == ""
        assert user.last_name == ""
        assert user.is_active is True
        assert user.is_admin is False
        assert isinstance(user.created_at, datetime)
        assert isinstance(user.updated_at, datetime)

    def test_user_full_constructor(self):
        """Test creating a user with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)

        user = User(
            id=1,
            username="admin",
            email="admin@example.com",
            hashed_password="hashed_password",
            first_name="Admin",
            last_name="User",
            is_active=True,
            is_admin=True,
            created_at=created_at,
            updated_at=updated_at,
        )

        assert user.id == 1
        assert user.username == "admin"
        assert user.email == "admin@example.com"
        assert user.hashed_password == "hashed_password"
        assert user.first_name == "Admin"
        assert user.last_name == "User"
        assert user.is_active is True
        assert user.is_admin is True
        assert user.created_at == created_at
        assert user.updated_at == updated_at

    def test_full_name_method(self):
        """Test the full_name method."""
        # With first and last name
        user1 = User(
            username="user1",
            email="user1@example.com",
            first_name="John",
            last_name="Doe",
        )
        assert user1.full_name == "John Doe"

        # With only username (no first/last name)
        user2 = User(username="user2", email="user2@example.com")
        assert user2.full_name == "user2"

        # With only first name
        user3 = User(username="user3", email="user3@example.com", first_name="Jane")
        assert user3.full_name == "user3"


class TestDataFileModel:
    """Tests for the DataFile domain model."""

    def test_datafile_creation(self):
        """Test creating a data file with default values."""
        data_file = DataFile(
            filename="test.csv",
            original_filename="original.csv",
            file_path="/path/to/file",
            file_type=FileType.CSV,
        )

        assert data_file.id is None
        assert data_file.user_id is None
        assert data_file.filename == "test.csv"
        assert data_file.original_filename == "original.csv"
        assert data_file.file_path == "/path/to/file"
        assert data_file.file_size == 0
        assert data_file.file_type == FileType.CSV
        assert data_file.description == ""
        assert isinstance(data_file.created_at, datetime)
        assert isinstance(data_file.updated_at, datetime)

    def test_datafile_full_constructor(self):
        """Test creating a data file with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)

        data_file = DataFile(
            id=1,
            user_id=2,
            filename="data.csv",
            original_filename="user_data.csv",
            file_path="/path/to/data.csv",
            file_size=1024,
            file_type=FileType.CSV,
            description="Test data file",
            created_at=created_at,
            updated_at=updated_at,
        )

        assert data_file.id == 1
        assert data_file.user_id == 2
        assert data_file.filename == "data.csv"
        assert data_file.original_filename == "user_data.csv"
        assert data_file.file_path == "/path/to/data.csv"
        assert data_file.file_size == 1024
        assert data_file.file_type == FileType.CSV
        assert data_file.description == "Test data file"
        assert data_file.created_at == created_at
        assert data_file.updated_at == updated_at

    def test_file_type_enum(self):
        """Test the FileType enum."""
        assert FileType.CSV.name == "CSV"
        assert FileType.EXCEL.name == "EXCEL"
        assert FileType.PARQUET.name == "PARQUET"
        assert FileType.JSON.name == "JSON"


class TestChatSessionModels:
    """Tests for ChatSession and Message domain models."""

    def test_message_sender_enum(self):
        """Test the MessageSender enum."""
        assert MessageSender.USER.name == "USER"
        assert MessageSender.ASSISTANT.name == "ASSISTANT"

    def test_message_creation(self):
        """Test creating a message with default values."""
        message = Message(content="Hello, world!")

        assert message.id is None
        assert message.session_id is None
        assert message.sender == MessageSender.USER
        assert message.content == "Hello, world!"
        assert isinstance(message.created_at, datetime)

    def test_message_full_constructor(self):
        """Test creating a message with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)

        message = Message(
            id=1,
            session_id=2,
            sender=MessageSender.ASSISTANT,
            content="I can help with that.",
            created_at=created_at,
        )

        assert message.id == 1
        assert message.session_id == 2
        assert message.sender == MessageSender.ASSISTANT
        assert message.content == "I can help with that."
        assert message.created_at == created_at

    def test_chat_session_creation(self):
        """Test creating a chat session with default values."""
        session = ChatSession(name="Test Session")

        assert session.id is None
        assert session.user_id is None
        assert session.name == "Test Session"
        assert session.description == ""
        assert session.data_file_id is None
        assert session.data_file is None
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert isinstance(session.last_active_at, datetime)

    def test_chat_session_full_constructor(self):
        """Test creating a chat session with all parameters."""
        created_at = datetime(2023, 1, 1, 12, 0, 0)
        updated_at = datetime(2023, 1, 2, 12, 0, 0)
        last_active_at = datetime(2023, 1, 3, 12, 0, 0)

        messages = [
            Message(id=1, content="Hello"),
            Message(id=2, content="Hi there", sender=MessageSender.ASSISTANT),
        ]

        data_file = DataFile(
            id=1,
            filename="test.csv",
            original_filename="original.csv",
            file_path="/path/to/file",
            file_type=FileType.CSV,
        )

        session = ChatSession(
            id=1,
            user_id=2,
            name="Data Analysis",
            description="Analysis of sales data",
            data_file_id=1,
            data_file=data_file,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            last_active_at=last_active_at,
        )

        assert session.id == 1
        assert session.user_id == 2
        assert session.name == "Data Analysis"
        assert session.description == "Analysis of sales data"
        assert session.data_file_id == 1
        assert session.data_file is data_file
        assert session.messages == messages
        assert session.created_at == created_at
        assert session.updated_at == updated_at
        assert session.last_active_at == last_active_at

    def test_add_message_method(self):
        """Test the add_message method."""
        session = ChatSession(id=1, name="Test Session")

        # Add a message
        message = session.add_message("Hello", MessageSender.USER)

        assert isinstance(message, Message)
        assert message.content == "Hello"
        assert message.sender == MessageSender.USER
        assert message.session_id == 1
        assert len(session.messages) == 1
        assert session.messages[0] is message

    def test_add_user_message_method(self):
        """Test the add_user_message method."""
        session = ChatSession(id=1, name="Test Session")

        # Add a user message
        message = session.add_user_message("What is the average sales?")

        assert isinstance(message, Message)
        assert message.content == "What is the average sales?"
        assert message.sender == MessageSender.USER
        assert message.session_id == 1
        assert len(session.messages) == 1

    def test_add_assistant_message_method(self):
        """Test the add_assistant_message method."""
        session = ChatSession(id=1, name="Test Session")

        # Add an assistant message
        message = session.add_assistant_message("The average sales is $1,234.56")

        assert isinstance(message, Message)
        assert message.content == "The average sales is $1,234.56"
        assert message.sender == MessageSender.ASSISTANT
        assert message.session_id == 1
        assert len(session.messages) == 1


class TestDataFrameCollection:
    """Tests for the DataFrameCollection model."""

    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for testing."""
        df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        df2 = pd.DataFrame({"id": [10, 20, 30], "value": [100, 200, 300]})
        df3 = pd.DataFrame({"date": ["2023-01-01", "2023-01-02"], "sales": [1000, 2000]})
        return [df1, df2, df3]

    def test_collection_init_with_names(self, sample_dataframes):
        """Test initialization with provided names."""
        names = ["customers", "products", "sales"]
        collection = DataFrameCollection(
            sample_dataframes,
            name="Sales Analysis",
            dataframe_names=names,
            description="Collection for sales analysis",
        )

        assert collection.name == "Sales Analysis"
        assert collection.description == "Collection for sales analysis"
        assert len(collection) == 3
        assert collection.dataframe_names == names
        assert collection.dataframes == sample_dataframes

    def test_collection_init_without_names(self, sample_dataframes):
        """Test initialization without provided names."""
        collection = DataFrameCollection(sample_dataframes, name="Test Collection")

        assert collection.name == "Test Collection"
        assert collection.description == ""
        assert len(collection) == 3
        assert collection.dataframe_names == ["dataframe_0", "dataframe_1", "dataframe_2"]
        assert collection.dataframes == sample_dataframes

    def test_collection_init_with_partial_names(self, sample_dataframes):
        """Test initialization with fewer names than dataframes."""
        names = ["customers", "products"]
        collection = DataFrameCollection(
            sample_dataframes, name="Test Collection", dataframe_names=names
        )

        assert len(collection) == 3
        assert collection.dataframe_names == ["customers", "products", "dataframe_2"]

    def test_add_dataframe(self, sample_dataframes):
        """Test adding a dataframe to the collection."""
        collection = DataFrameCollection(
            sample_dataframes[:2], name="Test Collection", dataframe_names=["df1", "df2"]
        )

        new_df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

        # Add with custom name
        result = collection.add_dataframe(new_df, "df3")
        assert result is True
        assert len(collection) == 3
        assert collection.dataframe_names == ["df1", "df2", "df3"]
        assert collection.dataframes[2] is new_df

        # Add without name (should generate one)
        another_df = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = collection.add_dataframe(another_df)
        assert result is True
        assert len(collection) == 4
        assert collection.dataframe_names == ["df1", "df2", "df3", "dataframe_3"]

        # Try to add with duplicate name (should fail)
        duplicate_df = pd.DataFrame({"c": [9, 10]})
        result = collection.add_dataframe(duplicate_df, "df2")
        assert result is False
        assert len(collection) == 4  # No change

    def test_remove_dataframe(self, sample_dataframes):
        """Test removing a dataframe from the collection."""
        collection = DataFrameCollection(
            sample_dataframes, name="Test Collection", dataframe_names=["df1", "df2", "df3"]
        )

        # Remove existing dataframe
        result = collection.remove_dataframe("df2")
        assert result is True
        assert len(collection) == 2
        assert collection.dataframe_names == ["df1", "df3"]
        assert collection.dataframes == [sample_dataframes[0], sample_dataframes[2]]

        # Try to remove non-existent dataframe
        result = collection.remove_dataframe("doesnt_exist")
        assert result is False
        assert len(collection) == 2  # No change

    def test_get_dataframe(self, sample_dataframes):
        """Test getting a dataframe by name."""
        collection = DataFrameCollection(
            sample_dataframes, name="Test Collection", dataframe_names=["df1", "df2", "df3"]
        )

        # Get existing dataframe
        df = collection.get_dataframe("df2")
        assert df is sample_dataframes[1]

        # Try to get non-existent dataframe
        df = collection.get_dataframe("doesnt_exist")
        assert df is None

    def test_get_dataframes(self, sample_dataframes):
        """Test getting all dataframes."""
        collection = DataFrameCollection(sample_dataframes, name="Test Collection")

        dfs = collection.get_dataframes()
        assert dfs == sample_dataframes

    def test_get_query_context(self, sample_dataframes):
        """Test getting query context."""
        collection = DataFrameCollection(
            sample_dataframes,
            name="Sales Analysis",
            dataframe_names=["customers", "products", "sales"],
            description="Collection for sales analysis",
        )

        context = collection.get_query_context()
        assert "Collection 'Sales Analysis'" in context
        assert "Collection for sales analysis" in context
        assert "Contains 3 dataframes" in context
        assert "customers, products, sales" in context
