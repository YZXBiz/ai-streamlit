"""Common test fixtures for preprocessing tests."""


class MockReader:
    """Mock reader for testing."""

    def __init__(self, data):
        """Initialize the reader with test data.

        Args:
            data: Test data to return
        """
        self.data = data

    def read(self):
        """Return the test data.

        Returns:
            The test data
        """
        return self.data


class MockWriter:
    """Mock writer for testing."""

    def __init__(self, requires_pandas=False):
        """Initialize the writer.

        Args:
            requires_pandas: Whether the writer requires pandas DataFrames
        """
        self.requires_pandas = requires_pandas
        self.written_data = None

    def write(self, data):
        """Store the written data.

        Args:
            data: The data to write
        """
        self.written_data = data
