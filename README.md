# Data Analyzer

A framework for managing and analyzing pandas DataFrames.

## Features

- Load data from various sources (CSV, Excel, Parquet, SQL)
- Organize DataFrames with names and descriptions
- Create collections of DataFrames for cross-dataframe analysis
- Define relationships between DataFrames
- Get DataFrame statistics and previews

## Installation

This project requires Python 3.10 or newer.

```bash
# Clone the repository
git clone https://github.com/yourusername/data-analyzer.git
cd data-analyzer

# Install dependencies using UV
make install
```

## Usage

Basic usage example:

```python
from backend import create_analyzer

# Create an analyzer instance
analyzer = create_analyzer()

# Load a CSV file
df = analyzer.load_csv("path/to/data.csv", "my_data", "Description of the data")

# Get a preview of the data
preview = analyzer.get_dataframe_preview("my_data", rows=5)
print(preview)

# Get statistics about the data
stats = analyzer.get_dataframe_stats("my_data")
print(stats)

# Load another DataFrame and create a collection
df2 = analyzer.load_csv("path/to/other_data.csv", "other_data")
collection = analyzer.create_collection(
    ["my_data", "other_data"], 
    "my_collection", 
    "A collection of related data"
)
```

## Development

Run tests:

```bash
make test
```

Run the application:

```bash
make run
```

Clean up temporary files:

```bash
make clean
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Test Coverage Plan

To achieve 100% test coverage for the backend, the following steps should be taken:

1. Fix import issues in the existing tests to make them runnable.
2. Complete tests for all uncovered modules:
   - main.py
   - core/config.py, core/security.py, and core/database/* modules
   - All adapters (storage_local.py, llm_pandasai.py, db_duckdb.py, etc.)
   - API endpoints and schemas
   - All ports (vectorstore.py, llm.py, datasource.py, etc.)
   - Domain models (user.py, datafile.py, dataframe.py, chat_session.py)
   - Services (auth_service.py, file_service.py, chat_service.py, etc.)

3. Follow these patterns for different types of tests:
   - For **Ports**: Test that each port is a Protocol and has all required methods
   - For **Adapters**: Test initialization and all public methods
   - For **Services**: Test all business logic methods using mocked dependencies
   - For **Domain Models**: Test model initialization and validation
   - For **API Endpoints**: Test correct status codes and responses

4. Add GitHub Actions workflow to automatically run tests and report coverage.

### Testing Strategy

- Use `pytest` for all tests
- Use fixtures for common test setup
- Mock external dependencies using unittest.mock
- Use parameterized tests where appropriate
- Aim for at least 95% line coverage and 90% branch coverage

