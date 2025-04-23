# DuckDBService Examples

This directory contains examples showing how to use the `DuckDBService` with LlamaIndex integration for natural language queries.

## Example Files

1. **duckdb_example.py** - Command-line example showing basic usage
2. **streamlit_example.py** - Interactive web interface for data loading and querying

## Running the Examples

### Prerequisites

Make sure you have installed the project with development dependencies:

```bash
# From the project root directory
make install
```

### Command-line Example

Run the command-line example:

```bash
# From the project root directory
python examples/duckdb_example.py
```

This will:
- Create sample product data
- Load it into DuckDB
- Run a SQL query
- Run natural language queries using both simple and advanced modes

### Streamlit Example

Run the interactive Streamlit example:

```bash
# From the project root directory
streamlit run examples/streamlit_example.py
```

This will start a web interface where you can:
1. Upload data or use sample data
2. Run SQL queries directly
3. Ask natural language questions about your data

## Features Demonstrated

These examples demonstrate:
- Loading data into DuckDB
- Querying with direct SQL
- Converting natural language to SQL using LlamaIndex
- Processing both simple and complex natural language queries

## Implementation Details

The `EnhancedDuckDBService` class (located in `src/assortment_chatbot/services/duckdb_service.py`) uses two different LlamaIndex query engines:

1. `NLSQLTableQueryEngine` - For simpler queries against a single table
2. `SQLTableRetrieverQueryEngine` with `ObjectIndex` - For more complex queries across multiple tables

The service intelligently selects the appropriate query engine based on the complexity parameter. 