# DuckDB to Polars Migration Notes

## Overview

This document outlines the migration of our clustering pipeline from using DuckDB SQL for data transformations to using native Polars operations.

## Completed Changes

1. **Pipeline Assets**: Updated the preprocessing pipeline in `internal.py` to use Polars operations instead of DuckDB SQL

   - Replaced all SQL string generation and queries with declarative Polars operations
   - Removed connection management and error handling related to DuckDB
   - Improved clarity by making data transformations more explicit

2. **Tests**: Updated unit tests to work with the new Polars implementation

   - Removed DuckDB mocking in tests
   - Adjusted test assertions to match the new Polars-based output format

3. **Core Package**: Updated the `clustering.core` package
   - Removed sql_templates from `__init__.py` exports

## Remaining Steps

1. **Clean Up Unused Files**:

   - Run the `cleanup_sql_dependencies.py` script with the `--execute` flag to remove unused files:
     ```
     python cleanup_sql_dependencies.py --execute
     ```

2. **Update Scripts**:

   - Update the following scripts that still use SQL templates:
     - `scripts/debug_columns.py`
     - `scripts/check_transformed_data.py`

3. **Documentation Updates**:

   - Update any documentation that references SQL templates or DuckDB operations
   - Update the development guidelines to prefer Polars for data transformations

4. **Consider Other Components**:
   - Evaluate other parts of the codebase that may still use DuckDB unnecessarily
   - Keep DuckDB for components where it's still beneficial (e.g., Snowflake reader cache)

## Benefits of the Migration

1. **Simpler Code**: No SQL string templates, connection management, or context switching
2. **Enhanced Readability**: Direct data transformations with clear, chainable operations
3. **Better IDE Support**: Full Python code enables better code completion and type checking
4. **Improved Debugging**: Error messages point directly to the code, not SQL strings
5. **Consistent Pattern**: Single approach to data transformations throughout the codebase

## Dependencies

We still need to keep the following DuckDB-related dependencies for now:

- `duckdb`: Used by the Snowflake reader and other components
- `dagster-duckdb-polars`: Used for the IO managers

These can be evaluated for removal in a future cleanup if we decide to fully remove DuckDB.
