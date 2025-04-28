# Changelog

## [0.2.0] - 2023-06-01

### Major Changes
- Migrated from PandasAI to pure pandas implementation
- Simplified architecture by removing natural language query capabilities
- Focused the library on DataFrames management and basic analysis

### Added
- New `clear()` method in DataFrameManager to clean up all resources
- Consistent type hints using modern Python 3.10+ syntax
- Makefile with common commands for development
- Proper pyproject.toml with clear dependencies
- Simple demo script in backend/run.py

### Changed
- Simplified PandasAnalyzer class by removing PandasAI dependencies
- Updated DataFrameCollection to work with pandas DataFrames directly
- Updated DataFrameManager to handle pandas DataFrames
- Improved type annotations throughout the codebase
- Streamlined configuration with simpler settings

### Removed
- Removed PandasAI dependencies and imports
- Removed chat and natural language query capabilities
- Removed unused methods and fields (last_code, last_error, last_result)
- Removed SQL timeout settings and other outdated configuration
- Eliminated complex PandasAI-specific code

### Fixed
- Type annotation inconsistencies (using Optional instead of | None)
- Updated tests to match the new implementation 