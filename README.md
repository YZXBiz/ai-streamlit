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

