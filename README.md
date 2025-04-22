# Data Chat Assistant

A web-based chatbot application for data exploration and analysis, powered by PydanticAI and DuckDB.

## Features

- **Data Explorer**: Upload and preview your data files
- **Interactive Visualization**: Create visualizations with PyGWalker 
- **Cluster Analysis**: Visualize clustering results
- **AI Chat**: Chat with your data using natural language

### PydanticAI-Powered Data Chat

The Data Chat Assistant uses PydanticAI to provide a powerful, natural language interface to your data. Key capabilities include:

- **Natural Language to SQL**: Ask questions about your data in plain English, get SQL-powered answers
- **Data Transformation**: Request transformations and manipulations of your data
- **Automated Insights**: Get AI-generated interpretations of query results
- **Result Downloads**: Download the results of your queries in various formats

## Technology Stack

- **Python 3.10+**: Core programming language
- **Streamlit**: Web UI framework
- **Pandas**: Data processing and manipulation
- **DuckDB**: In-memory SQL database for high-performance queries
- **PydanticAI**: Agent framework for AI-powered interactions
- **Plotly**: Interactive data visualizations
- **PyGWalker**: No-code data visualization tool

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/data-chat-assistant.git
cd data-chat-assistant
```

2. Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using uv:

```bash
uv pip install -e .
```

4. Set environment variables:

```bash
cp .env.example .env
# Edit .env to add your API keys
```

## Usage

1. Start the application:

```bash
uv run -m dashboard.app
```

2. Open your browser at http://localhost:8501
3. Upload your data file in the Data Explorer page
4. Navigate to the AI Chat page to start asking questions about your data

## Example Queries

- "Show me the top 5 records by revenue"
- "What's the average sales by region?"
- "Create a new column that calculates profit margin as (revenue - cost) / revenue"
- "Find outliers in the sales data"
- "Compare performance across different regions"

## Configuration

The application can be configured through environment variables:

- `OPENAI_API_KEY`: OpenAI API key for the chat functionality
- `DUCKDB_PATH`: Path to DuckDB database file (default: in-memory)
- `STREAMLIT_SERVER_PORT`: Port for the Streamlit server (default: 8501)

## Development

### Running Tests

```bash
uv run -m pytest
```

### Linting

```bash
uv run -m ruff check .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

