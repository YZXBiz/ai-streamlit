# Assortment Chatbot

A web-based chatbot application for data exploration and analysis, powered by PydanticAI and DuckDB.

## 📊 Features

- **Data Explorer**: Upload and preview your data files
- **Interactive Visualization**: Create visualizations with PyGWalker 
- **Cluster Analysis**: Visualize clustering results
- **AI Chat**: Chat with your data using natural language

### 🔍 PydanticAI-Powered Data Chat

The Assortment Chatbot uses PydanticAI to provide a powerful, natural language interface to your data. Key capabilities include:

- **Natural Language to SQL**: Ask questions about your data in plain English, get SQL-powered answers
- **Data Transformation**: Request transformations and manipulations of your data
- **Automated Insights**: Get AI-generated interpretations of query results
- **Result Downloads**: Download the results of your queries in various formats

## 🏗️ Project Structure

The project follows a clean, modular architecture:

```
assortment_chatbot/
├── core/               # Core application logic
├── utils/              # Utility functions and helpers
├── config/             # Configuration management
├── api/                # API endpoints
├── services/           # External service integrations
├── models/             # Data models
├── ui/                 # User interface components
│   ├── components/     # Reusable UI components
│   │   ├── data/       # Data-related components
│   │   ├── chat/       # Chat interface components
│   │   └── visualization/ # Visualization components
│   └── pages/          # Full pages
├── logs/               # Log files
├── data/               # Data storage
├── tests/              # Test suite
├── main.py             # Application entry point
├── pyproject.toml      # Project configuration
└── README.md           # Project documentation
```

## 🛠️ Technology Stack

- **Python 3.10+**: Core programming language
- **Streamlit**: Web UI framework
- **Pandas**: Data processing and manipulation
- **DuckDB**: In-memory SQL database for high-performance queries
- **PydanticAI**: Agent framework for AI-powered interactions
- **Plotly**: Interactive data visualizations
- **PyGWalker**: No-code data visualization tool
- **Loguru**: Modern logging system
- **Azure Key Vault**: Secure secret management

## 📋 Installation

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/assortment-chatbot.git
cd assortment-chatbot
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

## 🚀 Usage

1. Start the application:

```bash
python main.py
```

2. Open your browser at http://localhost:8501
3. Upload your data file in the Data Explorer page
4. Navigate to the AI Chat page to start asking questions about your data

## 💬 Example Queries

- "Show me the top 5 records by revenue"
- "What's the average sales by region?"
- "Create a new column that calculates profit margin as (revenue - cost) / revenue"
- "Find outliers in the sales data"
- "Compare performance across different regions"

## ⚙️ Configuration

The application can be configured through environment variables:

- `OPENAI_API_KEY`: OpenAI API key for the chat functionality
- `DUCKDB_PATH`: Path to DuckDB database file (default: in-memory)
- `STREAMLIT_SERVER_PORT`: Port for the Streamlit server (default: 8501)

### Logging Configuration

The application uses Loguru for comprehensive logging. Configure logging behavior with these environment variables:

- `LOG_LEVEL`: Minimum log level to capture (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
- `LOG_DIR`: Directory to store log files (default: logs)
- `LOG_RETENTION`: How long to keep log files (default: 7 days)
- `LOG_ROTATION`: When to rotate log files (default: 100 MB)
- `LOG_JSON`: Whether to output logs in JSON format (True/False)
- `LOG_CONSOLE`: Whether to output logs to console (True/False)

### Azure Key Vault Integration

For secure secret management, you can use Azure Key Vault:

- `KEY_VAULT_URL`: Azure Key Vault URL
- `KEY_VAULT_ENABLED`: Whether to use Azure Key Vault for secrets
- `USE_MANAGED_IDENTITY`: Whether to use managed identity authentication
- `CLIENT_ID`: Azure client ID for Key Vault access
- `CLIENT_SECRET`: Azure client secret for Key Vault access
- `TENANT_ID`: Azure tenant ID for Key Vault access

## 🧪 Development

### Running Tests

```bash
uv run -m pytest
```

### Linting

```bash
uv run -m ruff check .
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details. 