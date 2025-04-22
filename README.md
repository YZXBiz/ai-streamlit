# Data Chat Assistant

A web-based chatbot application for data exploration and analysis with natural language queries.

## Features

- **Data Ingestion**: Upload files (CSV, Excel, JSON, Parquet) or connect to Snowflake
- **Data Visualization**: Interactive exploration with PyWalker
- **AI-powered Chat**: Query your data using natural language
- **Data Persistence**: Save results and sessions with MongoDB

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for package management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-assortment.git
   cd chatbot-assortment
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip sync
   ```

3. Create `.env` file:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` file with your configuration settings

## Usage

### Running Locally

Start the application:

```bash
streamlit run app/main.py
```

### Docker Deployment

Build and run with Docker:

```bash
docker-compose up --build
```

## Development

### Adding Dependencies

Use `uv add` to install packages:

```bash
# Add a single package
uv add streamlit

# Add a development dependency
uv add --dev pytest
```

### Running Tests

```bash
pytest
```

## Project Structure

- `app/`: Main application code
  - `components/`: UI components
  - `services/`: Business logic
  - `models/`: Data models
  - `utils/`: Utilities
- `config/`: Configuration files
- `data/`: Sample data and schemas
- `tests/`: Test files

## License

[MIT License](LICENSE)

