# Flat Chatbot - Natural Language for Data

A modern, intuitive application for querying data using natural language and SQL with DuckDB and LlamaIndex integration.

![DuckDB Natural Language Query](https://img.shields.io/badge/DuckDB-Natural%20Language%20Query-yellow)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)

## Features

### Core Functionality
- **Natural Language Queries**: Ask questions about your data in plain English
- **SQL Query Interface**: Write and execute raw SQL queries
- **Multiple Data Formats**: Support for CSV, Parquet, JSON, and Excel files
- **Schema Inference**: Automatic detection of column types and structures
- **Smart Query Context**: Uses conversation history for better follow-up questions

### Enhanced User Experience
- **Interactive UI**: Clean, tabbed interface with modern design
- **Real-time Feedback**: Immediate results with progress indicators
- **Data Visualization**: Automatic visualization of query results
- **Export Options**: Download results as CSV, Excel, or JSON
- **Query History**: Track and re-run previous queries
- **Smart Suggestions**: Contextual example queries based on your data

### Technical Features
- **Modular Architecture**: Clean separation of UI, business logic, and services
- **LlamaIndex Integration**: Vector search and embeddings-based query understanding
- **Memory Management**: Efficient handling of large datasets
- **Session Management**: Persistent chat sessions
- **Error Handling**: Helpful suggestions for common errors

## Architecture

The application follows a clean, modular architecture:

```
flat_chatbot/
├── app.py              # Main entry point
├── config.py           # Configuration settings
├── controller.py       # Business logic orchestration
├── logger.py           # Custom logging
├── services/           # Service layer components
│   ├── duckdb_base.py  # Base DuckDB functionality
│   └── duckdb_enhanced.py # Natural language integration
└── ui/                 # User interface components
    ├── components.py   # Reusable UI elements
    ├── history.py      # Chat history display
    ├── query.py        # Query interface
    ├── schema.py       # Schema display
    └── upload.py       # File upload handling
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API Key (for embeddings and language model)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flat-chatbot.git
   cd flat-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

Start the application with:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 in your web browser.

## Usage Guide

### Data Upload

1. Navigate to the "Upload Data" tab
2. Select one or more data files to upload (CSV, Excel, Parquet, or JSON)
3. Optionally provide a custom table name
4. Click "Load Data" to process files

### Querying Data

#### Natural Language Queries

1. Go to the "Ask Questions" tab
2. Type a question in natural language, such as:
   - "What's the average age of users?"
   - "Show me the top 5 countries by total sales"
   - "How many transactions occurred each month?"
3. Select simple or advanced query mode
4. Click "Run Query" to see results

#### SQL Queries

1. Go to the "SQL Query" tab
2. Write your SQL query, such as:
   - `SELECT * FROM users LIMIT 10;`
   - `SELECT country, SUM(sales) FROM transactions GROUP BY country ORDER BY SUM(sales) DESC LIMIT 5;`
3. Click "Execute SQL" to run the query

### Exploring Results

- View the natural language response and data table
- Explore data statistics and visualizations
- Download results in CSV, Excel, or JSON format
- View the generated SQL query for natural language questions

## Configuration

Adjust settings in the `config.py` file:

- `query_timeout`: Maximum seconds for query execution
- `token_limit`: Token limit for conversation memory
- Various path settings for data and logs

## Development

### Testing

Run tests with:

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## Docker Support

Run the application in a Docker container for consistent environment across deployments.

### Building the Docker Image

```bash
docker build -t flat-chatbot:latest .
```

### Running the Container Locally

```bash
docker run -p 8501:8501 -e OPENAI_API_KEY=your_api_key_here flat-chatbot:latest
```

The application will be accessible at http://localhost:8501.

### Environment Variables

Pass environment variables to configure the application:

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_api_key_here \
  -e DEBUG=True \
  flat-chatbot:latest
```

## Deploying to Azure

### Azure Container Registry (ACR)

Deploy to Azure Container Registry using the included script:

1. Update the configuration in `deploy_to_azure.sh`:
   ```bash
   ACR_NAME="your-acr-name"  # Replace with your registry name
   IMAGE_NAME="flat-chatbot"
   IMAGE_TAG="latest"
   ```

2. Run the deployment script:
   ```bash
   chmod +x deploy_to_azure.sh
   ./deploy_to_azure.sh
   ```

### Running on Azure Container Instances (ACI)

After deploying to ACR, create a container instance:

```bash
az container create \
  --resource-group your-resource-group \
  --name flat-chatbot-instance \
  --image your-acr-name.azurecr.io/flat-chatbot:latest \
  --dns-name-label flat-chatbot \
  --ports 8501 \
  --environment-variables OPENAI_API_KEY=your_api_key_here
```

The application will be accessible at http://flat-chatbot.[region].azurecontainer.io:8501.

### Azure App Service

For a more managed solution, deploy to Azure App Service:

```bash
az webapp create \
  --resource-group your-resource-group \
  --plan your-app-service-plan \
  --name your-app-name \
  --deployment-container-image-name your-acr-name.azurecr.io/flat-chatbot:latest
```

Set environment variables through the Azure Portal or CLI:

```bash
az webapp config appsettings set \
  --resource-group your-resource-group \
  --name your-app-name \
  --settings OPENAI_API_KEY=your_api_key_here WEBSITES_PORT=8501
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [LlamaIndex](https://github.com/run-llama/llama_index) for natural language processing
- [DuckDB](https://github.com/duckdb/duckdb) for efficient data querying
- [Streamlit](https://github.com/streamlit/streamlit) for the interactive UI
- [OpenAI](https://openai.com/) for embeddings and language processing

