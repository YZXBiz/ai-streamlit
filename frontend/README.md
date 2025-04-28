# PandasAI Frontend

This directory contains the Streamlit frontend for the PandasAI application. The frontend provides a user-friendly interface for interacting with the PandasAI backend, allowing users to upload data, ask questions, and view results.

## Directory Structure

```
frontend/
├── app.py                 # Main entry point for the Streamlit application
├── components/            # Reusable UI components
│   ├── chat_interface.py  # Chat interface for interacting with PandasAI
│   ├── data_preview.py    # Data preview component for viewing datasets
│   ├── header.py          # Header component for the main page
│   └── sidebar.py         # Sidebar component for data management
├── styles/                # CSS styles and theme configurations
│   └── main.py            # Custom CSS styles for the application
├── utils/                 # Utility functions
│   ├── data_loader.py     # Functions for loading data from various sources
│   └── session.py         # Session state management functions
└── tests/                 # Unit tests for the frontend
    ├── test_app.py        # Tests for the main application
    ├── test_data_loader.py # Tests for the data loader utility
    └── test_session.py    # Tests for the session state utility
```

## Running the Frontend

To run the frontend, use the following command:

```bash
make run-frontend
```

This will start the Streamlit application on port 8503.

## Development

### Adding New Components

To add a new component:

1. Create a new file in the `components/` directory
2. Implement the component as a function that takes no arguments
3. Update `app.py` to use the new component

### Adding New Utilities

To add a new utility:

1. Create a new file in the `utils/` directory
2. Implement the utility functions
3. Add tests for the utility functions in the `tests/` directory

### Testing

To run the tests for the frontend:

```bash
pytest frontend/tests
```

## PandasAI v3 Integration

This frontend is designed to work with PandasAI v3, which provides a more streamlined API for data analysis. Key features used include:

- Direct CSV loading with `pai.read_csv()`
- DataFrame API with `pai.DataFrame()`
- Collection API for cross-dataframe analysis with `pai.Collection()`
- Chat interface with `df.chat()` and `collection.chat()`

## Design Principles

The frontend follows these design principles:

1. **Modularity**: Each component is self-contained and has a single responsibility
2. **Testability**: All components and utilities are designed to be easily testable
3. **Simplicity**: The interface is designed to be simple and intuitive
4. **Consistency**: The code follows consistent patterns and naming conventions
