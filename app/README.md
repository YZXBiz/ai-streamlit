# PandasAI Chat Application

## Application Structure

The application follows a clean, traditional package-based organization:

```
app/
├── components/                # UI components
│   ├── __init__.py            # Exports component interfaces
│   ├── chat_components.py     # Chat interface components
│   └── uploader_components.py # File upload components
│
├── utils/                     # Utility functions
│   ├── __init__.py            # Exports utility functions
│   ├── auth_utils.py          # Authentication utilities
│   └── pandasai_utils.py      # PandasAI integration utilities
│
├── __init__.py                # Application package init
└── main.py                    # Application entry point
```

## Package Structure

The application is organized into packages by their technical role:

1. **Components**: UI elements for the application
   - Chat interface components for displaying and managing the chat
   - File uploader components for handling data file uploads
   
2. **Utils**: Helper functions and utilities
   - Authentication utilities for user login/logout
   - PandasAI integration utilities for data processing and analysis

## Application Flow

1. The `main.py` file serves as the entry point
2. Authentication is handled through `utils/auth_utils.py`
3. File uploads are managed by `components/uploader_components.py`
4. Chat interactions are handled by `components/chat_components.py`
5. PandasAI integration is available in `utils/pandasai_utils.py`

## Adding New Functionality

To extend the application:

1. Add new UI components to the `components` package
2. Add new utility functions to the `utils` package
3. For more complex applications, consider adding:
   - A `services` directory for business logic
   - A `models` directory for data structures 