# Error Handling and Logging

This document describes the error handling and logging approach used in the Data Chat Assistant application.

## Logging System

The application uses a custom logging implementation based on Python's standard `logging` module, enhanced with MDC (Mapped Diagnostic Context) capabilities similar to those found in Java's logging frameworks.

### MDC (Mapped Diagnostic Context)

MDC allows storing contextual information that can be included in log messages. This context is thread-local, meaning each thread has its own context that doesn't interfere with other threads.

Key components:

- `get_logger(name)`: Creates a logger with MDC capabilities
- `configure_logging()`: Configures the root logger with appropriate handlers
- `mdc_context()`: Context manager for setting temporary MDC values
- `put_mdc(key, value)`: Sets a specific MDC value
- `clear_mdc()`: Clears all MDC values

### Example Usage

```python
from app.utils import get_logger, mdc_context, log_operation

# Get a logger for the current module
logger = get_logger(__name__)

# Add context to logs
with mdc_context(user_id="user123", request_id="req-456"):
    logger.info("Processing user request")
    
    # Log an operation with timing information
    with log_operation(logger, "database_query"):
        # Perform database operation
        results = db.execute_query(sql)
```

## Error Handling

The application uses a custom exception hierarchy to standardize error handling and reporting.

### Exception Hierarchy

- `AppError`: Base exception for all application errors
  - `ValidationError`: For validation errors
  - `DatabaseError`: For database-related errors
  - `NotFoundError`: For resource not found errors
  - `ExternalAPIError`: For errors from external APIs/services
  - `AuthenticationError`: For authentication failures
  - `AuthorizationError`: For authorization/permission errors
  - `InvalidConfigurationError`: For configuration errors

### Helper Functions

- `handle_errors`: Decorator for mapping exceptions to handlers
- `safe_operation`: Decorator for safely executing operations and returning None on error

### Example Usage

```python
from app.utils import handle_errors, DatabaseError, ValidationError

@handle_errors({
    ValidationError: lambda e: {"error": e.message},
    DatabaseError: lambda e: {"error": "Database error occurred"}
})
def process_data(data):
    # Process data with error handling
    validate_data(data)
    save_to_database(data)
    return {"success": True}
```

## Integration in Components

The logging and error handling system is integrated throughout the application:

1. **Application Startup**: The main application configures logging and handles startup errors.
2. **Snowflake Integration**: Uses both logging and error handling for queries and connections.
3. **Database Operations**: Standardizes database errors.
4. **API Routes**: Consistent error responses in API endpoints.

## Best Practices

When extending or modifying the application:

1. Always use the provided `get_logger` function instead of direct `logging` module use.
2. Use appropriate custom exceptions rather than generic Python exceptions.
3. Use the `mdc_context` context manager to add relevant context to logs.
4. Use the `log_operation` context manager for operations that should be timed.
5. Handle exceptions at appropriate boundaries using the provided decorators.

## Log File Location

Logs are stored in the `logs` directory by default. The location can be customized via the `LOGS_DIR` environment variable. 