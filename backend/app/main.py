"""Main FastAPI application."""

# import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.deps import get_analyzer_service
from backend.app.api.routers import auth, chat, files
from backend.app.core.config import settings
from backend.app.core.database.session import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    try:
        await create_tables()
    except Exception as e:
        print(f"Error creating database tables: {str(e)}")

    # Create data directories if they don't exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.LOGS_DIR, exist_ok=True)

    print("Application started - docs available at /docs")

    yield  # Application runs here

    # Shutdown
    analyzer = await get_analyzer_service()
    if hasattr(analyzer, "close"):
        await analyzer.close()

    print("Application shutdown complete")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for PandasAI Chat Application",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(files.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)


@app.get("/health", response_model=dict[str, Any])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Dict[str, Any]: Status information
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "api_version": settings.API_V1_STR,
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions globally."""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
