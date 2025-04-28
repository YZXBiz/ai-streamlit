"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, EmailStr, Field


# User schemas
class UserBase(BaseModel):
    """Base schema for user data."""

    username: str = Field(..., min_length=3, max_length=64)
    email: EmailStr
    first_name: str | None = Field(None, max_length=64)
    last_name: str | None = Field(None, max_length=64)


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str = Field(..., min_length=8)


class UserResponse(UserBase):
    """Schema for user response data."""

    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Auth schemas
class Token(BaseModel):
    """Schema for authentication token."""

    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Schema for login request."""

    username: str
    password: str


# DataFile schemas
class FileType(str, Enum):
    """Enum for file types."""

    CSV = "CSV"
    EXCEL = "EXCEL"
    PARQUET = "PARQUET"
    JSON = "JSON"


class DataFileBase(BaseModel):
    """Base schema for data file."""

    original_filename: str
    description: str | None = None


class DataFileCreate(DataFileBase):
    """Schema for creating a data file entry."""

    pass


class DataFileResponse(DataFileBase):
    """Schema for data file response."""

    id: int
    user_id: int
    filename: str
    file_path: str
    file_size: int
    file_type: FileType
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


# Chat session schemas
class ChatSessionBase(BaseModel):
    """Base schema for chat session."""

    name: str | None = None
    description: str | None = None
    data_file_id: int | None = None


class ChatSessionCreate(ChatSessionBase):
    """Schema for creating a chat session."""

    pass


class MessageType(str, Enum):
    """Enum for message types."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


class MessageBase(BaseModel):
    """Base schema for chat message."""

    content: str


class MessageResponse(MessageBase):
    """Schema for message response."""

    id: int
    session_id: int
    sender: MessageType
    created_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True


class ChatSessionResponse(ChatSessionBase):
    """Schema for chat session response."""

    id: int
    user_id: int
    messages: list[MessageResponse] | None = []
    created_at: datetime
    updated_at: datetime
    last_active_at: datetime

    # Optional relationship data
    data_file: DataFileResponse | None = None

    class Config:
        """Pydantic config."""

        from_attributes = True


# Message schemas
class MessageCreate(MessageBase):
    """Schema for creating a message."""

    pass


# Query schemas
class QueryRequest(BaseModel):
    """Schema for a data query request."""

    query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    """Schema for a data query response."""

    query: str
    response: str
    execution_time: float  # seconds
    generated_code: str | None = None
