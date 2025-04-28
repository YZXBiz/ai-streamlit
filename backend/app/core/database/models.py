"""
SQLAlchemy ORM models for database tables.

This module defines the database schema using SQLAlchemy ORM classes.
"""

from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SQLAlchemyEnum
from sqlalchemy.orm import relationship

from .database import Base


class MessageSenderEnum(str, PyEnum):
    """Enum for message sender types."""

    USER = "USER"
    ASSISTANT = "ASSISTANT"


# Define models
class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    files = relationship("DataFile", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class DataFile(Base):
    """Data file database model."""

    __tablename__ = "data_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    file_type = Column(String(50), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = relationship("User", back_populates="files")
    chat_sessions = relationship("ChatSession", back_populates="data_file")


class ChatSession(Base):
    """Chat session database model."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    description = Column(Text)
    data_file_id = Column(Integer, ForeignKey("data_files.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    last_active_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    user = relationship("User", back_populates="sessions")
    data_file = relationship("DataFile", back_populates="chat_sessions")
    messages = relationship("Message", back_populates="chat_session", cascade="all, delete-orphan")


class Message(Base):
    """Message database model."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    sender = Column(SQLAlchemyEnum(MessageSenderEnum), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    chat_session = relationship("ChatSession", back_populates="messages")
