"""
Chat API endpoints.

This module defines the routes for chat sessions and message handling.
"""

from fastapi import APIRouter, Depends, status

from ...core.security import TokenData
from ...services.chat_service import ChatService
from ..deps import get_chat_service, get_current_active_user
from ..schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    MessageCreate,
    MessageResponse,
)

router = APIRouter(tags=["chat"])


@router.post("/sessions", response_model=ChatSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    session_data: ChatSessionCreate,
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatSessionResponse:
    """Create a new chat session."""
    session = await chat_service.create_session(
        user_id=current_user.user_id,
        data_file_id=session_data.data_file_id,
        name=session_data.name or "",
        description=session_data.description or "",
    )

    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        name=session.name,
        description=session.description,
        data_file_id=session.data_file_id,
        messages=[],
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_active_at=session.last_active_at,
    )


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def get_chat_sessions(
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[ChatSessionResponse]:
    """Get all chat sessions for the current user."""
    sessions = await chat_service.get_user_sessions(current_user.user_id)
    return [
        ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            name=session.name,
            description=session.description,
            data_file_id=session.data_file_id,
            messages=[],  # Don't include messages in the list view
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_active_at=session.last_active_at,
        )
        for session in sessions
    ]


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatSessionResponse:
    """Get a specific chat session with messages."""
    session = await chat_service.get_session(session_id, current_user.user_id)

    return ChatSessionResponse(
        id=session.id,
        user_id=session.user_id,
        name=session.name,
        description=session.description,
        data_file_id=session.data_file_id,
        messages=[
            MessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                sender=msg.sender.name,
                content=msg.content,
                created_at=msg.created_at,
            )
            for msg in session.messages
        ],
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_active_at=session.last_active_at,
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat_session(
    session_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> None:
    """Delete a chat session."""
    await chat_service.delete_session(session_id, current_user.user_id)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=list[MessageResponse],
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: int,
    message_data: MessageCreate,
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[MessageResponse]:
    """
    Send a message in a chat session and get the AI response.

    Returns both the user message and the AI response.
    """
    user_message, ai_message = await chat_service.send_message(
        session_id=session_id,
        user_id=current_user.user_id,
        content=message_data.content,
    )

    # Return both messages
    return [
        MessageResponse(
            id=user_message.id,
            session_id=user_message.session_id,
            sender=user_message.sender.name,
            content=user_message.content,
            created_at=user_message.created_at,
        ),
        MessageResponse(
            id=ai_message.id,
            session_id=ai_message.session_id,
            sender=ai_message.sender.name,
            content=ai_message.content,
            created_at=ai_message.created_at,
        ),
    ]


@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: int,
    current_user: TokenData = Depends(get_current_active_user),
    chat_service: ChatService = Depends(get_chat_service),
) -> list[MessageResponse]:
    """Get messages for a specific chat session."""
    messages = await chat_service.get_messages(session_id, current_user.user_id)

    return [
        MessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            sender=msg.sender.name,
            content=msg.content,
            created_at=msg.created_at,
        )
        for msg in messages
    ]
