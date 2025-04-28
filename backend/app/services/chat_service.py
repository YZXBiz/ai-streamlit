"""Chat service for managing chat sessions and queries."""

from datetime import datetime
from typing import Any

from fastapi import HTTPException, status

from backend.app.domain.models.chat_session import ChatSession, Message, MessageSender
from backend.app.ports.llm import DataAnalysisService
from backend.app.ports.repository import ChatSessionRepository, DataFileRepository
from backend.app.ports.vectorstore import VectorStore


class ChatService:
    """Service for managing chat sessions and handling queries."""

    def __init__(
        self,
        session_repository: ChatSessionRepository,
        file_repository: DataFileRepository,
        data_service: DataAnalysisService,
        vector_store: VectorStore | None = None,
    ):
        """
        Initialize with repositories and the data analysis service.

        Args:
            session_repository: Repository for chat sessions
            file_repository: Repository for data files
            data_service: Service for data analysis
            vector_store: Optional vector store for long-term memory
        """
        self.session_repository = session_repository
        self.file_repository = file_repository
        self.data_service = data_service
        self.vector_store = vector_store

    async def create_session(
        self, user_id: int, data_file_id: int | None = None, name: str = "", description: str = ""
    ) -> ChatSession:
        """
        Create a new chat session.

        Args:
            user_id: ID of the user creating the session
            data_file_id: Optional ID of the data file to associate
            name: Optional name for the session
            description: Optional description

        Returns:
            Created ChatSession

        Raises:
            HTTPException: If data file not found or doesn't belong to user
        """
        # If a data file is specified, verify it exists and belongs to user
        if data_file_id:
            data_file = await self.file_repository.get(data_file_id)
            if data_file is None or data_file.user_id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Data file not found",
                )

        # Use default name if not provided
        if not name:
            if data_file_id:
                data_file = await self.file_repository.get(data_file_id)
                name = f"Chat about {data_file.original_filename}"
            else:
                name = f"Chat session {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        # Create session
        session = ChatSession(
            user_id=user_id,
            name=name,
            description=description,
            data_file_id=data_file_id,
        )

        return await self.session_repository.create(session)

    async def get_session(self, session_id: int, user_id: int) -> ChatSession:
        """
        Get a chat session by ID, ensuring it belongs to the user.

        Args:
            session_id: ID of the session
            user_id: ID of the user

        Returns:
            ChatSession if found

        Raises:
            HTTPException: If session not found or doesn't belong to user
        """
        session = await self.session_repository.get_with_messages(session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found",
            )
        return session

    async def get_user_sessions(self, user_id: int) -> list[ChatSession]:
        """
        Get all chat sessions for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of ChatSession objects
        """
        return await self.session_repository.get_by_user_id(user_id)

    async def delete_session(self, session_id: int, user_id: int) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: ID of the session
            user_id: ID of the user

        Returns:
            True if successful

        Raises:
            HTTPException: If session not found or doesn't belong to user
        """
        session = await self.session_repository.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found",
            )

        if session.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this session",
            )

        return await self.session_repository.delete(session_id)

    async def send_message(
        self, session_id: int, user_id: int, content: str
    ) -> tuple[Message, Message]:
        """
        Send a user message and get an AI response.

        Args:
            session_id: ID of the chat session
            user_id: ID of the user sending the message
            content: Message content

        Returns:
            Tuple of (user_message, ai_message)

        Raises:
            HTTPException: If session not found, doesn't belong to user, or query fails
        """
        # Get the session and verify ownership
        session = await self.get_session(session_id, user_id)

        # Add user message to session
        user_message = Message(
            session_id=session_id,
            sender=MessageSender.USER,
            content=content,
        )
        user_message = await self.session_repository.add_message(user_message)

        # Add to vector store for long-term memory if available
        if self.vector_store:
            await self.vector_store.add_memory(
                session_id=session_id,
                text=f"User: {content}",
                metadata={
                    "sender": "USER",
                    "timestamp": datetime.now().isoformat(),
                    "message_id": user_message.id,
                },
            )

        try:
            # Retrieve relevant context from vector store if available
            context = ""
            if self.vector_store:
                relevant_memories = await self.vector_store.query_memory(
                    session_id=session_id, query_text=content, top_k=5
                )

                if relevant_memories:
                    context = "Previous relevant conversation:\n"
                    for memory in relevant_memories:
                        context += f"{memory['text']}\n"
                    context += "\n"

            # Process the query using the data analysis service with context
            response = await self._get_query_response(session, content, context)

            # Add assistant message to session
            ai_message = Message(
                session_id=session_id,
                sender=MessageSender.ASSISTANT,
                content=str(response),
            )
            ai_message = await self.session_repository.add_message(ai_message)

            # Add to vector store for long-term memory if available
            if self.vector_store:
                await self.vector_store.add_memory(
                    session_id=session_id,
                    text=f"Assistant: {response}",
                    metadata={
                        "sender": "ASSISTANT",
                        "timestamp": datetime.now().isoformat(),
                        "message_id": ai_message.id,
                    },
                )

            return user_message, ai_message
        except Exception as e:
            # If query fails, still keep the user message but return error
            error_message = f"Failed to process query: {str(e)}"
            ai_message = Message(
                session_id=session_id,
                sender=MessageSender.ASSISTANT,
                content=error_message,
            )
            ai_message = await self.session_repository.add_message(ai_message)

            return user_message, ai_message

    async def get_messages(
        self, session_id: int, user_id: int, skip: int = 0, limit: int = 100
    ) -> list[Message]:
        """
        Get messages for a specific chat session.

        Args:
            session_id: ID of the chat session
            user_id: ID of the user
            skip: Number of messages to skip
            limit: Maximum number of messages to return

        Returns:
            List of Message objects

        Raises:
            HTTPException: If session not found or doesn't belong to user
        """
        # Verify session ownership
        await self.get_session(session_id, user_id)

        # Get messages
        return await self.session_repository.get_messages(session_id, skip, limit)

    async def _get_query_response(self, session: ChatSession, query: str, context: str = "") -> Any:
        """
        Get a response for a query using the data analysis service.

        Args:
            session: Chat session
            query: User query
            context: Optional context from previous conversations

        Returns:
            Response from data analysis service

        Raises:
            ValueError: If processing fails
        """
        # Determine if we're querying a specific dataframe or collection
        if session.data_file is not None:
            # Get the dataframe name from repository
            if session.data_file_id is None:
                raise ValueError("Data file ID is missing")

            data_file = await self.file_repository.get(session.data_file_id)
            if data_file is None:
                raise ValueError("Associated data file not found")

            # Extract dataframe name from filename
            dataframe_name = data_file.filename.split(".")[0]

            # Augment query with context if available
            augmented_query = query
            if context:
                augmented_query = f"{context}\nCurrent question: {query}"

            # Query the dataframe
            return await self.data_service.query_dataframe(augmented_query, dataframe_name)
        else:
            # Not implemented yet - will handle cross-dataframe queries later
            raise ValueError("Sessions without an associated data file are not yet supported")
