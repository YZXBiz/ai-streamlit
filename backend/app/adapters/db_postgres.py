"""
PostgreSQL repository implementations.

This module implements the repository interfaces using SQLAlchemy ORM and PostgreSQL.
"""

import builtins

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..core.database.models import ChatSession as ChatSessionModel
from ..core.database.models import DataFile as DataFileModel
from ..core.database.models import Message as MessageModel
from ..core.database.models import User as UserModel
from ..domain.models.chat_session import ChatSession, Message, MessageSender
from ..domain.models.datafile import DataFile, FileType
from ..domain.models.user import User
from ..ports.repository import ChatSessionRepository, DataFileRepository, UserRepository


class PostgresUserRepository(UserRepository):
    """PostgreSQL implementation of the UserRepository interface."""

    def __init__(self, session: AsyncSession):
        """Initialize with a database session."""
        self.session = session

    async def get(self, entity_id: int) -> User | None:
        """Get a user by ID."""
        result = await self.session.execute(select(UserModel).filter(UserModel.id == entity_id))
        db_user = result.scalars().first()

        if db_user is None:
            return None

        return self._db_to_domain(db_user)

    async def create(self, entity: User) -> User:
        """Create a new user."""
        db_user = UserModel(
            username=entity.username,
            email=entity.email,
            hashed_password=entity.hashed_password,
            first_name=entity.first_name,
            last_name=entity.last_name,
            is_active=entity.is_active,
            is_admin=entity.is_admin,
        )

        self.session.add(db_user)
        await self.session.commit()
        await self.session.refresh(db_user)

        return self._db_to_domain(db_user)

    async def update(self, entity: User) -> User:
        """Update an existing user."""
        result = await self.session.execute(select(UserModel).filter(UserModel.id == entity.id))
        db_user = result.scalars().first()

        if db_user is None:
            raise ValueError(f"User with ID {entity.id} not found")

        # Update fields
        db_user.username = entity.username
        db_user.email = entity.email
        db_user.hashed_password = entity.hashed_password
        db_user.first_name = entity.first_name
        db_user.last_name = entity.last_name
        db_user.is_active = entity.is_active
        db_user.is_admin = entity.is_admin

        await self.session.commit()
        await self.session.refresh(db_user)

        return self._db_to_domain(db_user)

    async def delete(self, entity_id: int) -> bool:
        """Delete a user by ID."""
        result = await self.session.execute(select(UserModel).filter(UserModel.id == entity_id))
        db_user = result.scalars().first()

        if db_user is None:
            return False

        await self.session.delete(db_user)
        await self.session.commit()

        return True

    async def list(self, skip: int = 0, limit: int = 100) -> list[User]:
        """List users with pagination."""
        result = await self.session.execute(select(UserModel).offset(skip).limit(limit))
        db_users = result.scalars().all()

        return [self._db_to_domain(db_user) for db_user in db_users]

    async def get_by_username(self, username: str) -> User | None:
        """Get a user by username."""
        result = await self.session.execute(
            select(UserModel).filter(UserModel.username == username)
        )
        db_user = result.scalars().first()

        if db_user is None:
            return None

        return self._db_to_domain(db_user)

    async def get_by_email(self, email: str) -> User | None:
        """Get a user by email."""
        result = await self.session.execute(select(UserModel).filter(UserModel.email == email))
        db_user = result.scalars().first()

        if db_user is None:
            return None

        return self._db_to_domain(db_user)

    def _db_to_domain(self, db_user: UserModel) -> User:
        """Convert a database user model to a domain user model."""
        return User(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            hashed_password=db_user.hashed_password,
            first_name=db_user.first_name,
            last_name=db_user.last_name,
            is_active=db_user.is_active,
            is_admin=db_user.is_admin,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
        )


class PostgresDataFileRepository(DataFileRepository):
    """PostgreSQL implementation of the DataFileRepository interface."""

    def __init__(self, session: AsyncSession):
        """Initialize with a database session."""
        self.session = session

    async def get(self, entity_id: int) -> DataFile | None:
        """Get a data file by ID."""
        result = await self.session.execute(
            select(DataFileModel).filter(DataFileModel.id == entity_id)
        )
        db_file = result.scalars().first()

        if db_file is None:
            return None

        return self._db_to_domain(db_file)

    async def create(self, entity: DataFile) -> DataFile:
        """Create a new data file."""
        db_file = DataFileModel(
            user_id=entity.user_id,
            filename=entity.filename,
            original_filename=entity.original_filename,
            file_path=entity.file_path,
            file_size=entity.file_size,
            file_type=entity.file_type.name,
            description=entity.description,
        )

        self.session.add(db_file)
        await self.session.commit()
        await self.session.refresh(db_file)

        return self._db_to_domain(db_file)

    async def update(self, entity: DataFile) -> DataFile:
        """Update an existing data file."""
        result = await self.session.execute(
            select(DataFileModel).filter(DataFileModel.id == entity.id)
        )
        db_file = result.scalars().first()

        if db_file is None:
            raise ValueError(f"DataFile with ID {entity.id} not found")

        # Update fields
        db_file.user_id = entity.user_id
        db_file.filename = entity.filename
        db_file.original_filename = entity.original_filename
        db_file.file_path = entity.file_path
        db_file.file_size = entity.file_size
        db_file.file_type = entity.file_type.name
        db_file.description = entity.description

        await self.session.commit()
        await self.session.refresh(db_file)

        return self._db_to_domain(db_file)

    async def delete(self, entity_id: int) -> bool:
        """Delete a data file by ID."""
        result = await self.session.execute(
            select(DataFileModel).filter(DataFileModel.id == entity_id)
        )
        db_file = result.scalars().first()

        if db_file is None:
            return False

        await self.session.delete(db_file)
        await self.session.commit()

        return True

    async def list(self, skip: int = 0, limit: int = 100) -> list[DataFile]:
        """List data files with pagination."""
        result = await self.session.execute(select(DataFileModel).offset(skip).limit(limit))
        db_files = result.scalars().all()

        return [self._db_to_domain(db_file) for db_file in db_files]

    async def get_by_user(
        self, user_id: int, skip: int = 0, limit: int = 100
    ) -> builtins.list[DataFile]:
        """Get all data files for a specific user."""
        result = await self.session.execute(
            select(DataFileModel).filter(DataFileModel.user_id == user_id).offset(skip).limit(limit)
        )
        db_files = result.scalars().all()

        return [self._db_to_domain(db_file) for db_file in db_files]

    def _db_to_domain(self, db_file: DataFileModel) -> DataFile:
        """Convert a database data file model to a domain data file model."""
        return DataFile(
            id=db_file.id,
            user_id=db_file.user_id,
            filename=db_file.filename,
            original_filename=db_file.original_filename,
            file_path=db_file.file_path,
            file_size=db_file.file_size,
            file_type=FileType[db_file.file_type],
            description=db_file.description,
            created_at=db_file.created_at,
            updated_at=db_file.updated_at,
        )


class PostgresChatSessionRepository(ChatSessionRepository):
    """PostgreSQL implementation of the ChatSessionRepository interface."""

    def __init__(self, session: AsyncSession):
        """Initialize with a database session."""
        self.session = session

    async def get(self, entity_id: int) -> ChatSession | None:
        """Get a chat session by ID."""
        result = await self.session.execute(
            select(ChatSessionModel).filter(ChatSessionModel.id == entity_id)
        )
        db_session = result.scalars().first()

        if db_session is None:
            return None

        return await self._db_to_domain(db_session)

    async def create(self, entity: ChatSession) -> ChatSession:
        """Create a new chat session."""
        db_session = ChatSessionModel(
            user_id=entity.user_id,
            name=entity.name,
            description=entity.description,
            data_file_id=entity.data_file_id,
        )

        self.session.add(db_session)
        await self.session.commit()
        await self.session.refresh(db_session)

        # Create messages if any
        for message in entity.messages:
            db_message = MessageModel(
                session_id=db_session.id,
                sender=message.sender.name,
                content=message.content,
            )
            self.session.add(db_message)

        if entity.messages:
            await self.session.commit()

        return await self._db_to_domain(db_session)

    async def update(self, entity: ChatSession) -> ChatSession:
        """Update an existing chat session."""
        result = await self.session.execute(
            select(ChatSessionModel).filter(ChatSessionModel.id == entity.id)
        )
        db_session = result.scalars().first()

        if not db_session:
            return None

        db_session.name = entity.name
        db_session.data_file_id = entity.data_file_id

        # Delete existing messages
        await self.session.execute(
            select(MessageModel).filter(MessageModel.session_id == entity.id).delete()
        )

        # Create new messages
        for message in entity.messages:
            db_message = MessageModel(
                session_id=db_session.id,
                sender=message.sender.name,
                content=message.content,
                created_at=message.created_at,
            )
            self.session.add(db_message)

        await self.session.commit()
        await self.session.refresh(db_session)

        return self._db_to_domain(db_session)

    async def delete(self, entity_id: int) -> bool:
        """Delete a chat session by ID."""
        result = await self.session.execute(
            select(ChatSessionModel).filter(ChatSessionModel.id == entity_id)
        )
        db_session = result.scalars().first()

        if not db_session:
            return False

        # Delete messages first
        await self.session.execute(
            select(MessageModel).filter(MessageModel.session_id == entity_id).delete()
        )

        # Then delete the session
        await self.session.delete(db_session)
        await self.session.commit()

        return True

    async def list_items(self, skip: int = 0, limit: int = 100) -> list[ChatSession]:
        """List chat sessions with pagination."""
        result = await self.session.execute(select(ChatSessionModel).offset(skip).limit(limit))
        db_sessions = result.scalars().all()

        return [self._db_to_domain(db_session) for db_session in db_sessions]

    async def get_by_user(self, user_id: int, skip: int = 0, limit: int = 100) -> list[ChatSession]:
        """Get all chat sessions for a specific user."""
        result = await self.session.execute(
            select(ChatSessionModel)
            .filter(ChatSessionModel.user_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        db_sessions = result.scalars().all()

        return [self._db_to_domain(db_session) for db_session in db_sessions]

    async def get_with_messages(self, session_id: int) -> ChatSession | None:
        """Get a chat session with all its messages."""
        return await self.get(session_id)

    async def add_message(self, message: Message) -> Message:
        """Add a message to a chat session."""
        db_message = MessageModel(
            session_id=message.session_id,
            sender=message.sender.name,
            content=message.content,
        )

        self.session.add(db_message)
        await self.session.commit()
        await self.session.refresh(db_message)

        # Update the last active time for the session
        result = await self.session.execute(
            select(ChatSessionModel).filter(ChatSessionModel.id == message.session_id)
        )
        db_session = result.scalars().first()

        if db_session:
            db_session.last_active_at = message.created_at
            await self.session.commit()

        return Message(
            id=db_message.id,
            session_id=db_message.session_id,
            sender=MessageSender[db_message.sender],
            content=db_message.content,
            created_at=db_message.created_at,
        )

    async def get_messages(
        self, session_id: int, skip: int = 0, limit: int = 100
    ) -> builtins.list[Message]:
        """Get messages for a specific chat session."""
        result = await self.session.execute(
            select(MessageModel)
            .filter(MessageModel.session_id == session_id)
            .order_by(MessageModel.created_at)
            .offset(skip)
            .limit(limit)
        )
        db_messages = result.scalars().all()

        return [
            Message(
                id=db_message.id,
                session_id=db_message.session_id,
                sender=MessageSender[db_message.sender],
                content=db_message.content,
                created_at=db_message.created_at,
            )
            for db_message in db_messages
        ]

    async def _db_to_domain(self, db_session: ChatSessionModel) -> ChatSession:
        """Convert a database chat session model to a domain chat session model."""
        # Get messages
        messages = []
        for db_message in db_session.messages:
            messages.append(
                Message(
                    id=db_message.id,
                    sender=MessageSender[db_message.sender],
                    content=db_message.content,
                    created_at=db_message.created_at,
                )
            )

        return ChatSession(
            id=db_session.id,
            user_id=db_session.user_id,
            name=db_session.name,
            data_file_id=db_session.data_file_id,
            created_at=db_session.created_at,
            messages=messages,
        )
