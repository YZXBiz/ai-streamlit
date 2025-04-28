"""User domain model."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class User:
    """User domain model representing a user of the application."""

    id: int | None = None
    username: str = ""
    email: str = ""
    hashed_password: str = ""
    first_name: str = ""
    last_name: str = ""
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        """Return the user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.username
