import hashlib
import hmac

from app.settings import settings


class AuthModel:
    """
    Model for handling authentication and user credentials.
    """

    def __init__(self):
        """Initialize the auth model with salt from settings."""
        self.salt = settings.password_salt.encode()

    def get_credentials(self) -> tuple[str, str]:
        """
        Get default username and password.

        Returns:
            Tuple of (username, password)
        """
        return settings.default_username, settings.default_password

    def hash_password(self, password: str) -> str:
        """
        Hash a password using HMAC SHA-256.

        Args:
            password: The plain text password

        Returns:
            The hashed password
        """
        return hmac.new(self.salt, password.encode(), hashlib.sha256).hexdigest()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against a hash.

        Args:
            password: The plain text password to verify
            hashed_password: The hashed password to compare against

        Returns:
            True if the password is correct, False otherwise
        """
        password_hash = self.hash_password(password)
        return password_hash == hashed_password

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate a user with username and password.

        Args:
            username: The username to authenticate
            password: The password to authenticate

        Returns:
            True if authentication is successful, False otherwise
        """
        default_username, default_password = self.get_credentials()

        # Simple comparison for development
        # In production, use hashed passwords and a proper user database
        if username == default_username and password == default_password:
            return True

        return False
