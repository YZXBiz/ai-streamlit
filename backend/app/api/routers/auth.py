"""Authentication API endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.app.api.deps import get_auth_service
from backend.app.api.schemas import LoginRequest, Token, UserCreate, UserResponse
from backend.app.services.auth_service import AuthService

router = APIRouter(tags=["auth"])


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """
    Authenticate and get access token.

    This endpoint follows the OAuth2 password flow standard.
    """
    _, access_token = await auth_service.login(form_data.username, form_data.password)
    return Token(access_token=access_token)


@router.post("/login/json", response_model=Token)
async def login_json(
    login_data: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> Token:
    """
    Authenticate and get access token using JSON request.

    Alternative to the form-based login for API clients.
    """
    user, access_token = await auth_service.login(login_data.username, login_data.password)
    return Token(access_token=access_token)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """Register a new user."""
    user = await auth_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        is_admin=False,  # Regular users can't create admin accounts
    )
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        is_active=user.is_active,
        is_admin=user.is_admin,
        created_at=user.created_at,
    )
