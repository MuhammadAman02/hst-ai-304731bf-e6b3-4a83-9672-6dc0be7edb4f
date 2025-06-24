from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from app.core import app_logger, security, settings
from app.models.user import Token, User

# Create a router for authentication endpoints
router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login, get an access token for future requests."""
    # This is a placeholder - in a real app, you would verify against a database
    # For demo purposes, we'll accept a hardcoded user
    if form_data.username != "demo" or form_data.password != "password":
        app_logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token with configured expiration time
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = security.create_access_token(
        data={"sub": form_data.username, "roles": ["user"]},
        expires_delta=access_token_expires
    )
    
    app_logger.info(f"User {form_data.username} logged in successfully")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.access_token_expire_minutes * 60
    }


@router.get("/me", response_model=User)
async def read_users_me(current_user = Depends(security.get_current_active_user)):
    """Get current user information."""
    # This is a placeholder - in a real app, you would fetch from a database
    # For demo purposes, we'll return a hardcoded user based on the token
    return {
        "id": 1,
        "username": current_user.username,
        "email": "demo@example.com",
        "full_name": "Demo User",
        "disabled": False,
        "created_at": "2023-01-01T00:00:00Z",
        "roles": current_user.roles
    }