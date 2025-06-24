import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Union, Any

from jose import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import settings
from app.core.logging import app_logger

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash.
    
    Args:
        plain_password: The plain-text password
        hashed_password: The hashed password to compare against
        
    Returns:
        True if the password matches the hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate a password hash.
    
    Args:
        password: The plain-text password to hash
        
    Returns:
        The hashed password
    """
    return pwd_context.hash(password)

def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token.
    
    Args:
        data: The data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        The encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    
    # Create the JWT token
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm=settings.algorithm
        )
        return encoded_jwt
    except Exception as e:
        app_logger.error(f"Error creating access token: {e}")
        raise

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode a JWT access token.
    
    Args:
        token: The JWT token to decode
        
    Returns:
        The decoded token payload
        
    Raises:
        HTTPException: If the token is invalid
    """
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> Optional[Dict[str, Any]]:
    """Get the current user from a JWT token.
    
    This is a dependency that can be used in FastAPI route functions.
    
    Args:
        token: The JWT token from the Authorization header
        
    Returns:
        The user data from the token, or None if no token is provided
        
    Raises:
        HTTPException: If the token is invalid
    """
    if not token:
        return None
    
    try:
        payload = decode_access_token(token)
        return payload
    except HTTPException:
        return None

async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get the current active user.
    
    This is a dependency that can be used in FastAPI route functions.
    It requires that a user is authenticated.
    
    Args:
        current_user: The current user from get_current_user
        
    Returns:
        The current user data
        
    Raises:
        HTTPException: If no user is authenticated or the user is inactive
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if the user is active (if you have an 'active' field in your user model)
    if current_user.get("disabled", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    return current_user

def generate_secure_random_string(length: int = 32) -> str:
    """Generate a secure random string.
    
    Args:
        length: The length of the string to generate
        
    Returns:
        A secure random string
    """
    return os.urandom(length).hex()[:length]