from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    """Base user model with common attributes."""
    username: str = Field(..., min_length=3, max_length=50, description="Username for login")
    email: EmailStr = Field(..., description="Email address")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name of the user")
    disabled: bool = Field(False, description="Whether the user is disabled")


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8, description="Password for login")

    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "password": "secretpassword"
            }
        }


class UserUpdate(BaseModel):
    """Model for updating user information."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    password: Optional[str] = Field(None, min_length=8)
    disabled: Optional[bool] = None

    class Config:
        schema_extra = {
            "example": {
                "email": "john.new@example.com",
                "full_name": "John New Name"
            }
        }


class UserInDB(UserBase):
    """Model for user information stored in the database."""
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    roles: List[str] = Field(default_factory=list)

    class Config:
        orm_mode = True


class User(UserBase):
    """Model for user information returned to clients."""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    roles: List[str] = Field(default_factory=list)

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "username": "johndoe",
                "email": "john.doe@example.com",
                "full_name": "John Doe",
                "disabled": False,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-02T00:00:00Z",
                "roles": ["user"]
            }
        }


class Token(BaseModel):
    """Model for JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


class TokenData(BaseModel):
    """Model for data stored in JWT token."""
    username: Optional[str] = None
    exp: Optional[datetime] = None
    roles: List[str] = Field(default_factory=list)