from typing import Any, Dict, List, Optional, Union
import os
from pydantic import AnyHttpUrl, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings.
    
    This class uses Pydantic's BaseSettings which automatically reads from environment variables.
    Environment variables take precedence over values defined in the class.
    
    Example:
        If you define PORT=9000 in your environment, it will override the default value of 8000.
    """
    # CORE SETTINGS
    app_name: str = "FastAPI Application"
    app_description: str = "A modern web application built with FastAPI"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # API SETTINGS
    api_prefix: str = "/api"
    
    # SERVER SETTINGS
    host: str = "0.0.0.0"  # 0.0.0.0 for Docker/production compatibility
    port: int = 8000
    
    # CORS SETTINGS
    # List of origins that are allowed to make cross-origin requests
    # Use ["*"] to allow any origin (not recommended for production)
    cors_origins: List[str] = []
    
    # SECURITY SETTINGS
    # Secret key for signing tokens - MUST be overridden in production
    secret_key: str = "CHANGEME_IN_PRODUCTION"
    # Algorithm used for token signing
    algorithm: str = "HS256"
    # Token expiration time in minutes
    access_token_expire_minutes: int = 30
    
    # DATABASE SETTINGS
    # Database connection string - override in production
    database_url: Optional[str] = None
    
    # STATIC FILES
    static_dir: str = "app/static"
    
    # TEMPLATES
    templates_dir: str = "app/templates"
    
    # Configure Pydantic to use environment variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Validators
    @field_validator("cors_origins", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        """Parse CORS origins from string to list.
        
        This allows setting CORS_ORIGINS as a comma-separated string in .env file.
        """
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

# Create a global settings instance
settings = Settings()

# Helper function to get settings as a dictionary
def get_settings_dict() -> Dict[str, Any]:
    """Return settings as a dictionary for easy access."""
    return settings.model_dump()

# Helper function to get a specific setting
def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific setting by key with an optional default value."""
    return getattr(settings, key, default)