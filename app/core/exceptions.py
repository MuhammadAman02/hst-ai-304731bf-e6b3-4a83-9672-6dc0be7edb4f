from typing import Any, Dict, Optional, List, Union
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorDetail(BaseModel):
    """Model for detailed error information."""
    loc: List[str] = []
    msg: str
    type: str

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: Union[str, List[ErrorDetail]]

class AppException(Exception):
    """Base exception for application-specific exceptions.
    
    This class serves as the base for all application-specific exceptions.
    It includes status code, detail message, and headers information.
    """
    def __init__(
        self, 
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "An unexpected error occurred",
        headers: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
        super().__init__(detail)
    
    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        return HTTPException(
            status_code=self.status_code,
            detail=self.detail,
            headers=self.headers
        )

class NotFoundError(AppException):
    """Exception raised when a resource is not found."""
    def __init__(
        self, 
        detail: str = "Resource not found",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            headers=headers
        )

class ValidationError(AppException):
    """Exception raised when validation fails."""
    def __init__(
        self, 
        detail: str = "Validation error",
        errors: Optional[List[Dict[str, Any]]] = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        self.errors = errors or []
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            headers=headers
        )

class AuthenticationError(AppException):
    """Exception raised when authentication fails."""
    def __init__(
        self, 
        detail: str = "Authentication failed",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers=headers
        )

class AuthorizationError(AppException):
    """Exception raised when authorization fails."""
    def __init__(
        self, 
        detail: str = "Not authorized to perform this action",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            headers=headers
        )

class RateLimitError(AppException):
    """Exception raised when rate limit is exceeded."""
    def __init__(
        self, 
        detail: str = "Rate limit exceeded",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers
        )

class DatabaseError(AppException):
    """Exception raised when a database operation fails."""
    def __init__(
        self, 
        detail: str = "Database operation failed",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )

class ExternalServiceError(AppException):
    """Exception raised when an external service call fails."""
    def __init__(
        self, 
        detail: str = "External service call failed",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers=headers
        )

class ConfigurationError(AppException):
    """Exception raised when there is a configuration error."""
    def __init__(
        self, 
        detail: str = "Application configuration error",
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            headers=headers
        )

# Exception handler for FastAPI
def app_exception_handler(app):
    """Register exception handlers for the application.
    
    Args:
        app: The FastAPI application instance
    """
    @app.exception_handler(AppException)
    async def handle_app_exception(request, exc):
        return await app.exception_handler(HTTPException, exc.to_http_exception())(request, exc.to_http_exception())
    
    # Add more exception handlers as needed