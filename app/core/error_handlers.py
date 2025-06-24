from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Dict, Any, List, Optional, Union, Callable
import traceback
import sys

from app.core.exceptions import AppException, ErrorResponse, ErrorDetail
from app.core.logging import app_logger

def setup_error_handlers(app: FastAPI) -> None:
    """Configure global exception handlers for the application.
    
    Args:
        app: The FastAPI application instance
    """
    
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        """Handle application-specific exceptions."""
        app_logger.error(
            f"AppException: {exc.detail}", 
            extra={
                "status_code": exc.status_code,
                "path": request.url.path,
                "method": request.method,
            }
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle FastAPI request validation errors."""
        errors = []
        for error in exc.errors():
            errors.append(ErrorDetail(
                loc=error.get("loc", []),
                msg=error.get("msg", "Validation error"),
                type=error.get("type", "validation_error")
            ))
        
        app_logger.warning(
            f"Validation error: {errors}",
            extra={
                "path": request.url.path,
                "method": request.method,
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": [error.dict() for error in errors]},
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """Handle Pydantic validation errors."""
        errors = []
        for error in exc.errors():
            errors.append(ErrorDetail(
                loc=error.get("loc", []),
                msg=error.get("msg", "Validation error"),
                type=error.get("type", "validation_error")
            ))
        
        app_logger.warning(
            f"Pydantic validation error: {errors}",
            extra={
                "path": request.url.path,
                "method": request.method,
            }
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": [error.dict() for error in errors]},
        )
    
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle any unhandled exceptions."""
        # Get the full exception traceback
        exc_info = sys.exc_info()
        traceback_str = "".join(traceback.format_exception(*exc_info))
        
        # Log the full traceback
        app_logger.error(
            f"Unhandled exception: {str(exc)}",
            extra={
                "traceback": traceback_str,
                "path": request.url.path,
                "method": request.method,
                "exception_type": exc.__class__.__name__,
            }
        )
        
        # In production, don't return the traceback to the client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An unexpected error occurred. Please try again later."
            },
        )

def create_error_response(
    status_code: int, 
    detail: Union[str, List[ErrorDetail]],
    headers: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Create a standardized error response.
    
    Args:
        status_code: HTTP status code
        detail: Error detail message or list of error details
        headers: Optional response headers
        
    Returns:
        JSONResponse with standardized error format
    """
    if isinstance(detail, str):
        content = {"detail": detail}
    else:
        content = {"detail": [error.dict() for error in detail]}
    
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers,
    )

def with_error_handling(func: Callable) -> Callable:
    """Decorator to add error handling to any function.
    
    This decorator catches exceptions and logs them appropriately.
    It can be used for non-FastAPI functions that need error handling.
    
    Args:
        func: The function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AppException as exc:
            app_logger.error(f"AppException in {func.__name__}: {exc.detail}")
            raise
        except Exception as exc:
            app_logger.error(
                f"Unhandled exception in {func.__name__}: {str(exc)}",
                extra={"traceback": traceback.format_exc()}
            )
            raise AppException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An unexpected error occurred. Please try again later."
            )
    
    return wrapper