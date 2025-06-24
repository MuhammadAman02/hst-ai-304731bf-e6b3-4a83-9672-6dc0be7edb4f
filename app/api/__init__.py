"""API package for the application.

This package contains all API endpoint definitions using FastAPI routers.
The main router is exposed here for easy importing in the main application.
"""

from app.api.router import api_router

__all__ = ["api_router"]