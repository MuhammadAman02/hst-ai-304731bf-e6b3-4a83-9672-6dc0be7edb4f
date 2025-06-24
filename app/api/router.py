from fastapi import APIRouter

# Import all API routers
from app.api.auth import router as auth_router
from app.api.example import router as example_router

# Create a main API router
api_router = APIRouter()

# Include all API routers
api_router.include_router(auth_router)
api_router.include_router(example_router)

# Add more routers here as your application grows
# api_router.include_router(users_router)
# api_router.include_router(items_router)
# etc.