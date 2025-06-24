import os
import sys
from dotenv import load_dotenv
from nicegui import ui

# Load environment variables from .env file (if present)
load_dotenv()

# Import the page definitions from app.main
# This ensures that the @ui.page decorators in app/main.py are executed
# and the routes are registered with NiceGUI before ui.run() is called.
try:
    import app.main  # noqa: F401 -> Ensure app.main is imported to register pages
except ImportError as e:
    print(f"Error importing app.main: {e}")
    print("Make sure the app directory is properly set up.")
    sys.exit(1)

if __name__ in {"__main__", "__mp_main__"}: # Recommended by NiceGUI for multiprocessing compatibility
    try:
        # Import core modules
        from app.core import (
            settings, 
            app_logger, 
            setup_logging,
            setup_middleware, 
            setup_routers, 
            validate_environment,
            setup_error_handlers,
            HealthCheck,
            is_healthy
        )
        
        # Set up logging
        setup_logging()
        app_logger.info(f"Starting {settings.app_name} v{settings.app_version}")
        
        # Create FastAPI app
        from fastapi import FastAPI, APIRouter
        app = FastAPI(
            title=settings.app_name,
            description=settings.app_description,
            version=settings.app_version,
            docs_url=f"{settings.api_prefix}/docs" if settings.api_prefix else "/docs",
            redoc_url=f"{settings.api_prefix}/redoc" if settings.api_prefix else "/redoc",
        )
        
        # Set up error handlers
        setup_error_handlers(app)
        
        # Set up middleware
        setup_middleware(app)
        
        # Set up routers
        setup_routers(app, api_prefix=settings.api_prefix)
        
        # Add health check endpoint
        health_router = APIRouter()
        
        @health_router.get("/health")
        async def health_check():
            return HealthCheck.check_all()
        
        app.include_router(health_router)
        
        # Validate environment
        errors = validate_environment()
        if errors:
            for error in errors:
                app_logger.error(f"Environment validation error: {error}")
        
        # Optional: Set up database if configured
        try:
            from app.core import setup_database
            setup_database()
        except (ImportError, AttributeError):
            app_logger.info("Database not configured, skipping setup")
        
        # Run the application
        app_logger.info(f"Starting server at {settings.host}:{settings.port}")
        ui.run(
            host=settings.host,
            port=settings.port,
            title=settings.app_name,
            uvicorn_logging_level='info' if settings.debug else 'warning',
            reload=settings.debug,  # IMPORTANT: Set to False for production/deployment
            app=app,
            storage_secret=settings.secret_key,  # Use the same secret key for session storage
        )
    except Exception as e:
        # Import traceback here to avoid circular imports
        import traceback
        
        # Try to use app_logger if available, otherwise fall back to print
        try:
            app_logger.critical(f"Error starting application: {e}")
            app_logger.critical(traceback.format_exc())
        except NameError:
            print(f"CRITICAL ERROR: {e}")
            print(traceback.format_exc())
        
        sys.exit(1)