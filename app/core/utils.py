import os
import importlib
import inspect
from typing import List, Dict, Any, Optional, Type, Callable
from fastapi import FastAPI, APIRouter
import pkgutil
from pathlib import Path

from app.core.logging import app_logger
from app.core.config import settings

def setup_routers(app: FastAPI, api_prefix: str = "/api") -> None:
    """Automatically set up all routers in the app/api directory.
    
    This function scans the app/api directory for modules containing APIRouter instances
    and includes them in the FastAPI application with the specified prefix.
    
    Args:
        app: The FastAPI application instance
        api_prefix: The prefix for all API routes (default: "/api")
    """
    try:
        # Get the path to the api directory
        api_dir = Path(os.path.dirname(os.path.dirname(__file__))) / "api"
        
        # Check if the api directory exists
        if not api_dir.exists() or not api_dir.is_dir():
            app_logger.warning(f"API directory not found: {api_dir}")
            return
        
        # Import all modules in the api directory
        package_name = "app.api"
        
        # Track the number of routers added
        router_count = 0
        
        # Scan the package for modules
        for _, module_name, is_pkg in pkgutil.iter_modules([str(api_dir)]):
            if is_pkg:
                continue  # Skip packages for now
            
            # Import the module
            module = importlib.import_module(f"{package_name}.{module_name}")
            
            # Look for APIRouter instances in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, APIRouter):
                    # Include the router in the app
                    app.include_router(attr, prefix=api_prefix)
                    router_count += 1
                    app_logger.info(f"Added router from {module_name}.{attr_name}")
        
        app_logger.info(f"Set up {router_count} routers with prefix '{api_prefix}'")
    except Exception as e:
        app_logger.error(f"Error setting up routers: {e}")

def validate_environment() -> List[str]:
    """Validate required environment variables.
    
    Returns:
        List of error messages for missing or invalid environment variables
    """
    errors = []
    
    # Check for required environment variables
    required_vars = [
        # Add required environment variables here
        # Example: ("DATABASE_URL", "Database connection string is required")
    ]
    
    for var_name, error_message in required_vars:
        if not os.getenv(var_name):
            errors.append(error_message)
    
    # Check for recommended environment variables
    if os.getenv("SECRET_KEY") == "CHANGEME_IN_PRODUCTION" and not settings.debug:
        errors.append("SECRET_KEY should be changed in production")
    
    return errors

def import_string(dotted_path: str) -> Any:
    """Import a dotted module path and return the attribute/class designated by the last name.
    
    Args:
        dotted_path: The dotted path to import (e.g., "app.core.utils.import_string")
        
    Returns:
        The imported attribute/class
        
    Raises:
        ImportError: If the import failed
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as e:
        raise ImportError(f"{dotted_path} doesn't look like a module path") from e

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(f"Module '{module_path}' does not define a '{class_name}' attribute") from e

def get_subclasses(base_class: Type, package: str) -> List[Type]:
    """Get all subclasses of a base class in a package.
    
    Args:
        base_class: The base class to find subclasses of
        package: The package to search in (e.g., "app.models")
        
    Returns:
        List of subclasses
    """
    subclasses = []
    package_dir = Path(importlib.import_module(package).__file__).parent
    
    for _, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if is_pkg:
            continue  # Skip packages for now
        
        # Import the module
        module = importlib.import_module(f"{package}.{module_name}")
        
        # Look for subclasses in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isclass(attr) and issubclass(attr, base_class) and attr != base_class:
                subclasses.append(attr)
    
    return subclasses

def create_dir_if_not_exists(directory: str) -> None:
    """Create a directory if it doesn't exist.
    
    Args:
        directory: The directory path to create
    """
    os.makedirs(directory, exist_ok=True)

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # The project root is two directories up from this file
    return Path(__file__).parent.parent.parent

def get_app_dir() -> Path:
    """Get the app directory.
    
    Returns:
        Path to the app directory
    """
    # The app directory is one directory up from this file
    return Path(__file__).parent.parent