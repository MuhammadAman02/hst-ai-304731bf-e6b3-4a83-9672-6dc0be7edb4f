import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger for the application
app_logger = logging.getLogger("app")

# Set the default level
app_logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
app_logger.addHandler(console_handler)

# Create a file handler if LOG_FILE is set in environment
log_file = os.getenv("LOG_FILE")
if log_file:
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a rotating file handler (10 MB max size, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)

# Set log level from environment variable if provided
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
    app_logger.setLevel(getattr(logging, log_level))

# Helper function to create a logger for a specific module
def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create a logger for a specific module.
    
    Args:
        name: The name of the module (typically __name__)
        level: Optional log level override
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level from parameter or environment
    if level and level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(app_logger.level)
    
    # Add handlers if not already present
    if not logger.handlers:
        # Add console handler
        logger.addHandler(console_handler)
        
        # Add file handler if available
        if log_file and 'file_handler' in locals():
            logger.addHandler(file_handler)
    
    return logger

# Helper function to log structured data
def log_structured(logger: logging.Logger, level: str, message: str, data: Dict[str, Any]) -> None:
    """Log a message with structured data.
    
    Args:
        logger: The logger instance
        level: The log level (debug, info, warning, error, critical)
        message: The log message
        data: Dictionary of structured data to include
    """
    if level.lower() == "debug":
        logger.debug(f"{message} - {data}")
    elif level.lower() == "info":
        logger.info(f"{message} - {data}")
    elif level.lower() == "warning":
        logger.warning(f"{message} - {data}")
    elif level.lower() == "error":
        logger.error(f"{message} - {data}")
    elif level.lower() == "critical":
        logger.critical(f"{message} - {data}")