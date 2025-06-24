import os
import sys
import time
import platform
import psutil
from typing import Dict, Any, List, Optional

from app.core.logging import app_logger

class HealthCheck:
    """Health check utility for the application.
    
    This class provides methods to check the health of various components
    of the application, including system resources, database connections,
    and external services.
    """
    
    @staticmethod
    def check_system() -> Dict[str, Any]:
        """Check system health (CPU, memory, disk).
        
        Returns:
            Dict with system health information
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage for the current directory
            disk = psutil.disk_usage(os.getcwd())
            disk_percent = disk.percent
            
            # Get process information
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            return {
                "status": "healthy",
                "cpu": {
                    "percent": cpu_percent,
                    "status": "warning" if cpu_percent > 80 else "healthy",
                },
                "memory": {
                    "percent": memory_percent,
                    "status": "warning" if memory_percent > 80 else "healthy",
                },
                "disk": {
                    "percent": disk_percent,
                    "status": "warning" if disk_percent > 80 else "healthy",
                },
                "process": {
                    "memory_mb": round(process_memory_mb, 2),
                    "status": "warning" if process_memory_mb > 500 else "healthy",
                },
                "platform": platform.platform(),
                "python": sys.version,
            }
        except Exception as e:
            app_logger.error(f"Error checking system health: {e}")
            return {
                "status": "error",
                "message": str(e),
            }
    
    @staticmethod
    def check_database() -> Dict[str, Any]:
        """Check database connection health.
        
        Returns:
            Dict with database health information
        """
        # This is a placeholder. In a real application, you would check
        # the database connection here.
        try:
            # Placeholder for database check
            # In a real application, you would try to connect to the database
            # and perform a simple query to verify the connection.
            return {
                "status": "not_configured",
                "message": "Database health check not configured",
            }
        except Exception as e:
            app_logger.error(f"Error checking database health: {e}")
            return {
                "status": "error",
                "message": str(e),
            }
    
    @staticmethod
    def check_external_services() -> Dict[str, List[Dict[str, Any]]]:
        """Check external service dependencies.
        
        Returns:
            Dict with external service health information
        """
        # This is a placeholder. In a real application, you would check
        # external service connections here.
        return {
            "services": [],
            "status": "not_configured",
            "message": "External service health checks not configured",
        }
    
    @staticmethod
    def check_all() -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Dict with all health check information
        """
        start_time = time.time()
        
        # Run all checks
        system_health = HealthCheck.check_system()
        database_health = HealthCheck.check_database()
        services_health = HealthCheck.check_external_services()
        
        # Determine overall status
        if system_health.get("status") == "error" or database_health.get("status") == "error":
            overall_status = "error"
        elif system_health.get("status") == "warning" or database_health.get("status") == "warning":
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Calculate response time
        response_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "response_time_ms": response_time_ms,
            "system": system_health,
            "database": database_health,
            "services": services_health,
        }

# Helper function to check if a specific component is healthy
def is_healthy(component: str = "all") -> bool:
    """Check if a specific component is healthy.
    
    Args:
        component: The component to check ("system", "database", "services", or "all")
        
    Returns:
        True if the component is healthy, False otherwise
    """
    try:
        if component == "system":
            return HealthCheck.check_system().get("status") == "healthy"
        elif component == "database":
            return HealthCheck.check_database().get("status") == "healthy"
        elif component == "services":
            return HealthCheck.check_external_services().get("status") == "healthy"
        else:  # "all"
            health = HealthCheck.check_all()
            return health.get("status") == "healthy"
    except Exception as e:
        app_logger.error(f"Error checking health for {component}: {e}")
        return False