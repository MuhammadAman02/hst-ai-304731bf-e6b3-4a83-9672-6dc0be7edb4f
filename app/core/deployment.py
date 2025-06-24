import os
import subprocess
import platform
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import time

from app.core.logging import app_logger
from app.core.config import settings

class DeploymentManager:
    """Manages deployment operations for the application.
    
    This class provides methods for deploying the application to various
    environments, including Fly.io, Docker, and local development.
    """
    
    @staticmethod
    def check_fly_installed() -> bool:
        """Check if the Fly.io CLI is installed.
        
        Returns:
            True if the Fly.io CLI is installed, False otherwise
        """
        try:
            # Check if flyctl is installed
            result = subprocess.run(
                ["flyctl", "version"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def check_docker_installed() -> bool:
        """Check if Docker is installed.
        
        Returns:
            True if Docker is installed, False otherwise
        """
        try:
            # Check if Docker is installed
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    @staticmethod
    def build_docker_image(tag: str = "app:latest") -> Tuple[bool, str]:
        """Build a Docker image for the application.
        
        Args:
            tag: The tag to use for the Docker image
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if Docker is installed
            if not DeploymentManager.check_docker_installed():
                return False, "Docker is not installed"
            
            # Build the Docker image
            result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                return True, f"Docker image built successfully: {tag}"
            else:
                return False, f"Failed to build Docker image: {result.stderr}"
        except Exception as e:
            app_logger.error(f"Error building Docker image: {e}")
            return False, f"Error building Docker image: {str(e)}"
    
    @staticmethod
    def run_docker_container(
        tag: str = "app:latest",
        port_mapping: str = "8000:8000",
        env_vars: Optional[Dict[str, str]] = None,
        container_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Run a Docker container for the application.
        
        Args:
            tag: The tag of the Docker image to run
            port_mapping: The port mapping to use (host:container)
            env_vars: Environment variables to pass to the container
            container_name: Optional name for the container
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if Docker is installed
            if not DeploymentManager.check_docker_installed():
                return False, "Docker is not installed"
            
            # Build the command
            command = ["docker", "run", "-p", port_mapping]
            
            # Add environment variables
            if env_vars:
                for key, value in env_vars.items():
                    command.extend(["-e", f"{key}={value}"])
            
            # Add container name if provided
            if container_name:
                command.extend(["--name", container_name])
            
            # Add detached mode
            command.append("-d")
            
            # Add the image tag
            command.append(tag)
            
            # Run the container
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                return True, f"Docker container started: {container_id}"
            else:
                return False, f"Failed to start Docker container: {result.stderr}"
        except Exception as e:
            app_logger.error(f"Error running Docker container: {e}")
            return False, f"Error running Docker container: {str(e)}"
    
    @staticmethod
    def deploy_to_fly(app_name: Optional[str] = None) -> Tuple[bool, str]:
        """Deploy the application to Fly.io.
        
        Args:
            app_name: Optional name for the Fly.io app
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if flyctl is installed
            if not DeploymentManager.check_fly_installed():
                return False, "Fly.io CLI is not installed"
            
            # Check if fly.toml exists
            if not os.path.exists("fly.toml"):
                # If app_name is provided, create a new app
                if app_name:
                    app_logger.info(f"Creating new Fly.io app: {app_name}")
                    result = subprocess.run(
                        ["flyctl", "launch", "--name", app_name, "--no-deploy"],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    if result.returncode != 0:
                        return False, f"Failed to create Fly.io app: {result.stderr}"
                else:
                    return False, "fly.toml not found and no app name provided"
            
            # Deploy the application
            app_logger.info("Deploying to Fly.io...")
            result = subprocess.run(
                ["flyctl", "deploy"],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                # Extract the deployment URL from the output
                match = re.search(r'https://[\w.-]+\.fly\.dev', result.stdout)
                url = match.group(0) if match else "unknown URL"
                
                return True, f"Deployed to Fly.io successfully: {url}"
            else:
                return False, f"Failed to deploy to Fly.io: {result.stderr}"
        except Exception as e:
            app_logger.error(f"Error deploying to Fly.io: {e}")
            return False, f"Error deploying to Fly.io: {str(e)}"
    
    @staticmethod
    def analyze_deployment_error(error_message: str) -> Dict[str, Any]:
        """Analyze a deployment error message and provide suggestions.
        
        Args:
            error_message: The error message to analyze
            
        Returns:
            Dictionary with error analysis and suggestions
        """
        # Common error patterns and their solutions
        error_patterns = [
            {
                "pattern": r"port.*already.*use",
                "type": "port_conflict",
                "message": "Port conflict detected",
                "suggestion": "Change the port in your configuration or stop the process using that port."
            },
            {
                "pattern": r"(out of memory|memory limit exceeded)",
                "type": "memory_issue",
                "message": "Memory limit exceeded",
                "suggestion": "Increase the memory allocation for your application or optimize memory usage."
            },
            {
                "pattern": r"(disk.*full|no space left on device)",
                "type": "disk_space",
                "message": "Disk space issue",
                "suggestion": "Free up disk space or increase the disk allocation for your application."
            },
            {
                "pattern": r"(failed to build|build failed)",
                "type": "build_failure",
                "message": "Build process failed",
                "suggestion": "Check your Dockerfile and application code for errors."
            },
            {
                "pattern": r"(invalid.*configuration|config.*invalid)",
                "type": "configuration_error",
                "message": "Configuration error",
                "suggestion": "Check your fly.toml or Docker configuration for errors."
            },
            {
                "pattern": r"(authentication.*failed|not logged in|permission denied)",
                "type": "authentication_error",
                "message": "Authentication error",
                "suggestion": "Run 'flyctl auth login' to authenticate with Fly.io."
            },
            {
                "pattern": r"(network.*error|connection.*refused|timeout)",
                "type": "network_error",
                "message": "Network error",
                "suggestion": "Check your internet connection and firewall settings."
            },
            {
                "pattern": r"(ModuleNotFoundError|ImportError)",
                "type": "python_error",
                "message": "Python module error",
                "suggestion": "Check your requirements.txt and ensure all dependencies are installed."
            },
            {
                "pattern": r"(database.*error|sql.*error)",
                "type": "database_error",
                "message": "Database error",
                "suggestion": "Check your database configuration and connection settings."
            },
            {
                "pattern": r"(environment.*variable|env.*var)",
                "type": "environment_error",
                "message": "Environment variable error",
                "suggestion": "Check your environment variables and ensure they are set correctly."
            }
        ]
        
        # Check for matches
        for pattern in error_patterns:
            if re.search(pattern["pattern"], error_message, re.IGNORECASE):
                return {
                    "type": pattern["type"],
                    "message": pattern["message"],
                    "suggestion": pattern["suggestion"],
                    "original_error": error_message
                }
        
        # No specific pattern matched
        return {
            "type": "unknown_error",
            "message": "Unknown deployment error",
            "suggestion": "Check the logs for more details and ensure your application is configured correctly.",
            "original_error": error_message
        }