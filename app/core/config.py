"""Configuration settings for the fraud detection system."""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application Info
    APP_NAME: str = "Irish Bank Fraud Detection System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Real-time fraud detection and prevention system"
    
    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8080
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "irish-bank-fraud-detection-secret-key-2024"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/fraud_detection.db"
    
    # Fraud Detection Settings
    FRAUD_THRESHOLD: float = 0.7
    HIGH_RISK_THRESHOLD: float = 0.5
    MONITORING_INTERVAL: int = 5  # seconds
    
    # Irish Banking Compliance
    CENTRAL_BANK_REPORTING: bool = True
    GDPR_COMPLIANCE: bool = True
    PCI_DSS_COMPLIANCE: bool = True
    
    # Alert Settings
    EMAIL_ALERTS: bool = True
    SMS_ALERTS: bool = True
    WEBHOOK_ALERTS: bool = True
    
    # ML Model Settings
    MODEL_UPDATE_INTERVAL: int = 3600  # seconds (1 hour)
    TRAINING_DATA_RETENTION: int = 90  # days
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    
    # File Paths
    DATA_DIR: Path = Path("./data")
    LOGS_DIR: Path = Path("./logs")
    MODELS_DIR: Path = Path("./models")
    
    @validator('DATA_DIR', 'LOGS_DIR', 'MODELS_DIR')
    def create_directories(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()