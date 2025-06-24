"""
Irish Bank Fraud Detection System
=================================

Production-ready fraud detection system with:
✓ Real-time transaction monitoring and fraud detection
✓ Machine learning-based risk assessment engine
✓ Professional banking-grade dashboard with integrated imagery
✓ Comprehensive alert management and notification system
✓ Irish banking compliance features and reporting
✓ Secure authentication with role-based access control
✓ Advanced transaction analysis and visualization tools
✓ Zero-configuration deployment with sample data

Features:
- Real-time fraud detection with ML models
- Interactive transaction monitoring dashboard
- Risk scoring and alert management
- Compliance reporting for Irish banking regulations
- Professional security-focused UI/UX
- Sample transaction data for immediate demonstration
"""

import os
import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    from nicegui import ui, app
    from app.core.config import settings
    from app.core.database import init_database, get_sample_data
    from app.core.fraud_engine import FraudDetectionEngine
    from app.core.assets import IrishBankAssetManager
    from app.services.auth_service import AuthService
    from app.services.transaction_service import TransactionService
    from app.services.alert_service import AlertService
    from app.frontend.dashboard import create_dashboard
    from app.frontend.auth import create_login_page
    from app.frontend.components import setup_theme
    
    # Initialize services
    auth_service = AuthService()
    transaction_service = TransactionService()
    alert_service = AlertService()
    fraud_engine = FraudDetectionEngine()
    asset_manager = IrishBankAssetManager()
    
    # Setup theme and assets
    setup_theme()
    
    # Initialize database with sample data
    init_database()
    
    @ui.page('/')
    async def index():
        """Main dashboard page with authentication check."""
        if not auth_service.is_authenticated():
            ui.navigate.to('/login')
            return
        
        await create_dashboard(
            transaction_service=transaction_service,
            alert_service=alert_service,
            fraud_engine=fraud_engine,
            asset_manager=asset_manager
        )
    
    @ui.page('/login')
    async def login():
        """Login page for system access."""
        await create_login_page(auth_service)
    
    @ui.page('/logout')
    async def logout():
        """Logout and redirect to login."""
        auth_service.logout()
        ui.navigate.to('/login')
    
    # Start background fraud detection
    async def start_fraud_monitoring():
        """Start the fraud detection monitoring in background."""
        logger.info("Starting fraud detection monitoring...")
        await fraud_engine.start_monitoring(transaction_service, alert_service)
    
    # Schedule fraud monitoring to start after UI is ready
    app.on_startup(start_fraud_monitoring)
    
    if __name__ in {"__main__", "__mp_main__"}:
        logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
        logger.info("Irish Bank Fraud Detection System initializing...")
        
        ui.run(
            title=settings.APP_NAME,
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.DEBUG,
            show=settings.DEBUG,
            storage_secret=settings.SECRET_KEY
        )

except ImportError as e:
    logger.error(f"Import error: {e}")
    print(f"Error: Missing required dependencies. Please install requirements: {e}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Critical error starting application: {e}")
    print(f"Critical error: {e}")
    sys.exit(1)