"""Database models and initialization for fraud detection system."""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
import json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .config import settings

# Database setup
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Transaction(Base):
    """Transaction model for fraud detection."""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    account_id = Column(String, index=True)
    amount = Column(Float)
    transaction_type = Column(String)  # debit, credit, transfer
    merchant = Column(String)
    location = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    fraud_score = Column(Float, default=0.0)
    is_fraud = Column(Boolean, default=False)
    is_flagged = Column(Boolean, default=False)
    risk_factors = Column(Text)  # JSON string of risk factors
    
class Alert(Base):
    """Alert model for fraud notifications."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    alert_type = Column(String)  # fraud, suspicious, high_risk
    severity = Column(String)  # low, medium, high, critical
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolved_by = Column(String)

class User(Base):
    """User model for system access."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="analyst")  # analyst, admin, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_database():
    """Initialize database with tables and sample data."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Add sample data if database is empty
    db = SessionLocal()
    try:
        if db.query(Transaction).count() == 0:
            _create_sample_transactions(db)
        if db.query(User).count() == 0:
            _create_sample_users(db)
        db.commit()
    finally:
        db.close()

def _create_sample_transactions(db: Session):
    """Create sample transaction data for demonstration."""
    irish_merchants = [
        "Tesco Ireland", "SuperValu", "Dunnes Stores", "Penneys", "Brown Thomas",
        "Boots Ireland", "Argos Ireland", "Harvey Norman", "DID Electrical",
        "Centra", "Spar", "Circle K", "Applegreen", "Insomnia Coffee",
        "Eddie Rockets", "Supermacs", "AIB ATM", "Bank of Ireland ATM",
        "Ulster Bank ATM", "Permanent TSB ATM", "An Post", "Vodafone Ireland",
        "Three Ireland", "Eir", "Sky Ireland", "Electric Ireland", "Bord Gáis"
    ]
    
    irish_locations = [
        "Dublin", "Cork", "Galway", "Limerick", "Waterford", "Kilkenny",
        "Wexford", "Sligo", "Drogheda", "Dundalk", "Bray", "Navan",
        "Ennis", "Tralee", "Carlow", "Naas", "Athlone", "Letterkenny",
        "Tullamore", "Killarney", "Arklow", "Cobh", "Midleton", "Bandon"
    ]
    
    # Generate 1000 sample transactions
    transactions = []
    for i in range(1000):
        # 95% normal transactions, 5% fraudulent
        is_fraud = random.random() < 0.05
        
        # Base transaction
        base_time = datetime.now() - timedelta(days=random.randint(0, 30))
        
        if is_fraud:
            # Fraudulent transaction patterns
            amount = random.choice([
                random.uniform(500, 2000),  # High amounts
                random.uniform(1, 50),      # Small amounts (card testing)
                random.uniform(2000, 5000)  # Very high amounts
            ])
            # Unusual times (late night/early morning)
            hour = random.choice([2, 3, 4, 23, 0, 1])
            timestamp = base_time.replace(hour=hour, minute=random.randint(0, 59))
            fraud_score = random.uniform(0.7, 1.0)
        else:
            # Normal transaction patterns
            amount = random.uniform(5, 500)
            # Normal business hours
            hour = random.randint(8, 22)
            timestamp = base_time.replace(hour=hour, minute=random.randint(0, 59))
            fraud_score = random.uniform(0.0, 0.3)
        
        risk_factors = {
            "unusual_time": is_fraud and hour in [2, 3, 4, 23, 0, 1],
            "high_amount": amount > 1000,
            "new_merchant": random.random() < 0.1,
            "unusual_location": random.random() < 0.05,
            "velocity_check": random.random() < 0.1
        }
        
        transaction = Transaction(
            transaction_id=f"TXN{i+1:06d}",
            account_id=f"ACC{random.randint(1000, 9999)}",
            amount=round(amount, 2),
            transaction_type=random.choice(["debit", "credit", "transfer"]),
            merchant=random.choice(irish_merchants),
            location=random.choice(irish_locations),
            timestamp=timestamp,
            fraud_score=round(fraud_score, 3),
            is_fraud=is_fraud,
            is_flagged=fraud_score > 0.5,
            risk_factors=json.dumps(risk_factors)
        )
        transactions.append(transaction)
    
    db.add_all(transactions)

def _create_sample_users(db: Session):
    """Create sample users for system access."""
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    users = [
        User(
            username="admin",
            email="admin@irishbank.ie",
            hashed_password=pwd_context.hash("admin123"),
            role="admin"
        ),
        User(
            username="analyst",
            email="analyst@irishbank.ie", 
            hashed_password=pwd_context.hash("analyst123"),
            role="analyst"
        ),
        User(
            username="viewer",
            email="viewer@irishbank.ie",
            hashed_password=pwd_context.hash("viewer123"),
            role="viewer"
        )
    ]
    
    db.add_all(users)

def get_db_session():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_sample_data() -> Dict[str, Any]:
    """Get sample data for dashboard display."""
    db = SessionLocal()
    try:
        # Get recent transactions
        recent_transactions = db.query(Transaction).order_by(
            Transaction.timestamp.desc()
        ).limit(100).all()
        
        # Get fraud statistics
        total_transactions = db.query(Transaction).count()
        fraud_transactions = db.query(Transaction).filter(Transaction.is_fraud == True).count()
        flagged_transactions = db.query(Transaction).filter(Transaction.is_flagged == True).count()
        
        # Get recent alerts
        recent_alerts = db.query(Alert).order_by(
            Alert.created_at.desc()
        ).limit(50).all()
        
        return {
            "transactions": [
                {
                    "id": t.transaction_id,
                    "account": t.account_id,
                    "amount": t.amount,
                    "merchant": t.merchant,
                    "location": t.location,
                    "timestamp": t.timestamp.isoformat(),
                    "fraud_score": t.fraud_score,
                    "is_fraud": t.is_fraud,
                    "is_flagged": t.is_flagged
                } for t in recent_transactions
            ],
            "stats": {
                "total_transactions": total_transactions,
                "fraud_transactions": fraud_transactions,
                "flagged_transactions": flagged_transactions,
                "fraud_rate": round((fraud_transactions / total_transactions) * 100, 2) if total_transactions > 0 else 0
            },
            "alerts": [
                {
                    "id": a.id,
                    "transaction_id": a.transaction_id,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "created_at": a.created_at.isoformat(),
                    "resolved": a.resolved
                } for a in recent_alerts
            ]
        }
    finally:
        db.close()