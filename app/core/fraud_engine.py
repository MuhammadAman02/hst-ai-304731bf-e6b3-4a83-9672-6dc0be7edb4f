"""Machine learning-based fraud detection engine."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from .config import settings
from .database import SessionLocal, Transaction

logger = logging.getLogger(__name__)

class FraudDetectionEngine:
    """Advanced fraud detection engine with ML models."""
    
    def __init__(self):
        self.isolation_forest = None
        self.random_forest = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training = None
        self.feature_columns = [
            'amount', 'hour', 'day_of_week', 'is_weekend',
            'amount_zscore', 'velocity_1h', 'velocity_24h'
        ]
        
        # Load or train models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and train fraud detection models."""
        try:
            # Try to load existing models
            self.isolation_forest = joblib.load(settings.MODELS_DIR / "isolation_forest.pkl")
            self.random_forest = joblib.load(settings.MODELS_DIR / "random_forest.pkl")
            self.scaler = joblib.load(settings.MODELS_DIR / "scaler.pkl")
            self.is_trained = True
            logger.info("Loaded existing fraud detection models")
        except FileNotFoundError:
            logger.info("No existing models found, training new models...")
            self._train_models()
    
    def _train_models(self):
        """Train fraud detection models with sample data."""
        db = SessionLocal()
        try:
            # Get training data
            transactions = db.query(Transaction).all()
            if len(transactions) < 100:
                logger.warning("Insufficient training data, using default models")
                self._create_default_models()
                return
            
            # Prepare features
            df = pd.DataFrame([{
                'amount': t.amount,
                'hour': t.timestamp.hour,
                'day_of_week': t.timestamp.weekday(),
                'is_weekend': t.timestamp.weekday() >= 5,
                'is_fraud': t.is_fraud
            } for t in transactions])
            
            # Feature engineering
            df['amount_zscore'] = np.abs((df['amount'] - df['amount'].mean()) / df['amount'].std())
            df['velocity_1h'] = df.groupby(df.index // 10)['amount'].transform('sum')  # Simplified velocity
            df['velocity_24h'] = df.groupby(df.index // 100)['amount'].transform('sum')  # Simplified velocity
            
            # Prepare training data
            X = df[self.feature_columns]
            y = df['is_fraud']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest (unsupervised anomaly detection)
            self.isolation_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.isolation_forest.fit(X_scaled)
            
            # Train Random Forest (supervised classification)
            if y.sum() > 0:  # Only if we have fraud examples
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                self.random_forest = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                )
                self.random_forest.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = self.random_forest.predict(X_test)
                logger.info(f"Model performance:\n{classification_report(y_test, y_pred)}")
            
            # Save models
            joblib.dump(self.isolation_forest, settings.MODELS_DIR / "isolation_forest.pkl")
            joblib.dump(self.random_forest, settings.MODELS_DIR / "random_forest.pkl")
            joblib.dump(self.scaler, settings.MODELS_DIR / "scaler.pkl")
            
            self.is_trained = True
            self.last_training = datetime.now()
            logger.info("Fraud detection models trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            self._create_default_models()
        finally:
            db.close()
    
    def _create_default_models(self):
        """Create default models when training data is insufficient."""
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Fit with dummy data
        dummy_data = np.random.randn(100, len(self.feature_columns))
        self.scaler.fit(dummy_data)
        self.isolation_forest.fit(self.scaler.transform(dummy_data))
        
        self.is_trained = True
        logger.info("Created default fraud detection models")
    
    def predict_fraud(self, transaction_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Predict fraud probability for a transaction."""
        if not self.is_trained:
            return 0.0, {"error": "Models not trained"}
        
        try:
            # Extract features
            features = self._extract_features(transaction_data)
            features_scaled = self.scaler.transform([features])
            
            # Get predictions
            isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
            isolation_anomaly = self.isolation_forest.predict(features_scaled)[0] == -1
            
            rf_prob = 0.5  # Default probability
            if self.random_forest:
                try:
                    rf_prob = self.random_forest.predict_proba(features_scaled)[0][1]
                except:
                    rf_prob = 0.5
            
            # Combine scores (weighted average)
            fraud_score = (0.4 * (1 - (isolation_score + 1) / 2) + 0.6 * rf_prob)
            fraud_score = max(0.0, min(1.0, fraud_score))  # Clamp to [0, 1]
            
            # Risk factors analysis
            risk_factors = self._analyze_risk_factors(transaction_data, features)
            
            return fraud_score, risk_factors
            
        except Exception as e:
            logger.error(f"Error predicting fraud: {e}")
            return 0.0, {"error": str(e)}
    
    def _extract_features(self, transaction_data: Dict[str, Any]) -> List[float]:
        """Extract features from transaction data."""
        timestamp = datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat()))
        amount = float(transaction_data.get('amount', 0))
        
        # Basic features
        features = [
            amount,
            timestamp.hour,
            timestamp.weekday(),
            1.0 if timestamp.weekday() >= 5 else 0.0,
            abs((amount - 100) / 200),  # Simplified z-score
            amount,  # Simplified velocity (would need account history in real system)
            amount * 2  # Simplified 24h velocity
        ]
        
        return features
    
    def _analyze_risk_factors(self, transaction_data: Dict[str, Any], features: List[float]) -> Dict[str, Any]:
        """Analyze specific risk factors for the transaction."""
        timestamp = datetime.fromisoformat(transaction_data.get('timestamp', datetime.now().isoformat()))
        amount = float(transaction_data.get('amount', 0))
        
        risk_factors = {
            "high_amount": amount > 1000,
            "unusual_time": timestamp.hour < 6 or timestamp.hour > 23,
            "weekend_transaction": timestamp.weekday() >= 5,
            "round_amount": amount % 50 == 0 and amount > 100,
            "late_night": timestamp.hour >= 23 or timestamp.hour <= 5,
            "high_velocity": features[5] > 500,  # Simplified velocity check
        }
        
        # Calculate overall risk level
        risk_count = sum(risk_factors.values())
        if risk_count >= 3:
            risk_level = "high"
        elif risk_count >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        risk_factors["risk_level"] = risk_level
        risk_factors["risk_count"] = risk_count
        
        return risk_factors
    
    async def start_monitoring(self, transaction_service, alert_service):
        """Start continuous fraud monitoring."""
        logger.info("Starting fraud monitoring...")
        
        while True:
            try:
                # Get recent unprocessed transactions
                recent_transactions = await transaction_service.get_recent_transactions(limit=10)
                
                for transaction in recent_transactions:
                    if transaction.get('fraud_score', 0) == 0:  # Unprocessed
                        # Predict fraud
                        fraud_score, risk_factors = self.predict_fraud(transaction)
                        
                        # Update transaction with fraud score
                        await transaction_service.update_fraud_score(
                            transaction['id'], fraud_score, risk_factors
                        )
                        
                        # Create alerts if necessary
                        if fraud_score > settings.FRAUD_THRESHOLD:
                            await alert_service.create_alert(
                                transaction_id=transaction['id'],
                                alert_type="fraud",
                                severity="critical",
                                message=f"High fraud probability detected: {fraud_score:.2f}"
                            )
                        elif fraud_score > settings.HIGH_RISK_THRESHOLD:
                            await alert_service.create_alert(
                                transaction_id=transaction['id'],
                                alert_type="suspicious",
                                severity="high",
                                message=f"Suspicious transaction detected: {fraud_score:.2f}"
                            )
                
                # Check if models need retraining
                if (self.last_training is None or 
                    datetime.now() - self.last_training > timedelta(seconds=settings.MODEL_UPDATE_INTERVAL)):
                    logger.info("Retraining fraud detection models...")
                    self._train_models()
                
                await asyncio.sleep(settings.MONITORING_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in fraud monitoring: {e}")
                await asyncio.sleep(settings.MONITORING_INTERVAL)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models."""
        return {
            "is_trained": self.is_trained,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "models": {
                "isolation_forest": str(type(self.isolation_forest).__name__) if self.isolation_forest else None,
                "random_forest": str(type(self.random_forest).__name__) if self.random_forest else None
            },
            "feature_columns": self.feature_columns,
            "thresholds": {
                "fraud_threshold": settings.FRAUD_THRESHOLD,
                "high_risk_threshold": settings.HIGH_RISK_THRESHOLD
            }
        }