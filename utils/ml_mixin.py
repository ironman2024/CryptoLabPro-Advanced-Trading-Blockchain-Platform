import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, List, Optional, Any

class MLMixin:
    """Mixin class providing machine learning capabilities"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = f"models/{self.__class__.__name__.lower()}_model.pkl"
        self.scaler_path = f"models/{self.__class__.__name__.lower()}_scaler.pkl"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Load model if exists
        self._load_model()

    def _load_model(self):
        """Load trained model if exists"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def _save_model(self):
        """Save trained model"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model."""
        features = []
        
        # Price-based features
        features.append(df['ema_8'] / df['ema_21'] - 1)  # EMA ratio
        features.append(df['ema_13_slope'])  # EMA slope
        
        # Momentum features
        features.append(df['momentum_3'])
        features.append(df['momentum_5'])
        features.append(df['momentum_10'])
        features.append(df['rsi_14'] / 100)  # Normalize RSI
        
        # MACD features
        features.append(df['macd_hist'])
        features.append(df['macd_hist_slope'])
        
        # Volume features
        features.append(df['volume_ratio'])
        features.append(df['volume_trend'])
        
        # Bollinger Band features
        features.append(df['bb_position'])
        features.append(df['bb_squeeze'].astype(float))
        
        # Market regime features
        features.append(df['market_regime'] / 2)  # Normalize market regime
        features.append(df['market_strength'])
        
        # Anchor asset features
        if 'btc_momentum_5' in df.columns:
            features.append(df['btc_momentum_5'])
            features.append(df['eth_momentum_5'])
        
        # Stack features into a 2D array
        X = np.column_stack([f.fillna(0) for f in features])
        return X

    def train_model(self, historical_data: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[float, float]:
        """Train ML model on historical data."""
        print("Training ML model...")
        
        # Process historical data
        candles_target, candles_anchor = historical_data
        
        # Calculate indicators
        candles_target = self.calculate_indicators(candles_target)
        
        # Prepare anchor data
        candles_anchor = candles_anchor.copy()
        anchor_cols = [col for col in candles_anchor.columns if col.startswith('close_')]
        for col in anchor_cols:
            candles_anchor[col] = candles_anchor[col].astype(float)
        
        # Merge data
        df = pd.merge(candles_target, candles_anchor, on='timestamp', how='inner')
        df = df.dropna().reset_index(drop=True)
        df = self.calculate_anchor_signals(df)
        
        # Generate labels using traditional strategy
        signals = self.generate_traditional_signals(candles_target, candles_anchor)
        df = df.merge(signals, on='timestamp', how='left')
        
        # Convert signals to numeric labels
        df['label'] = 0
        df.loc[df['signal'] == 'BUY', 'label'] = 1
        df.loc[df['signal'] == 'SELL', 'label'] = -1
        
        # Prepare features
        X = self._prepare_features(df)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Prepare labels (predict buy signals)
        y = (df['label'] == 1).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model training complete. Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")
        
        # Save model
        self._save_model()
        
        return train_score, test_score

    def predict_signals(self, df: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
        """Generate ML-based signals."""
        if self.model is None:
            return pd.Series('HOLD', index=df.index), np.zeros(len(df))
            
        # Prepare features
        X = self._prepare_features(df)
        
        # Scale features
        X = self.scaler.transform(X)
        
        # Predict buy probabilities
        buy_probs = self.model.predict_proba(X)[:, 1]
        
        # Generate ML signals
        signals = pd.Series('HOLD', index=df.index)
        position_active = False
        
        for i in range(len(df)):
            if not position_active and buy_probs[i] > 0.7:
                signals.iloc[i] = 'BUY'
                position_active = True
            elif position_active and (buy_probs[i] < 0.3 or i == len(df) - 1):
                signals.iloc[i] = 'SELL'
                position_active = False
                
        return signals, buy_probs