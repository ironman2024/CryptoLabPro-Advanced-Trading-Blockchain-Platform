import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

class MLModelManager:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_models(self, X, y):
        """Train multiple ML models with time-series cross validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Cross-validation
        scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.models['rf'].fit(X_train, y_train)
            score = self.models['rf'].score(X_val, y_val)
            scores.append(score)
            
        return np.mean(scores)
    
    def get_ensemble_predictions(self, X):
        """Get probability predictions from all models"""
        X_scaled = self.scalers['standard'].transform(X)
        return self.models['rf'].predict_proba(X_scaled)[:, 1]
