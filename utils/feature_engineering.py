import pandas as pd
import numpy as np

class FeatureEngineer:
    """Enhanced feature engineering with alternative data sources"""
    
    def __init__(self):
        self.scalers = {}
        
    def create_features(self, df, anchor_data=None):
        features = pd.DataFrame(index=df.index)
        
        # Technical features
        features = pd.concat([
            features,
            self._create_price_features(df),
            self._create_volume_features(df),
            self._create_volatility_features(df)
        ], axis=1)
        
        # Add anchor correlations if provided
        if anchor_data is not None:
            features = pd.concat([
                features,
                self._create_correlation_features(df, anchor_data)
            ], axis=1)
            
        # Add funding rate features if available
        funding_features = self._create_funding_features(df)
        if funding_features is not None:
            features = pd.concat([features, funding_features], axis=1)
        
        return features
    
    def _create_price_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Price momentum features
        for window in [4, 12, 24]:
            # Returns
            features[f'return_{window}h'] = df['close'].pct_change(window)
            # Log returns
            features[f'log_return_{window}h'] = np.log1p(features[f'return_{window}h'])
            # Volatility-adjusted returns
            vol = df['close'].pct_change().rolling(window).std()
            features[f'vol_adj_return_{window}h'] = features[f'return_{window}h'] / vol
        
        return features
        
    def _create_volume_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Basic volume features
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(24).mean()
        features['volume_std'] = df['volume'].rolling(24).std() / df['volume'].rolling(24).mean()
        
        # Buy/Sell volume pressure
        features['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(float)
        features['sell_volume'] = df['volume'] * (df['close'] < df['open']).astype(float)
        features['buy_sell_ratio'] = (features['buy_volume'].rolling(12).mean() / 
                                    features['sell_volume'].rolling(12).mean())
        
        return features
    
    def _create_volatility_features(self, df):
        features = pd.DataFrame(index=df.index)
        
        # Volatility windows
        for window in [12, 24, 48]:
            returns = df['close'].pct_change()
            features[f'volatility_{window}h'] = returns.rolling(window).std()
            features[f'volatility_ma_{window}h'] = features[f'volatility_{window}h'].rolling(window//2).mean()
        
        # Parkinson volatility estimator
        features['parkinsons_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(df['high'] / df['low'])) ** 2)
        ).rolling(24).mean()
        
        return features
        
    def _create_correlation_features(self, df, anchor_data):
        features = pd.DataFrame(index=df.index)
        
        # Rolling correlations with anchors
        for window in [12, 24, 48]:
            for col in anchor_data.columns:
                if col.startswith('close_'):
                    symbol = col.split('_')[1]
                    corr = df['close'].rolling(window).corr(anchor_data[col])
                    features[f'corr_{symbol}_{window}h'] = corr
                    
        return features
        
    def _create_funding_features(self, df):
        """Create futures market features if available"""
        try:
            # Funding rate features (if available)
            if 'funding_rate' in df.columns:
                features = pd.DataFrame(index=df.index)
                
                # Rolling stats of funding
                for window in [8, 24, 72]:
                    features[f'funding_ma_{window}h'] = df['funding_rate'].rolling(window).mean()
                    features[f'funding_std_{window}h'] = df['funding_rate'].rolling(window).std()
                
                # Funding rate extremes
                features['funding_zscore'] = (
                    (df['funding_rate'] - features['funding_ma_24h']) / 
                    features['funding_std_24h']
                )
                
                return features
            return None
            
        except Exception:
            return None
            
    def scale_features(self, X_train, X_val=None):
        """Scale features with reversible transforms"""
        X_scaled_train = pd.DataFrame(index=X_train.index)
        X_scaled_val = pd.DataFrame(index=X_val.index) if X_val is not None else None
        
        for col in X_train.columns:
            if col not in self.scalers:
                # Use robust scaling based on quantiles
                q1 = X_train[col].quantile(0.25)
                q3 = X_train[col].quantile(0.75)
                median = X_train[col].median()
                iqr = q3 - q1
                self.scalers[col] = {
                    'median': median,
                    'iqr': iqr if iqr != 0 else 1.0
                }
            
            # Scale train data
            X_scaled_train[col] = ((X_train[col] - self.scalers[col]['median']) / 
                                 self.scalers[col]['iqr'])
            
            # Scale validation data if provided
            if X_val is not None:
                X_scaled_val[col] = ((X_val[col] - self.scalers[col]['median']) / 
                                   self.scalers[col]['iqr'])
                
        return (X_scaled_train, X_scaled_val) if X_val is not None else X_scaled_train
