import pandas as pd
import numpy as np
from strategy import Strategy, DEFAULT_CONFIG
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EnsembleStrategy(Strategy):
    """Enhanced strategy that combines multiple algorithms for better performance"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/ensemble_model.pkl"
        self.scaler_path = "models/scaler.pkl"
        
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
                print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _prepare_features(self, df):
        """Prepare features for ML model"""
        features = []
        
        # Price-based features
        features.append(df['ema_8'] / df['ema_21'] - 1)  # EMA ratio
        features.append(df['ema_13_slope'])  # EMA slope
        
        # Momentum features
        features.append(df['momentum_3'])
        features.append(df['momentum_5'])
        features.append(df['momentum_10'])
        
        # RSI features
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
        X = np.column_stack(features)
        return X
    
    def train_model(self, historical_data, lookback=500):
        """Train ML model on historical data"""
        print("Training ensemble model...")
        
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
        signals = super().generate_signals(candles_target, candles_anchor)
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
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            random_state=42
        )
        self.model.fit(X, y)
        
        # Save model
        self._save_model()
        
        print("Model training complete")
    
    def generate_signals(self, candles_target, candles_anchor):
        """Generate trading signals using ensemble of algorithms"""
        try:
            # Calculate indicators using parent method
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
            
            # Initialize signal column
            df['signal'] = 'HOLD'
            
            # Generate signals using traditional strategy
            traditional_signals = super().generate_signals(candles_target, candles_anchor)
            df = df.merge(traditional_signals[['timestamp', 'signal']], on='timestamp', how='left', suffixes=('', '_trad'))
            df['signal'] = df['signal_trad']
            
            # If ML model is available, enhance signals
            if self.model is not None:
                # Prepare features
                X = self._prepare_features(df)
                
                # Scale features
                X = self.scaler.transform(X)
                
                # Predict buy probabilities
                buy_probs = self.model.predict_proba(X)[:, 1]
                
                # Generate ML signals
                ml_signals = pd.Series('HOLD', index=df.index)
                ml_signals[buy_probs > 0.7] = 'BUY'  # High confidence for buy
                
                # Combine signals (ML overrides traditional for BUY)
                df.loc[ml_signals == 'BUY', 'signal'] = 'BUY'
                
                # Generate SELL signals
                position_active = False
                for i in range(len(df)):
                    if not position_active and df.iloc[i]['signal'] == 'BUY':
                        position_active = True
                    elif position_active:
                        # Sell when probability drops or traditional strategy says sell
                        if buy_probs[i] < 0.3 or df.iloc[i]['signal_trad'] == 'SELL':
                            df.iloc[i, df.columns.get_loc('signal')] = 'SELL'
                            position_active = False
            
            print("Enhanced Signals value counts:", df['signal'].value_counts())
            return df[['timestamp', 'signal']].set_index('timestamp').reindex(
                candles_target['timestamp'],
                fill_value='HOLD'
            ).reset_index()
            
        except Exception as e:
            error_info = f"Signal generation error: {str(e)}"
            print(error_info)
            return pd.DataFrame({'timestamp': candles_target['timestamp'], 'signal': 'HOLD'})

# Example usage
if __name__ == "__main__":
    from fetch_data import DataFetcher
    
    # Initialize strategy
    strategy = EnsembleStrategy()
    
    # Get historical data
    data_fetcher = DataFetcher()
    target_data = data_fetcher.get_recent_data("LDO", "1h")
    
    # Get anchor data
    anchor_data = pd.DataFrame()
    for symbol in ["BTC", "ETH", "SOL"]:
        data = data_fetcher.get_recent_data(symbol, "1h")
        if data is not None:
            anchor_data[f'close_{symbol}'] = data['close']
            anchor_data[f'volume_{symbol}'] = data['volume']
            anchor_data['timestamp'] = data['timestamp']
    
    # Train model
    strategy.train_model((target_data, anchor_data))
    
    # Generate signals
    signals = strategy.generate_signals(target_data, anchor_data)
    print(signals.tail())