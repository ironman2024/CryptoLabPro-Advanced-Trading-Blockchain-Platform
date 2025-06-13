import pandas as pd
import numpy as np
from strategy import Strategy, DEFAULT_CONFIG

# Create a fallback module for TA-Lib functions
class TALibFallback:
    @staticmethod
    def RSI(prices, timeperiod=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=timeperiod).mean()
        avg_loss = loss.rolling(window=timeperiod).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9):
        fast_ema = prices.ewm(span=fastperiod, adjust=False).mean()
        slow_ema = prices.ewm(span=slowperiod, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    @staticmethod
    def MOM(prices, timeperiod=10):
        return prices.diff(timeperiod)
    
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=timeperiod).mean()
    
    @staticmethod
    def BBANDS(prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
        middle = prices.rolling(window=timeperiod).mean()
        std = prices.rolling(window=timeperiod).std()
        upper = middle + (std * nbdevup)
        lower = middle - (std * nbdevdn)
        return upper, middle, lower

# Initialize TALib to use fallback implementation
talib = TALibFallback()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class AdvancedStrategy(Strategy):
    """Advanced strategy with machine learning and advanced technical indicators"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/advanced_model.pkl"
        self.scaler_path = "models/advanced_scaler.pkl"
        
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
                print("Advanced model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            print("Advanced model saved successfully")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def calculate_advanced_indicators(self, df):
        """Calculate advanced technical indicators using TA-Lib or fallback implementation"""
        df = df.copy()
        
        try:
            # Use pure Python implementation with pandas Series
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['mom'] = talib.MOM(df['close'], timeperiod=10)
            
            # Volatility Indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            
        except Exception as e:
            print(f"Error calculating advanced indicators: {e}")
            # If indicators calculation fails, use basic indicators
            df['rsi'] = df['rsi_14']
            df['mom'] = df['momentum_10']
            # Create MACD from EMAs
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _prepare_features(self, df):
        """Prepare features for ML model"""
        # Basic features from parent class
        df = self.calculate_indicators(df)
        
        # Advanced features
        try:
            df = self.calculate_advanced_indicators(df)
        except:
            pass
        
        # Feature engineering
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
        
        # Add advanced features if available
        if 'stoch_k' in df.columns:
            features.append(df['stoch_k'] / 100)
            features.append(df['stoch_d'] / 100)
        
        if 'linear_reg_slope' in df.columns:
            features.append(df['linear_reg_slope'])
        
        # Stack features into a 2D array
        X = np.column_stack([f.fillna(0) for f in features])
        return X
    
    def train_model(self, historical_data):
        """Train ML model on historical data"""
        print("Training advanced model...")
        
        # Process historical data
        candles_target, candles_anchor = historical_data
        
        # Calculate indicators
        candles_target = self.calculate_indicators(candles_target)
        
        # Try to calculate advanced indicators
        try:
            candles_target = self.calculate_advanced_indicators(candles_target)
        except:
            pass
        
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
    
    def generate_signals(self, candles_target, candles_anchor):
        """Generate trading signals using advanced strategy"""
        try:
            # Calculate indicators
            candles_target = self.calculate_indicators(candles_target)
            
            # Try to calculate advanced indicators
            try:
                candles_target = self.calculate_advanced_indicators(candles_target)
            except:
                pass
            
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
            
            # If ML model is available, use it
            if self.model is not None:
                # Prepare features
                X = self._prepare_features(df)
                
                # Scale features
                X = self.scaler.transform(X)
                
                # Predict buy probabilities
                buy_probs = self.model.predict_proba(X)[:, 1]
                
                # Generate ML signals
                position_active = False
                
                for i in range(len(df)):
                    if not position_active and buy_probs[i] > 0.7:
                        df.at[df.index[i], 'signal'] = 'BUY'
                        position_active = True
                    elif position_active and (buy_probs[i] < 0.3 or i == len(df) - 1):
                        df.at[df.index[i], 'signal'] = 'SELL'
                        position_active = False
            else:
                # Fall back to traditional strategy
                traditional_signals = super().generate_signals(candles_target, candles_anchor)
                df = df.merge(traditional_signals[['timestamp', 'signal']], on='timestamp', how='left', suffixes=('', '_trad'))
                df['signal'] = df['signal_trad']
            
            print("Advanced Signals value counts:", df['signal'].value_counts())
            return df[['timestamp', 'signal']].set_index('timestamp').reindex(
                candles_target['timestamp'],
                fill_value='HOLD'
            ).reset_index()
            
        except Exception as e:
            error_info = f"Advanced signal generation error: {str(e)}"
            print(error_info)
            # Fall back to traditional strategy
            return super().generate_signals(candles_target, candles_anchor)

# Example usage
if __name__ == "__main__":
    from fetch_data import DataFetcher
    
    # Initialize strategy
    strategy = AdvancedStrategy()
    
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
    if target_data is not None and not anchor_data.empty:
        strategy.train_model((target_data, anchor_data))
        
        # Generate signals
        signals = strategy.generate_signals(target_data, anchor_data)
        print(signals.tail())