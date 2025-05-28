import numpy as np
import pandas as pd

class TechnicalIndicators:
    @staticmethod
    def add_trend_indicators(df):
        """Calculate trend-based indicators"""
        # Moving averages
        windows = [20, 50, 200]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df

    @staticmethod
    def add_momentum_indicators(df):
        """Calculate momentum-based indicators"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ROC
        for period in [12, 24, 48]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
            
        return df

    @staticmethod
    def add_volatility_indicators(df):
        """Calculate volatility-based indicators"""
        # ATR
        high_low = df['high'] - df['low']
        high_cp = np.abs(df['high'] - df['close'].shift())
        low_cp = np.abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_stddev'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_stddev'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_stddev'] * 2)
        
        return df
