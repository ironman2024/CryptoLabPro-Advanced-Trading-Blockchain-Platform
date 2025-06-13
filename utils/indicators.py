import numpy as np
import pandas as pd
from .technical_analysis import MA, RSI, MACD, BBANDS, ATR

class TechnicalIndicators:
    @staticmethod
    def add_trend_indicators(df):
        """Calculate trend-based indicators"""
        # Moving averages
        windows = [20, 50, 200]
        for window in windows:
            df[f'sma_{window}'] = MA(df['close'], timeperiod=window, matype=0)
            
        # MACD
        macd, signal, hist = MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        return df

    @staticmethod
    def add_momentum_indicators(df):
        """Calculate momentum-based indicators"""
        # RSI
        df['rsi'] = RSI(df['close'])
        
        # ROC (Rate of Change - using pandas as it's not in TA-Lib core)
        for period in [12, 24, 48]:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
            
        return df

    @staticmethod
    def add_volatility_indicators(df):
        """Calculate volatility-based indicators"""
        # ATR
        df['atr'] = ATR(df['high'], df['low'], df['close'])
        
        # Bollinger Bands
        upper, middle, lower = BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_stddev'] = (upper - middle) / 2  # Standard deviation can be derived from bands
        
        return df