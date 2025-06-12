import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

class IndicatorsMixin:
    """Mixin class providing technical indicator calculations"""
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core technical indicators."""
        df = df.copy()
        df[['close', 'volume', 'high', 'low']] = df[['close', 'volume', 'high', 'low']].astype(float)
        
        # Multiple EMA periods for trend detection
        for period in [8, 13, 21, 34, 55]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # EMA slope for trend strength
        df['ema_13_slope'] = df['ema_13'].diff(3) / df['ema_13'].shift(3)
        df['ema_21_slope'] = df['ema_21'].diff(5) / df['ema_21'].shift(5)
        
        # RSI calculations
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_21'] = self._calculate_rsi(df['close'], 21)
        
        # RSI divergence signals
        df['price_higher_high'] = (df['close'] > df['close'].shift(10)) & (df['close'].shift(5) < df['close'].shift(15))
        df['rsi_lower_high'] = (df['rsi_14'] < df['rsi_14'].shift(10)) & (df['rsi_14'].shift(5) > df['rsi_14'].shift(15))
        df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        df['macd_hist_slope'] = df['macd_hist'].diff(2)
        
        # Volume analysis
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_30'] = df['volume'].rolling(window=30).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_30']
        df['volume_trend'] = df['volume_sma_10'] / df['volume_sma_30']
        
        # Momentum
        df['momentum_3'] = df['close'] / df['close'].shift(3) - 1
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        df['atr_pct'] = df['atr'] / df['close']
        
        # Bollinger Bands
        df = self._calculate_bollinger_bands(df)
        
        # Support/Resistance
        df['recent_high'] = df['high'].rolling(window=20).max()
        df['recent_low'] = df['low'].rolling(window=20).min()
        df['near_resistance'] = df['close'] > (df['recent_high'] * 0.98)
        df['near_support'] = df['close'] < (df['recent_low'] * 1.02)
        
        return df

    def calculate_anchor_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate anchor asset indicators and market regime."""
        df = df.copy()
        
        # Calculate momentum and correlation for anchor assets
        for symbol in ['BTC', 'ETH']:
            df[f'{symbol.lower()}_momentum_3'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(3) - 1)
            df[f'{symbol.lower()}_momentum_5'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(5) - 1)
            df[f'{symbol.lower()}_momentum_10'] = (df[f'close_{symbol}'] / df[f'close_{symbol}'].shift(10) - 1)
            
            # RSI for anchors
            df[f'{symbol.lower()}_rsi'] = self._calculate_rsi(df[f'close_{symbol}'], 14)
            
            # Volume analysis
            df[f'{symbol.lower()}_volume_ratio'] = df[f'volume_{symbol}'] / df[f'volume_{symbol}'].rolling(window=20).mean()
        
        # Correlation analysis
        df['target_btc_corr'] = df['close'].rolling(window=20).corr(df['close_BTC'])
        df['target_eth_corr'] = df['close'].rolling(window=20).corr(df['close_ETH'])
        
        # Market regime classification
        df = self._classify_market_regime(df)
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR indicator."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).rolling(window=14).mean()

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_squeeze'] = df['bb_std'] < df['bb_std'].rolling(window=20).mean() * 0.8
        return df

    def _classify_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify market regime based on anchor assets."""
        strong_bull = (
            (df['btc_momentum_5'] > 0.02) & 
            (df['eth_momentum_5'] > 0.02) &
            (df['btc_rsi'] > 55) & 
            (df['eth_rsi'] > 55)
        )
        
        mild_bull = (
            (df['btc_momentum_10'] > 0) | 
            (df['eth_momentum_10'] > 0) |
            (df['btc_rsi'] > 50) | 
            (df['eth_rsi'] > 50)
        )
        
        df['market_regime'] = 0  # Bearish
        df.loc[mild_bull, 'market_regime'] = 1  # Mildly bullish  
        df.loc[strong_bull, 'market_regime'] = 2  # Strongly bullish
        
        # Composite market strength score
        df['market_strength'] = (
            df['btc_momentum_5'] * 0.3 +
            df['eth_momentum_5'] * 0.3 +
            (df['btc_rsi'] - 50) / 50 * 0.2 +
            (df['eth_rsi'] - 50) / 50 * 0.2
        )
        
        return df