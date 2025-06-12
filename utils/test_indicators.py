import pytest
import pandas as pd
import numpy as np
from .indicators import TechnicalIndicators

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='h')
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100),
        'high': 101 + np.random.randn(100),
        'low': 99 + np.random.randn(100),
        'close': 100 + np.random.randn(100),
        'volume': 1000 + np.random.randn(100)
    })
    
    return df

def test_momentum_indicators(sample_data):
    """Test momentum indicators calculation including RSI"""
    df = TechnicalIndicators.add_momentum_indicators(sample_data)
    
    # Check RSI
    assert 'rsi' in df.columns
    assert df['rsi'].notna().any()  # Check if there are non-NaN values
    assert (df['rsi'] >= 0).all()  # RSI should be >= 0
    assert (df['rsi'] <= 100).all()  # RSI should be <= 100
    
    # Check ROC
    for period in [12, 24, 48]:
        assert f'roc_{period}' in df.columns
        assert df[f'roc_{period}'].notna().any()

def test_trend_indicators(sample_data):
    """Test trend indicators calculation"""
    df = TechnicalIndicators.add_trend_indicators(sample_data)
    
    # Check SMAs
    for window in [20, 50, 200]:
        assert f'sma_{window}' in df.columns
        assert df[f'sma_{window}'].notna().any()
    
    # Check MACD
    assert 'macd' in df.columns
    assert 'macd_signal' in df.columns
    assert 'macd_hist' in df.columns
    assert df['macd'].notna().any()

def test_volatility_indicators(sample_data):
    """Test volatility indicators calculation"""
    df = TechnicalIndicators.add_volatility_indicators(sample_data)
    
    # Check ATR
    assert 'atr' in df.columns
    assert df['atr'].notna().any()
    assert (df['atr'] >= 0).all()  # ATR should be non-negative
    
    # Check Bollinger Bands
    assert 'bb_middle' in df.columns
    assert 'bb_upper' in df.columns
    assert 'bb_lower' in df.columns
    assert df['bb_middle'].notna().any()
    assert (df['bb_upper'] >= df['bb_middle']).all()  # Upper band should be >= middle band
    assert (df['bb_middle'] >= df['bb_lower']).all()  # Middle band should be >= lower band