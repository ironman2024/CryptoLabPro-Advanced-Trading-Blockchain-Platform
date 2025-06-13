"""
Tests for technical analysis functions and their fallbacks.
"""

import pytest
import numpy as np
import pandas as pd
from utils.technical_analysis import MA, RSI, MACD, BBANDS, ATR
from utils.talib_fallback import TALIB_FALLBACKS

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    n_points = 100
    close = pd.Series(np.random.randn(n_points).cumsum() + 100)
    high = close + np.random.rand(n_points)
    low = close - np.random.rand(n_points)
    return high, low, close

def test_ma_calculation(sample_data):
    """Test Moving Average calculation."""
    _, _, close = sample_data
    ma = MA(close, timeperiod=5)
    
    # Basic checks
    assert len(ma) == len(close)
    assert isinstance(ma, np.ndarray)
    assert not np.isnan(ma).all()
    
    # First few values should be NaN due to window
    assert np.isnan(ma[:4]).all()
    
    # Values after window should not be NaN
    assert not np.isnan(ma[4:]).any()

def test_rsi_calculation(sample_data):
    """Test RSI calculation."""
    _, _, close = sample_data
    rsi = RSI(close)
    
    # Basic checks
    assert len(rsi) == len(close)
    assert isinstance(rsi, np.ndarray)
    
    # RSI should be between 0 and 100
    valid_rsi = rsi[~np.isnan(rsi)]
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

def test_macd_calculation(sample_data):
    """Test MACD calculation."""
    _, _, close = sample_data
    macd, signal, hist = MACD(close)
    
    # Basic checks
    assert len(macd) == len(close)
    assert len(signal) == len(close)
    assert len(hist) == len(close)
    
    # Histogram should equal MACD minus Signal
    np.testing.assert_array_almost_equal(hist, macd - signal)

def test_bbands_calculation(sample_data):
    """Test Bollinger Bands calculation."""
    _, _, close = sample_data
    upper, middle, lower = BBANDS(close)
    
    # Basic checks
    assert len(upper) == len(close)
    assert len(middle) == len(close)
    assert len(lower) == len(close)
    
    # Upper should be greater than middle, which should be greater than lower
    valid_idx = ~np.isnan(upper)
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()

def test_atr_calculation(sample_data):
    """Test Average True Range calculation."""
    high, low, close = sample_data
    atr = ATR(high, low, close)
    
    # Basic checks
    assert len(atr) == len(close)
    assert isinstance(atr, np.ndarray)
    
    # ATR should be positive
    valid_atr = atr[~np.isnan(atr)]
    assert (valid_atr >= 0).all()

def test_fallback_availability():
    """Test that all advertised fallbacks are actually implemented."""
    for func_name in TALIB_FALLBACKS:
        assert callable(TALIB_FALLBACKS[func_name])

def test_ma_types():
    """Test different types of moving averages."""
    _, _, close = sample_data()
    
    # Test SMA
    sma = MA(close, timeperiod=5, matype=0)
    assert not np.isnan(sma[4:]).any()
    
    # Test EMA
    ema = MA(close, timeperiod=5, matype=1)
    assert not np.isnan(ema[4:]).any()
    
    # Test unsupported MA type
    with pytest.raises(ValueError):
        MA(close, timeperiod=5, matype=5)