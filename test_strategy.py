import pytest
import pandas as pd
import numpy as np
from strategy import Strategy, DEFAULT_CONFIG

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2025-01-01', periods=100, freq='h')
    
    # Target data
    target = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.random.randn(100),
        'high': 101 + np.random.randn(100),
        'low': 99 + np.random.randn(100),
        'close': 100 + np.random.randn(100),
        'volume': 1000 + np.random.randn(100)
    })
    
    # Anchor data
    anchor = pd.DataFrame({'timestamp': dates})
    for symbol in ['BTC', 'ETH', 'SOL']:
        anchor[f'close_{symbol}'] = 100 + np.random.randn(100)
        anchor[f'volume_{symbol}'] = 1000 + np.random.randn(100)
    
    return target, anchor

def test_strategy_initialization():
    """Test strategy initialization with default and custom configs"""
    # Test default config
    strategy = Strategy()
    assert strategy.target_symbol == "LDO"
    
    # Test custom config
    custom_config = {
        "target": {"symbol": "BONK", "timeframe": "1H"},
        "anchors": [{"symbol": "BTC", "timeframe": "1H"}]
    }
    strategy = Strategy(custom_config)
    assert strategy.target_symbol == "BONK"

def test_generate_signals(sample_data):
    """Test signal generation"""
    target, anchor = sample_data
    strategy = Strategy()
    
    # Test signal generation
    signals = strategy.generate_signals(target, anchor)
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert len(signals) == len(target)
    assert signals['signal'].isin(['BUY', 'SELL', 'HOLD']).all()

def test_indicator_calculation(sample_data):
    """Test indicator calculations"""
    target, _ = sample_data
    strategy = Strategy()
    
    df = strategy.calculate_enhanced_indicators(target)
    assert isinstance(df, pd.DataFrame)
    assert 'ema_8' in df.columns
    assert 'rsi_14' in df.columns
    assert 'atr' in df.columns

def test_anchor_signal_calculation(sample_data):
    """Test anchor signal calculations"""
    target, anchor = sample_data
    strategy = Strategy()
    
    df = pd.merge(target, anchor, on='timestamp', how='inner')
    df = strategy.calculate_anchor_signals(df)
    assert isinstance(df, pd.DataFrame)
    assert 'market_regime' in df.columns
    assert 'market_strength' in df.columns

def test_live_signal(sample_data):
    """Test live signal generation"""
    target, anchor = sample_data
    strategy = Strategy()
    
    # Mock get_live_data
    strategy.get_live_data = lambda: (target, anchor)
    signal, message = strategy.get_live_signal()
    assert signal in ['BUY', 'SELL', 'HOLD']
    assert isinstance(message, str)
