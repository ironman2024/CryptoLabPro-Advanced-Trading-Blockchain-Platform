import pandas as pd
import numpy as np

def get_coin_metadata():
    """
    Define strategy metadata per challenge requirements
    """
    return {
        "target": {"symbol": "LDO", "timeframe": "1H"},
        "anchors": [
            {"symbol": "BTC", "timeframe": "1H"},
            {"symbol": "ETH", "timeframe": "1H"},
            {"symbol": "SOL", "timeframe": "1H"}
        ]
    }

def calculate_lagged_indicators(price_series, lookback=24, lag=2):
    """
    Calculate indicators based on lagged price action
    
    Args:
        price_series: Series of closing prices
        lookback: Rolling window size for calculations
        lag: Number of periods to lag
        
    Returns:
        Dictionary of indicators
    """
    indicators = {}
    
    # Calculate returns and momentum
    returns = price_series.pct_change()
    lagged_returns = returns.shift(lag)
    
    # Rolling stats
    indicators['momentum'] = returns.rolling(lookback).mean()
    indicators['volatility'] = returns.rolling(lookback).std()
    
    # Lag-adjusted signals  
    indicators['lagged_signal'] = lagged_returns.rolling(lookback).mean() / indicators['volatility']
    
    return indicators

def generate_signals(candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on lagged relationships with anchor coins
    """
    try:
        df = pd.DataFrame({'timestamp': candles_target['timestamp']})
        signals = pd.Series('HOLD', index=df.index)
        
        # Calculate lagged indicators for each anchor
        btc_ind = calculate_lagged_indicators(candles_anchor['close_BTC'])
        eth_ind = calculate_lagged_indicators(candles_anchor['close_ETH']) 
        sol_ind = calculate_lagged_indicators(candles_anchor['close_SOL'])
        
        # Combined signal thresholds
        momentum_threshold = 1.5
        volatility_threshold = 0.015
        
        # Generate BUY signals
        buy_conditions = (
            (btc_ind['lagged_signal'] > momentum_threshold) &
            (eth_ind['lagged_signal'] > momentum_threshold) &
            (sol_ind['lagged_signal'] > momentum_threshold) &
            (btc_ind['volatility'] < volatility_threshold) &
            (eth_ind['volatility'] < volatility_threshold)
        )
        
        # Generate SELL signals  
        sell_conditions = (
            (btc_ind['lagged_signal'] < -momentum_threshold) |
            (eth_ind['lagged_signal'] < -momentum_threshold) |
            (sol_ind['lagged_signal'] < -momentum_threshold) |
            (btc_ind['volatility'] > volatility_threshold * 2)
        )
        
        # Apply signals
        signals[buy_conditions] = 'BUY'
        signals[sell_conditions] = 'SELL'
        
        # Ensure no NaN values
        signals = signals.fillna('HOLD')
        
        # Return required format
        return pd.DataFrame({
            'timestamp': df['timestamp'],
            'signal': signals
        })
        
    except Exception as e:
        raise RuntimeError(f"Strategy failed: {e}")

