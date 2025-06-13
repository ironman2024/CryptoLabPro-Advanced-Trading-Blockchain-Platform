"""
Fallback implementations of essential TA-Lib functions in pure Python.
This module is used when TA-Lib installation fails.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple

def MA(close: Union[pd.Series, np.ndarray], timeperiod: int = 30, matype: int = 0) -> np.ndarray:
    """
    Moving Average fallback implementation.
    
    Args:
        close: Price series
        timeperiod: Period for calculation
        matype: Moving average type (0: SMA, 1: EMA, 2: WMA, 3: DEMA, 4: TEMA)
    
    Returns:
        numpy.ndarray: Moving average values
    """
    if isinstance(close, pd.Series):
        close = close.values
        
    if matype == 0:  # SMA
        return pd.Series(close).rolling(window=timeperiod).mean().values
    elif matype == 1:  # EMA
        return pd.Series(close).ewm(span=timeperiod, adjust=False).mean().values
    else:
        raise ValueError(f"MA type {matype} not implemented in fallback")

def RSI(close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> np.ndarray:
    """
    Relative Strength Index fallback implementation.
    
    Args:
        close: Price series
        timeperiod: Period for calculation
    
    Returns:
        numpy.ndarray: RSI values
    """
    if isinstance(close, pd.Series):
        close = close.values
        
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def MACD(close: Union[pd.Series, np.ndarray], 
         fastperiod: int = 12, 
         slowperiod: int = 26, 
         signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Moving Average Convergence/Divergence fallback implementation.
    
    Args:
        close: Price series
        fastperiod: Fast EMA period
        slowperiod: Slow EMA period
        signalperiod: Signal line period
    
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: MACD, Signal, and Histogram values
    """
    if isinstance(close, pd.Series):
        close = close.values
        
    close_series = pd.Series(close)
    fast_ema = close_series.ewm(span=fastperiod, adjust=False).mean()
    slow_ema = close_series.ewm(span=slowperiod, adjust=False).mean()
    
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    
    return macd.values, signal.values, hist.values

def BBANDS(close: Union[pd.Series, np.ndarray], 
           timeperiod: int = 5, 
           nbdevup: float = 2.0, 
           nbdevdn: float = 2.0, 
           matype: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bollinger Bands fallback implementation.
    
    Args:
        close: Price series
        timeperiod: Period for calculation
        nbdevup: Standard deviations above MA
        nbdevdn: Standard deviations below MA
        matype: Moving average type (only 0: SMA supported in fallback)
    
    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Upper, Middle, and Lower band values
    """
    if isinstance(close, pd.Series):
        close = close.values
        
    if matype != 0:
        raise ValueError("Only SMA (matype=0) is supported in fallback BBANDS")
        
    close_series = pd.Series(close)
    middle_band = close_series.rolling(window=timeperiod).mean()
    std_dev = close_series.rolling(window=timeperiod).std()
    
    upper_band = middle_band + (std_dev * nbdevup)
    lower_band = middle_band - (std_dev * nbdevdn)
    
    return upper_band.values, middle_band.values, lower_band.values

def ATR(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14) -> np.ndarray:
    """
    Average True Range fallback implementation.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        timeperiod: Period for calculation
    
    Returns:
        numpy.ndarray: ATR values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
        
    prev_close = pd.Series(close).shift(1)
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - prev_close)
    tr3 = abs(pd.Series(low) - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    
    return atr.values

def try_talib_import():
    """
    Attempts to import TA-Lib, returns True if successful, False otherwise.
    """
    try:
        import talib
        return True
    except ImportError:
        return False

# Dictionary mapping TA-Lib function names to their fallback implementations
TALIB_FALLBACKS = {
    'MA': MA,
    'RSI': RSI,
    'MACD': MACD,
    'BBANDS': BBANDS,
    'ATR': ATR
}