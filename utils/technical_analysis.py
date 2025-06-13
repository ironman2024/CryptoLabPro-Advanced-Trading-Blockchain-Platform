"""
Technical Analysis module using pure Python implementations.
"""

from typing import Union, Tuple, Callable
import numpy as np
import pandas as pd
from .talib_fallback import TALIB_FALLBACKS

def get_function(name: str) -> Callable:
    """
    Get the pure Python implementation for a given function name.
    
    Args:
        name: Name of the technical analysis function
        
    Returns:
        Callable: Function implementation
        
    Raises:
        ValueError: If the function is not available in the implementations
    """
    if name in TALIB_FALLBACKS:
        return TALIB_FALLBACKS[name]
    else:
        raise ValueError(f"Function {name} not available in implementations")

def MA(close: Union[pd.Series, np.ndarray], timeperiod: int = 30, matype: int = 0) -> np.ndarray:
    """Moving Average wrapper."""
    func = get_function('MA')
    return func(close, timeperiod=timeperiod, matype=matype)

def RSI(close: Union[pd.Series, np.ndarray], timeperiod: int = 14) -> np.ndarray:
    """Relative Strength Index wrapper."""
    func = get_function('RSI')
    return func(close, timeperiod=timeperiod)

def MACD(close: Union[pd.Series, np.ndarray],
         fastperiod: int = 12,
         slowperiod: int = 26,
         signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence/Divergence wrapper."""
    func = get_function('MACD')
    return func(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

def BBANDS(close: Union[pd.Series, np.ndarray],
           timeperiod: int = 5,
           nbdevup: float = 2.0,
           nbdevdn: float = 2.0,
           matype: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands wrapper."""
    func = get_function('BBANDS')
    return func(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

def ATR(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        timeperiod: int = 14) -> np.ndarray:
    """Average True Range wrapper."""
    func = get_function('ATR')
    return func(high, low, close, timeperiod=timeperiod)

# Add more wrapper functions as needed...