import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

class StrategyBase:
    """Base class for all trading strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "target": {"symbol": "LDO", "timeframe": "1h"},
            "anchors": [
                {"symbol": "BTC", "timeframe": "1h"},
                {"symbol": "ETH", "timeframe": "1h"},
                {"symbol": "SOL", "timeframe": "1h"}
            ],
            "params": {
                "lookback": 500,
                "min_trades": 10,
                "win_rate_target": 0.45,
                "profit_target": 0.004,
                "max_drawdown": 0.035
            }
        }
        self.target_symbol = self.config['target']['symbol']
        self.timeframe = self.config['target']['timeframe']
        self.anchor_symbols = [anchor['symbol'] for anchor in self.config['anchors']]

    def get_live_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Fetch live data for target and anchor assets."""
        try:
            target_data = self._get_recent_data(self.target_symbol, self.timeframe)
            if target_data is None:
                return None, None

            anchor_data = pd.DataFrame()
            for symbol in self.anchor_symbols:
                data = self._get_recent_data(symbol, self.timeframe)
                if data is not None:
                    anchor_data[f'close_{symbol}'] = data['close']
                    anchor_data[f'volume_{symbol}'] = data['volume']
                    anchor_data['timestamp'] = data['timestamp']

            if anchor_data.empty:
                return None, None

            return target_data, anchor_data
        except Exception as e:
            print(f"Error fetching live data: {str(e)}")
            return None, None

    def _get_recent_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Get recent data for a symbol."""
        try:
            interval_lower = interval.lower()
            df = pd.read_csv(f'data/{symbol}_{interval_lower}.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
            df = df.sort_values('timestamp', ascending=True).tail(500)
            return df
        except Exception as e:
            print(f"Error reading data for {symbol}: {str(e)}")
            return None

    def get_live_signal(self) -> Tuple[str, str]:
        """Get live trading signal."""
        try:
            target_data, anchor_data = self.get_live_data()
            if target_data is None or anchor_data is None:
                return "HOLD", "No data available"
            signals = self.generate_signals(target_data, anchor_data)
            latest_signal = signals.iloc[-1]['signal']
            latest_price = target_data.iloc[-1]['close']
            return latest_signal, f"Price: ${latest_price:.4f}"
        except Exception as e:
            error_info = f"Live signal error: {str(e)}"
            print(error_info)
            return "HOLD", error_info

    def generate_signals(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata."""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
            "target": self.config["target"],
            "anchors": self.config["anchors"]
        }