import pandas as pd
import numpy as np
from datetime import datetime

class EnhancedStrategy:
    def __init__(self, config):
        self.config = config

    def calculate_indicators(self, df):
        """Calculate advanced indicators for the strategy."""
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_55'] = df['close'].ewm(span=55, adjust=False).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
        df['atr'] = self.calculate_atr(df)
        return df

    def calculate_rsi(self, prices, period):
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices):
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_atr(self, df):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        return ranges.max(axis=1).rolling(window=14).mean()

    def generate_signals(self, df):
        """Generate trading signals based on enhanced logic."""
        df = self.calculate_indicators(df)
        df['buy_signal'] = (df['rsi'] < 30) & (df['macd'] > df['macd_signal'])
        df['sell_signal'] = (df['rsi'] > 70) & (df['macd'] < df['macd_signal'])
        df['signal'] = 'HOLD'
        df.loc[df['buy_signal'], 'signal'] = 'BUY'
        df.loc[df['sell_signal'], 'signal'] = 'SELL'
        return df[['timestamp', 'signal']]

    def backtest(self, df):
        """Run a backtest on the generated signals."""
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['returns'] * (df['signal'].shift() == 'BUY')
        total_return = df['strategy_returns'].sum()
        sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std()
        return total_return, sharpe_ratio

if __name__ == "__main__":
    # Example usage
    config = {
        "target": "LDO",
        "timeframe": "1H"
    }
    strategy = EnhancedStrategy(config)

    # Load data
    data = pd.read_csv("data/LDO_1H.csv")
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Generate signals
    signals = strategy.generate_signals(data)

    # Backtest
    total_return, sharpe_ratio = strategy.backtest(data)
    print(f"Total Return: {total_return:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")
