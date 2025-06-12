import pandas as pd
import numpy as np
from datetime import datetime

class TradingStrategies:
    """Collection of trading strategies for crypto markets"""
    
    @staticmethod
    def ema_crossover_strategy(price_data, fast_period=9, slow_period=21):
        """EMA Crossover Strategy
        
        Buy when fast EMA crosses above slow EMA
        Sell when fast EMA crosses below slow EMA
        """
        # Calculate EMAs
        price_data['ema_fast'] = price_data['close'].ewm(span=fast_period, adjust=False).mean()
        price_data['ema_slow'] = price_data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Initialize signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Generate signals
        position = 0
        
        for i in range(1, len(price_data)):
            # Buy signal: fast EMA crosses above slow EMA
            if (price_data['ema_fast'].iloc[i-1] <= price_data['ema_slow'].iloc[i-1] and 
                price_data['ema_fast'].iloc[i] > price_data['ema_slow'].iloc[i] and
                position == 0):
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: fast EMA crosses below slow EMA
            elif (price_data['ema_fast'].iloc[i-1] >= price_data['ema_slow'].iloc[i-1] and 
                  price_data['ema_fast'].iloc[i] < price_data['ema_slow'].iloc[i] and
                  position == 1):
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def rsi_strategy(price_data, rsi_period=14, oversold=30, overbought=70):
        """RSI Strategy
        
        Buy when RSI crosses above oversold level
        Sell when RSI crosses below overbought level
        """
        # Calculate RSI
        delta = price_data['close'].diff()
        gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
        loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
        rs = gain / (loss + 1e-8)
        price_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Initialize signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Generate signals
        position = 0
        
        for i in range(1, len(price_data)):
            # Buy signal: RSI crosses above oversold level
            if (price_data['rsi'].iloc[i-1] <= oversold and 
                price_data['rsi'].iloc[i] > oversold and
                position == 0):
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: RSI crosses below overbought level
            elif (price_data['rsi'].iloc[i-1] >= overbought and 
                  price_data['rsi'].iloc[i] < overbought and
                  position == 1):
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def macd_strategy(price_data, fast_period=12, slow_period=26, signal_period=9):
        """MACD Strategy
        
        Buy when MACD line crosses above signal line
        Sell when MACD line crosses below signal line
        """
        # Calculate MACD
        price_data['ema_fast'] = price_data['close'].ewm(span=fast_period, adjust=False).mean()
        price_data['ema_slow'] = price_data['close'].ewm(span=slow_period, adjust=False).mean()
        price_data['macd'] = price_data['ema_fast'] - price_data['ema_slow']
        price_data['macd_signal'] = price_data['macd'].ewm(span=signal_period, adjust=False).mean()
        price_data['macd_hist'] = price_data['macd'] - price_data['macd_signal']
        
        # Initialize signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Generate signals
        position = 0
        
        for i in range(1, len(price_data)):
            # Buy signal: MACD crosses above signal line
            if (price_data['macd'].iloc[i-1] <= price_data['macd_signal'].iloc[i-1] and 
                price_data['macd'].iloc[i] > price_data['macd_signal'].iloc[i] and
                position == 0):
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: MACD crosses below signal line
            elif (price_data['macd'].iloc[i-1] >= price_data['macd_signal'].iloc[i-1] and 
                  price_data['macd'].iloc[i] < price_data['macd_signal'].iloc[i] and
                  position == 1):
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def bollinger_bands_strategy(price_data, window=20, num_std=2):
        """Bollinger Bands Strategy
        
        Buy when price touches lower band
        Sell when price touches upper band
        """
        # Calculate Bollinger Bands
        price_data['sma'] = price_data['close'].rolling(window=window).mean()
        price_data['std'] = price_data['close'].rolling(window=window).std()
        price_data['upper_band'] = price_data['sma'] + (price_data['std'] * num_std)
        price_data['lower_band'] = price_data['sma'] - (price_data['std'] * num_std)
        
        # Initialize signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Generate signals
        position = 0
        
        for i in range(1, len(price_data)):
            # Buy signal: price touches lower band
            if (price_data['close'].iloc[i] <= price_data['lower_band'].iloc[i] and
                position == 0):
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: price touches upper band
            elif (price_data['close'].iloc[i] >= price_data['upper_band'].iloc[i] and
                  position == 1):
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def support_resistance_strategy(price_data, lookback=50, threshold=0.02):
        """Support and Resistance Strategy
        
        Buy when price bounces off support
        Sell when price hits resistance
        """
        # Initialize signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Need enough data for lookback
        if len(price_data) < lookback:
            return signals
        
        # Generate signals
        position = 0
        
        for i in range(lookback, len(price_data)):
            # Get recent price action
            recent_data = price_data.iloc[i-lookback:i]
            
            # Find support and resistance
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            
            current_price = price_data['close'].iloc[i]
            
            # Buy signal: price bounces off support
            if (current_price <= support * (1 + threshold) and
                current_price > support and
                position == 0):
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: price hits resistance
            elif (current_price >= resistance * (1 - threshold) and
                  current_price < resistance and
                  position == 1):
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def multi_strategy_consensus(price_data):
        """Multi-Strategy Consensus
        
        Combines multiple strategies and generates signals based on consensus
        """
        # Get signals from individual strategies
        ema_signals = TradingStrategies.ema_crossover_strategy(price_data.copy())
        rsi_signals = TradingStrategies.rsi_strategy(price_data.copy())
        macd_signals = TradingStrategies.macd_strategy(price_data.copy())
        bb_signals = TradingStrategies.bollinger_bands_strategy(price_data.copy())
        
        # Initialize consensus signals
        signals = pd.DataFrame(index=price_data.index)
        signals['timestamp'] = price_data['timestamp']
        signals['signal'] = 'HOLD'
        
        # Generate consensus signals
        position = 0
        
        for i in range(1, len(price_data)):
            # Count buy and sell signals
            buy_count = sum(1 for s in [ema_signals, rsi_signals, macd_signals, bb_signals] 
                           if s['signal'].iloc[i] == 'BUY')
            sell_count = sum(1 for s in [ema_signals, rsi_signals, macd_signals, bb_signals] 
                            if s['signal'].iloc[i] == 'SELL')
            
            # Buy signal: majority of strategies say buy
            if buy_count >= 2 and position == 0:
                signals['signal'].iloc[i] = 'BUY'
                position = 1
            
            # Sell signal: majority of strategies say sell
            elif sell_count >= 2 and position == 1:
                signals['signal'].iloc[i] = 'SELL'
                position = 0
        
        return signals
    
    @staticmethod
    def get_strategy_performance(price_data, signals, initial_capital=10000, transaction_cost=0.001):
        """Calculate strategy performance metrics"""
        # Initialize portfolio and positions
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['timestamp'] = signals['timestamp']
        portfolio['signal'] = signals['signal']
        portfolio['price'] = price_data['close']
        portfolio['cash'] = initial_capital
        portfolio['position'] = 0
        portfolio['holdings'] = 0
        portfolio['total'] = initial_capital
        
        # Track trades
        trades = []
        
        # Simulate trading
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(portfolio)):
            if portfolio['signal'].iloc[i] == 'BUY' and position == 0:
                # Buy with all cash
                price = portfolio['price'].iloc[i] * (1 + transaction_cost)  # Include transaction cost
                position = portfolio['cash'].iloc[i] / price
                entry_price = price
                entry_time = portfolio['timestamp'].iloc[i]
                
                portfolio['position'].iloc[i:] = position
                portfolio['cash'].iloc[i:] = 0
            
            elif portfolio['signal'].iloc[i] == 'SELL' and position > 0:
                # Sell all position
                price = portfolio['price'].iloc[i] * (1 - transaction_cost)  # Include transaction cost
                cash = position * price
                
                # Record trade
                exit_time = portfolio['timestamp'].iloc[i]
                profit = (price - entry_price) * position
                roi = (price / entry_price) - 1
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position': position,
                    'profit': profit,
                    'roi': roi
                })
                
                position = 0
                portfolio['position'].iloc[i:] = 0
                portfolio['cash'].iloc[i:] = cash
        
        # Calculate holdings and total value
        portfolio['holdings'] = portfolio['position'] * portfolio['price']
        portfolio['total'] = portfolio['cash'] + portfolio['holdings']
        
        # Calculate performance metrics
        initial_value = portfolio['total'].iloc[0]
        final_value = portfolio['total'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Calculate drawdown
        portfolio['drawdown'] = (portfolio['total'].cummax() - portfolio['total']) / portfolio['total'].cummax()
        max_drawdown = portfolio['drawdown'].max()
        
        # Calculate Sharpe ratio (assuming daily returns)
        returns = portfolio['total'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate win rate
        if trades:
            winning_trades = [t for t in trades if t['profit'] > 0]
            win_rate = len(winning_trades) / len(trades)
        else:
            win_rate = 0
        
        # Compile performance metrics
        performance = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'final_value': final_value,
            'trades': trades
        }
        
        return performance, portfolio

# Example usage
if __name__ == "__main__":
    # Load sample data
    data_dir = "data"
    symbol = "BTC"
    timeframe = "1h"
    
    file_path = f"{data_dir}/{symbol}_{timeframe}.csv"
    if os.path.exists(file_path):
        price_data = pd.read_csv(file_path)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        
        # Test strategy
        signals = TradingStrategies.ema_crossover_strategy(price_data.copy())
        performance, portfolio = TradingStrategies.get_strategy_performance(price_data, signals)
        
        print(f"Total Return: {performance['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {performance['win_rate']*100:.2f}%")
        print(f"Total Trades: {performance['total_trades']}")