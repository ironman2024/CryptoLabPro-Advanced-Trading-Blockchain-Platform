import pandas as pd
from strategy import generate_signals
from backtesting import BacktestEngine

# Load historical data
candles_target = pd.read_csv('data/LDO_1H.csv')
candles_anchor = pd.read_csv('data/BTC_1H.csv')
candles_anchor_eth = pd.read_csv('data/ETH_1H.csv')

# Merge BTC and ETH data into a single DataFrame
candles_anchor = candles_anchor.rename(columns={'close': 'close_BTC'})
candles_anchor_eth = candles_anchor_eth.rename(columns={'close': 'close_ETH'})
candles_anchor = pd.merge(candles_anchor, candles_anchor_eth[['timestamp', 'close_ETH']], on='timestamp', how='inner')

# Generate signals
signals_df = generate_signals(candles_target, candles_anchor)

# Run backtest
engine = BacktestEngine()
results = engine.run_backtest(prices=candles_target['close'], signals=signals_df['signal'])

# Display results
print("Performance Metrics:")
for metric, value in results['metrics'].items():
    print(f"{metric}: {value}")

print("\nEquity Curve:")
print(results['equity_curve'].tail())

print("\nTrades:")
for trade in results['trades']:
    print(trade)
