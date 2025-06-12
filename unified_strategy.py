from utils.strategy_base import StrategyBase
from utils.indicators_mixin import IndicatorsMixin
from utils.ml_mixin import MLMixin
from utils.signal_generator_mixin import SignalGeneratorMixin
import pandas as pd
from typing import Dict, Any, Optional, Tuple

class UnifiedStrategy(StrategyBase, IndicatorsMixin, MLMixin, SignalGeneratorMixin):
    """
    Unified trading strategy that combines traditional technical analysis with machine learning.
    Features:
    - Technical indicators and market regime analysis
    - Machine learning signal enhancement
    - Advanced risk management and position sizing
    - Comprehensive backtesting capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        StrategyBase.__init__(self, config)
        MLMixin.__init__(self)
    
    def generate_signals(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using both traditional and ML approaches."""
        try:
            # Calculate indicators
            candles_target = self.calculate_indicators(candles_target)
            
            # Prepare anchor data
            candles_anchor = candles_anchor.copy()
            anchor_cols = [col for col in candles_anchor.columns if col.startswith('close_')]
            for col in anchor_cols:
                candles_anchor[col] = candles_anchor[col].astype(float)
            
            # Merge data
            df = pd.merge(candles_target, candles_anchor, on='timestamp', how='inner')
            df = df.dropna().reset_index(drop=True)
            df = self.calculate_anchor_signals(df)
            
            # Generate traditional signals
            traditional_signals = self.generate_traditional_signals(candles_target, candles_anchor)
            df = df.merge(traditional_signals[['timestamp', 'signal']], on='timestamp', how='left', suffixes=('', '_trad'))
            
            # If ML model is available, enhance signals
            if self.model is not None:
                ml_signals, buy_probs = self.predict_signals(df)
                
                # Combine signals (ML overrides traditional for high-confidence predictions)
                df['signal'] = df['signal_trad']
                df.loc[ml_signals == 'BUY', 'signal'] = 'BUY'
                df.loc[ml_signals == 'SELL', 'signal'] = 'SELL'
            else:
                df['signal'] = df['signal_trad']
            
            print("Signal value counts:", df['signal'].value_counts())
            return df[['timestamp', 'signal']].set_index('timestamp').reindex(
                candles_target['timestamp'],
                fill_value='HOLD'
            ).reset_index()
            
        except Exception as e:
            error_info = f"Signal generation error: {str(e)}"
            print(error_info)
            return pd.DataFrame({'timestamp': candles_target['timestamp'], 'signal': 'HOLD'})

    def backtest(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest with detailed performance metrics."""
        signals = self.generate_signals(candles_target, candles_anchor)
        return self.backtest_signals(candles_target, signals)

def main():
    """Example usage of UnifiedStrategy"""
    from fetch_data import DataFetcher
    
    # Initialize strategy
    strategy = UnifiedStrategy()
    
    # Get historical data
    data_fetcher = DataFetcher()
    target_data = data_fetcher.get_recent_data("LDO", "1h")
    
    # Get anchor data
    anchor_data = pd.DataFrame()
    for symbol in ["BTC", "ETH", "SOL"]:
        data = data_fetcher.get_recent_data(symbol, "1h")
        if data is not None:
            anchor_data[f'close_{symbol}'] = data['close']
            anchor_data[f'volume_{symbol}'] = data['volume']
            anchor_data['timestamp'] = data['timestamp']
    
    if target_data is not None and not anchor_data.empty:
        # Train model
        print("\nTraining model...")
        strategy.train_model((target_data, anchor_data))
        
        # Generate signals
        print("\nGenerating signals...")
        signals = strategy.generate_signals(target_data, anchor_data)
        print("\nSignal preview:")
        print(signals.tail())
        
        # Run backtest
        print("\nRunning backtest...")
        results = strategy.backtest(target_data, anchor_data)
        
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Average Trade Return: {results['avg_return']*100:.2f}%")
    else:
        print("Error: Could not load data")

if __name__ == "__main__":
    main()