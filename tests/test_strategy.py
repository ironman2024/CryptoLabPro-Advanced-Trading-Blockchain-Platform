import unittest
import pandas as pd
import numpy as np
from unified_strategy import UnifiedStrategy

class TestUnifiedStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        self.strategy = UnifiedStrategy()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        self.target_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.anchor_data = pd.DataFrame({
            'timestamp': dates,
            'close_BTC': np.random.randn(100).cumsum() + 30000,
            'volume_BTC': np.random.randint(100000, 1000000, 100),
            'close_ETH': np.random.randn(100).cumsum() + 2000,
            'volume_ETH': np.random.randint(50000, 500000, 100),
            'close_SOL': np.random.randn(100).cumsum() + 100,
            'volume_SOL': np.random.randint(10000, 100000, 100)
        })

    def test_indicator_calculation(self):
        """Test technical indicator calculations."""
        df = self.strategy.calculate_indicators(self.target_data)
        
        # Check that key indicators are present
        required_indicators = [
            'ema_8', 'ema_13', 'ema_21', 'rsi_14', 'macd', 
            'momentum_3', 'atr', 'bb_middle'
        ]
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns)
        
        # Check that indicators have valid values
        self.assertTrue(df['ema_8'].notna().any())
        self.assertTrue(df['rsi_14'].between(0, 100).all())
        self.assertTrue(df['atr'].notna().any())

    def test_anchor_signals(self):
        """Test anchor signal calculations."""
        df = pd.merge(self.target_data, self.anchor_data, on='timestamp')
        df = self.strategy.calculate_anchor_signals(df)
        
        # Check that anchor indicators are present
        required_indicators = [
            'btc_momentum_5', 'eth_momentum_5', 'market_regime', 'market_strength'
        ]
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns)
        
        # Check market regime values
        self.assertTrue(df['market_regime'].between(0, 2).all())
        self.assertTrue(df['market_strength'].notna().any())

    def test_signal_generation(self):
        """Test trading signal generation."""
        signals = self.strategy.generate_signals(self.target_data, self.anchor_data)
        
        # Check signal DataFrame structure
        self.assertIn('timestamp', signals.columns)
        self.assertIn('signal', signals.columns)
        
        # Check signal values
        valid_signals = ['BUY', 'SELL', 'HOLD']
        self.assertTrue(signals['signal'].isin(valid_signals).all())
        
        # Check that signals align with input data
        self.assertEqual(len(signals), len(self.target_data))

    def test_backtest(self):
        """Test backtesting functionality."""
        results = self.strategy.backtest(self.target_data, self.anchor_data)
        
        # Check that key metrics are present
        required_metrics = [
            'total_return', 'sharpe_ratio', 'max_drawdown', 
            'win_rate', 'num_trades', 'avg_return'
        ]
        for metric in required_metrics:
            self.assertIn(metric, results)
        
        # Check metric values
        self.assertIsInstance(results['total_return'], float)
        self.assertIsInstance(results['sharpe_ratio'], float)
        self.assertTrue(0 <= results['max_drawdown'] <= 1)
        self.assertTrue(0 <= results['win_rate'] <= 1)
        self.assertGreaterEqual(results['num_trades'], 0)

    def test_ml_integration(self):
        """Test machine learning integration."""
        # Train model
        train_score, test_score = self.strategy.train_model((self.target_data, self.anchor_data))
        
        # Check that model was trained
        self.assertIsNotNone(self.strategy.model)
        
        # Check scores
        self.assertTrue(0 <= train_score <= 1)
        self.assertTrue(0 <= test_score <= 1)
        
        # Test prediction
        df = pd.merge(self.target_data, self.anchor_data, on='timestamp')
        df = self.strategy.calculate_indicators(df)
        df = self.strategy.calculate_anchor_signals(df)
        
        signals, probs = self.strategy.predict_signals(df)
        
        # Check predictions
        self.assertEqual(len(signals), len(df))
        self.assertEqual(len(probs), len(df))
        self.assertTrue(all(0 <= p <= 1 for p in probs))

class TestAdvancedStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test case."""
        from advanced_strategy import AdvancedStrategy
        self.strategy = AdvancedStrategy()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        self.target_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        self.anchor_data = pd.DataFrame({
            'timestamp': dates,
            'close_BTC': np.random.randn(100).cumsum() + 30000,
            'volume_BTC': np.random.randint(100000, 1000000, 100),
            'close_ETH': np.random.randn(100).cumsum() + 2000,
            'volume_ETH': np.random.randint(50000, 500000, 100),
            'close_SOL': np.random.randn(100).cumsum() + 100,
            'volume_SOL': np.random.randint(10000, 100000, 100)
        })

    def test_talib_fallback(self):
        """Test TALibFallback indicator calculations."""
        df = self.strategy.calculate_advanced_indicators(self.target_data)
        
        # Check that key indicators are present
        required_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'mom',
            'atr', 'bbands_upper', 'bbands_middle', 'bbands_lower'
        ]
        for indicator in required_indicators:
            self.assertIn(indicator, df.columns)
        
        # Check that indicators have valid values
        self.assertTrue(df['rsi'].between(0, 100).all())
        self.assertTrue(df['macd'].notna().any())
        self.assertTrue(df['mom'].notna().any())
        self.assertTrue(df['atr'].notna().any())
        self.assertTrue((df['bbands_upper'] > df['bbands_middle']).all())
        self.assertTrue((df['bbands_middle'] > df['bbands_lower']).all())

    def test_advanced_signal_generation(self):
        """Test advanced trading signal generation."""
        signals = self.strategy.generate_signals(self.target_data, self.anchor_data)
        
        # Check signal DataFrame structure
        self.assertIn('timestamp', signals.columns)
        self.assertIn('signal', signals.columns)
        
        # Check signal values
        valid_signals = ['BUY', 'SELL', 'HOLD']
        self.assertTrue(signals['signal'].isin(valid_signals).all())
        
        # Check that signals align with input data
        self.assertEqual(len(signals), len(self.target_data))

    def test_advanced_ml_model(self):
        """Test advanced ML model training and prediction."""
        # Train model
        train_score, test_score = self.strategy.train_model((self.target_data, self.anchor_data))
        
        # Check that model was trained
        self.assertIsNotNone(self.strategy.model)
        
        # Check scores
        self.assertTrue(0 <= train_score <= 1)
        self.assertTrue(0 <= test_score <= 1)
        
        # Test signal generation with trained model
        signals = self.strategy.generate_signals(self.target_data, self.anchor_data)
        
        # Check predictions
        self.assertEqual(len(signals), len(self.target_data))
        self.assertTrue(signals['signal'].isin(['BUY', 'SELL', 'HOLD']).all())

if __name__ == '__main__':
    unittest.main()