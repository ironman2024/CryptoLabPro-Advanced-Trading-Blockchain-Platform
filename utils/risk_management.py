import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, max_drawdown=0.1, volatility_target=0.2, min_reward_risk=1.5):
        self.max_drawdown = max_drawdown
        self.volatility_target = volatility_target
        self.min_reward_risk = min_reward_risk
        
    def calculate_position_sizes(self, df, predictions):
        """Calculate position sizes based on volatility targeting"""
        # Calculate volatility
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Kelly fraction
        win_prob = np.mean(predictions > 0.5)
        win_ratio = np.mean(returns[returns > 0]) / abs(np.mean(returns[returns < 0]))
        kelly = (win_prob * win_ratio - (1 - win_prob)) / win_ratio
        
        # Volatility scaling
        vol_scalar = self.volatility_target / vol
        position_sizes = kelly * vol_scalar
        
        # Apply maximum position constraint
        position_sizes = np.minimum(position_sizes, 1.0)
        position_sizes = np.maximum(position_sizes, 0.0)
        
        return position_sizes
        
    def get_risk_score(self, df, idx):
        """Calculate risk score for current market conditions"""
        return {
            'volatility': df['volatility'].iloc[idx],
            'drawdown': df['drawdown'].iloc[idx],
            'reward_risk': df['reward_risk_ratio'].iloc[idx]
        }
