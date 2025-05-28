import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class BacktestEngine:
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 position_size: float = 0.1,
                 max_position: float = 1.0,
                 stop_loss: float = 0.02,
                 take_profit: float = 0.04):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
    def run_backtest(self, prices: pd.Series, signals: pd.Series) -> Dict:
        """Run backtest and return performance metrics"""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades: List[Dict] = []
        
        equity_curve = []
        drawdowns = []
        max_equity = capital
        
        for i in range(len(prices)):
            price = prices.iloc[i]
            signal = signals.iloc[i]
            
            # Handle open position
            if position != 0:
                pnl = position * (price - entry_price)
                current_equity = capital + pnl
                
                # Update max equity and drawdown
                max_equity = max(max_equity, current_equity)
                drawdown = (max_equity - current_equity) / max_equity
                drawdowns.append(drawdown)
                
                # Check stop loss / take profit
                pnl_pct = (price - entry_price) / entry_price
                if (position > 0 and pnl_pct <= -self.stop_loss) or \
                   (position < 0 and pnl_pct >= self.stop_loss) or \
                   (position > 0 and pnl_pct >= self.take_profit) or \
                   (position < 0 and pnl_pct <= -self.take_profit):
                    # Close position
                    capital += pnl
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': i - entry_idx
                    })
                    position = 0
            
            # Handle new signals
            if position == 0:
                if signal == 'BUY':
                    position = self.position_size
                    entry_price = price
                    entry_idx = i
                elif signal == 'SELL':
                    position = -self.position_size
                    entry_price = price
                    entry_idx = i
                    
            equity_curve.append(capital + position * (price - entry_price))
            
        # Calculate metrics
        equity_curve = pd.Series(equity_curve)
        returns = equity_curve.pct_change()
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std(),
            'max_drawdown': max(drawdowns) if drawdowns else 0,
            'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
            'avg_trade': np.mean([t['pnl'] for t in trades]) if trades else 0,
            'num_trades': len(trades)
        }
        
        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trades': trades,
            'drawdowns': drawdowns
        }

