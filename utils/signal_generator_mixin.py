import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class SignalGeneratorMixin:
    """Mixin class providing signal generation capabilities"""
    
    def generate_traditional_signals(self, candles_target: pd.DataFrame, candles_anchor: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using traditional technical analysis."""
        try:
            # Calculate indicators
            candles_target = self.calculate_indicators(candles_target)
            candles_anchor = candles_anchor.copy()
            
            # Prepare anchor data
            anchor_cols = [col for col in candles_anchor.columns if col.startswith('close_')]
            for col in anchor_cols:
                candles_anchor[col] = candles_anchor[col].astype(float)
                
            # Merge data
            df = pd.merge(candles_target, candles_anchor, on='timestamp', how='inner')
            df = df.dropna().reset_index(drop=True)
            df = self.calculate_anchor_signals(df)
            df['signal'] = 'HOLD'
            
            # --- Entry conditions ---
            ema_trend = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21'])
            ema_slope_positive = (df['ema_13_slope'] > 0)
            
            # Momentum indicators
            rsi_good = (df['rsi_14'] > 40) & (df['rsi_14'] < 70)
            momentum_good = (df['momentum_3'] > 0) | (df['momentum_5'] > 0)
            
            # Volume indicators
            volume_good = (df['volume_ratio'] > 0.8)
            
            # Primary entry - target in uptrend with good momentum
            primary_entry = (ema_trend & momentum_good & rsi_good)
            
            # Secondary entry - simpler conditions
            secondary_entry = (ema_slope_positive & momentum_good)
            
            # Combine entry conditions
            can_enter_mask = (primary_entry | secondary_entry)
            
            # --- Position management ---
            position_active = False
            entry_price = 0
            bars_in_trade = 0
            cooldown_remaining = 0
            cooldown_period = 3
            stop_loss = 0
            take_profit = 0
            trailing_stop = 0
            best_price = 0
            
            # Track recent performance
            recent_trades = []
            max_recent_trades = 5
            
            for i in range(50, len(df)):
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue
                    
                current_price = df.at[df.index[i], 'close']
                current_atr = df.at[df.index[i], 'atr']
                
                if pd.isna(current_atr) or current_atr <= 0:
                    continue
                
                # Entry logic
                if not position_active and can_enter_mask.iloc[i]:
                    # Risk management
                    atr_multiplier = 1.2
                    tp_multiplier = 3.0
                    
                    stop_loss = current_price - (atr_multiplier * current_atr)
                    take_profit = current_price + (tp_multiplier * current_atr)
                    trailing_stop = stop_loss
                    
                    df.at[df.index[i], 'signal'] = 'BUY'
                    position_active = True
                    entry_price = current_price
                    best_price = current_price
                    bars_in_trade = 0
                    
                elif position_active:
                    bars_in_trade += 1
                    
                    # Dynamic trailing stop
                    if current_price > best_price:
                        best_price = current_price
                        profit_pct = (best_price - entry_price) / entry_price
                        
                        # More aggressive trailing stop as profit increases
                        if profit_pct > 0.05:
                            trailing_stop = max(trailing_stop, best_price - (0.5 * current_atr))
                        elif profit_pct > 0.03:
                            trailing_stop = max(trailing_stop, best_price - (0.7 * current_atr))
                        elif profit_pct > 0.015:
                            trailing_stop = max(trailing_stop, best_price - (1.0 * current_atr))
                    
                    # Exit conditions
                    exit_triggered = False
                    
                    # Basic stop conditions
                    if current_price <= stop_loss:
                        exit_triggered = True
                    elif current_price >= take_profit:
                        exit_triggered = True
                    elif current_price <= trailing_stop:
                        exit_triggered = True
                    elif bars_in_trade >= 20:  # Maximum hold time
                        exit_triggered = True
                        
                    # Dynamic exit based on trend change
                    elif bars_in_trade >= 2:
                        trend_weakening = False
                        
                        # Severe trend reversal
                        if (df.at[df.index[i], 'ema_8'] < df.at[df.index[i], 'ema_13']) and (df.at[df.index[i], 'macd_hist'] < 0):
                            trend_weakening = True
                            
                        # RSI overbought with negative momentum
                        elif (df.at[df.index[i], 'rsi_14'] > 70) and (df.at[df.index[i], 'momentum_3'] < 0):
                            trend_weakening = True
                            
                        # Market regime shift to bearish with profit
                        elif (df.at[df.index[i], 'market_regime'] == 0) and (current_price > entry_price * 1.01):
                            trend_weakening = True
                            
                        # Anchor assets turning bearish with profit
                        elif (df.at[df.index[i], 'btc_momentum_3'] < -0.005 and 
                              df.at[df.index[i], 'eth_momentum_3'] < -0.005 and
                              current_price > entry_price * 1.005):
                            trend_weakening = True
                            
                        if trend_weakening:
                            exit_triggered = True
                    
                    if exit_triggered:
                        df.at[df.index[i], 'signal'] = 'SELL'
                        position_active = False
                        cooldown_remaining = cooldown_period
                        
                        # Track trade performance
                        trade_return = (current_price - entry_price) / entry_price
                        if len(recent_trades) >= max_recent_trades:
                            recent_trades.pop(0)
                        recent_trades.append(trade_return)
                        
                        entry_price = 0
                        bars_in_trade = 0
                        best_price = 0
            
            return df[['timestamp', 'signal']].set_index('timestamp').reindex(
                candles_target['timestamp'],
                fill_value='HOLD'
            ).reset_index()
            
        except Exception as e:
            error_info = f"Signal generation error: {str(e)}"
            print(error_info)
            return pd.DataFrame({'timestamp': candles_target['timestamp'], 'signal': 'HOLD'})

    def backtest_signals(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
        """Run backtest on generated signals."""
        try:
            df = df.merge(signals, on='timestamp', how='left')
            df['signal'] = df['signal'].fillna('HOLD')
            
            # Initialize tracking variables
            position = 0
            entry_price = 0
            equity_curve = [10000]  # Start with $10,000
            trade_returns = []
            max_equity = 10000
            drawdowns = []
            
            # Transaction costs
            transaction_cost = 0.001  # 0.1% per trade
            
            for i in range(1, len(df)):
                price = df.iloc[i]['close']
                signal = df.iloc[i]['signal']
                
                if position == 0 and signal == 'BUY':
                    position = 1
                    entry_price = price * (1 + transaction_cost)
                elif position > 0 and signal == 'SELL':
                    exit_price = price * (1 - transaction_cost)
                    returns = (exit_price - entry_price) / entry_price
                    trade_returns.append(returns)
                    equity_curve.append(equity_curve[-1] * (1 + returns))
                    position = 0
                else:
                    equity_curve.append(equity_curve[-1])
                
                max_equity = max(max_equity, equity_curve[-1])
                drawdowns.append((max_equity - equity_curve[-1]) / max_equity)
            
            # Calculate metrics
            total_return = (equity_curve[-1] / equity_curve[0]) - 1
            max_drawdown = max(drawdowns) if drawdowns else 0
            
            if len(trade_returns) > 1:
                sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252*24)
                win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            else:
                sharpe = 0
                win_rate = 0
                
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': len(trade_returns),
                'avg_return': np.mean(trade_returns) if trade_returns else 0,
                'equity_curve': equity_curve
            }
            
        except Exception as e:
            print(f"Backtest error: {e}")
            return {}