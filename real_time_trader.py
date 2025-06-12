import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import os
from strategy import Strategy, DEFAULT_CONFIG
import json
import matplotlib.pyplot as plt
from fetch_data import DataFetcher

class RealTimeTrader:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG
        self.strategy = Strategy(self.config)
        self.data_fetcher = DataFetcher()
        self.running = False
        self.update_thread = None
        self.positions = {}
        self.trade_history = []
        self.equity_curve = []
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.last_update = None
        self.transaction_cost = 0.001  # 0.1% per trade
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Load trade history if exists
        self._load_trade_history()
    
    def _load_trade_history(self):
        try:
            if os.path.exists("results/trade_history.json"):
                with open("results/trade_history.json", "r") as f:
                    self.trade_history = json.load(f)
                    
            if os.path.exists("results/equity_curve.json"):
                with open("results/equity_curve.json", "r") as f:
                    self.equity_curve = json.load(f)
                    if self.equity_curve:
                        self.current_capital = self.equity_curve[-1]["equity"]
        except Exception as e:
            print(f"Error loading trade history: {e}")
    
    def _save_trade_history(self):
        try:
            with open("results/trade_history.json", "w") as f:
                json.dump(self.trade_history, f)
                
            with open("results/equity_curve.json", "w") as f:
                json.dump(self.equity_curve, f)
        except Exception as e:
            print(f"Error saving trade history: {e}")
    
    def start(self, update_interval_minutes=5):
        """Start real-time trading"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._trading_loop, args=(update_interval_minutes,))
            self.update_thread.daemon = True
            self.update_thread.start()
            print(f"Real-time trading started with {update_interval_minutes} minute updates")
    
    def stop(self):
        """Stop real-time trading"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
        print("Real-time trading stopped")
    
    def _trading_loop(self, update_interval_minutes):
        """Main trading loop"""
        while self.running:
            try:
                self._execute_trading_cycle()
                self.last_update = datetime.now()
                
                # Save results
                self._save_trade_history()
                
                # Generate performance report
                self.generate_performance_report()
                
            except Exception as e:
                print(f"Error in trading cycle: {e}")
            
            # Sleep until next update
            time.sleep(update_interval_minutes * 60)
    
    def _execute_trading_cycle(self):
        """Execute one trading cycle"""
        # Get latest data
        target_data, anchor_data = self.strategy.get_live_data()
        if target_data is None or anchor_data is None:
            print("No data available for trading")
            return
        
        # Generate signals
        signals = self.strategy.generate_signals(target_data, anchor_data)
        latest_signal = signals.iloc[-1]['signal']
        latest_price = float(target_data.iloc[-1]['close'])
        timestamp = target_data.iloc[-1]['timestamp']
        
        # Execute trades based on signals
        if latest_signal == 'BUY' and self.config['target']['symbol'] not in self.positions:
            # Calculate position size (could be enhanced with risk management)
            position_size = 0.95  # Use 95% of capital
            
            # Execute buy
            cost = latest_price * (1 + self.transaction_cost)
            quantity = (self.current_capital * position_size) / cost
            
            self.positions[self.config['target']['symbol']] = {
                'entry_price': latest_price,
                'quantity': quantity,
                'entry_time': timestamp,
                'cost': cost * quantity
            }
            
            # Log trade
            trade = {
                'type': 'BUY',
                'symbol': self.config['target']['symbol'],
                'price': latest_price,
                'quantity': quantity,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'cost': cost * quantity
            }
            self.trade_history.append(trade)
            print(f"BUY: {quantity:.6f} {self.config['target']['symbol']} at ${latest_price:.4f}")
            
        elif latest_signal == 'SELL' and self.config['target']['symbol'] in self.positions:
            # Execute sell
            position = self.positions[self.config['target']['symbol']]
            exit_price = latest_price * (1 - self.transaction_cost)
            proceeds = exit_price * position['quantity']
            profit = proceeds - position['cost']
            roi = profit / position['cost']
            
            # Update capital
            self.current_capital += profit
            
            # Log trade
            trade = {
                'type': 'SELL',
                'symbol': self.config['target']['symbol'],
                'price': latest_price,
                'quantity': position['quantity'],
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'proceeds': proceeds,
                'profit': profit,
                'roi': roi
            }
            self.trade_history.append(trade)
            print(f"SELL: {position['quantity']:.6f} {self.config['target']['symbol']} at ${latest_price:.4f}, Profit: ${profit:.2f} ({roi*100:.2f}%)")
            
            # Remove position
            del self.positions[self.config['target']['symbol']]
        
        # Update equity curve
        if self.positions:
            # Calculate unrealized P&L
            position = self.positions[self.config['target']['symbol']]
            unrealized_value = latest_price * position['quantity']
            equity = self.current_capital - position['cost'] + unrealized_value
        else:
            equity = self.current_capital
        
        self.equity_curve.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'equity': equity
        })
    
    def generate_performance_report(self):
        """Generate performance report"""
        if not self.trade_history:
            print("No trades to report")
            return
        
        # Calculate metrics
        trades = [t for t in self.trade_history if t['type'] == 'SELL']
        if not trades:
            return
            
        profits = [t['profit'] for t in trades]
        rois = [t['roi'] for t in trades]
        
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        # Calculate metrics
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = np.mean(profits) if profits else 0
        avg_roi = np.mean(rois) if rois else 0
        
        # Calculate drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        max_equity = 0
        drawdowns = []
        
        for equity in equity_values:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        # Save report
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_roi': avg_roi,
            'max_drawdown': max_drawdown,
            'current_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1
        }
        
        with open("results/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate equity curve chart
        if len(self.equity_curve) > 1:
            timestamps = [e['timestamp'] for e in self.equity_curve]
            equity_values = [e['equity'] for e in self.equity_curve]
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, equity_values)
            plt.title('Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('results/equity_curve.png')
            plt.close()
        
        return report

# Example usage
if __name__ == "__main__":
    trader = RealTimeTrader()
    trader.start(update_interval_minutes=5)
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        trader.stop()
        print("Trading stopped by user")