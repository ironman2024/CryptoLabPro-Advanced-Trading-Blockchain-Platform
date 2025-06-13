import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

class PerformanceAnalyzer:
    """Advanced performance analysis for trading strategies"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def calculate_metrics(self, equity_curve, trades):
        """Calculate performance metrics"""
        metrics = {}
        
        # Basic metrics
        initial_equity = equity_curve[0] if isinstance(equity_curve, list) else equity_curve.iloc[0]
        final_equity = equity_curve[-1] if isinstance(equity_curve, list) else equity_curve.iloc[-1]
        
        metrics['total_return'] = (final_equity / initial_equity) - 1
        metrics['total_trades'] = len(trades)
        
        # Calculate drawdowns
        if isinstance(equity_curve, list):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve
            
        rolling_max = equity_series.cummax()
        drawdowns = (rolling_max - equity_series) / rolling_max
        metrics['max_drawdown'] = drawdowns.max()
        
        # Trade metrics
        if trades:
            profits = [t.get('profit', 0) for t in trades]
            rois = [t.get('roi', 0) for t in trades]
            
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p <= 0]
            
            metrics['win_rate'] = len(winning_trades) / len(profits) if profits else 0
            metrics['avg_profit'] = np.mean(profits) if profits else 0
            metrics['avg_roi'] = np.mean(rois) if rois else 0
            metrics['avg_win'] = np.mean(winning_trades) if winning_trades else 0
            metrics['avg_loss'] = np.mean(losing_trades) if losing_trades else 0
            
            # Calculate profit factor
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate Sharpe ratio (assuming daily returns)
            if isinstance(equity_curve, list):
                returns = pd.Series(equity_curve).pct_change().dropna()
            else:
                returns = equity_curve.pct_change().dropna()
                
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate Sortino ratio (downside risk only)
            downside_returns = returns[returns < 0]
            metrics['sortino_ratio'] = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 and downside_returns.std() > 0 else 0
            
            # Calculate Calmar ratio (return / max drawdown)
            metrics['calmar_ratio'] = metrics['total_return'] / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        return metrics
    
    def generate_report(self, equity_curve, trades, strategy_name="Default Strategy"):
        """Generate performance report"""
        # Calculate metrics
        metrics = self.calculate_metrics(equity_curve, trades)
        
        # Add metadata
        report = {
            'strategy_name': strategy_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics
        }
        
        # Save report
        report_path = f"{self.results_dir}/{strategy_name.replace(' ', '_').lower()}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def plot_equity_curve(self, equity_curve, timestamps=None, title="Equity Curve"):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        
        if timestamps is not None:
            plt.plot(timestamps, equity_curve)
            plt.xlabel('Date')
        else:
            plt.plot(equity_curve)
            plt.xlabel('Trade Number')
            
        plt.ylabel('Equity ($)')
        plt.title(title)
        plt.grid(True)
        
        # Save plot
        plt.savefig(f"{self.results_dir}/equity_curve.png")
        plt.close()
    
    def plot_drawdown(self, equity_curve, timestamps=None, title="Drawdown"):
        """Plot drawdown"""
        if isinstance(equity_curve, list):
            equity_series = pd.Series(equity_curve)
        else:
            equity_series = equity_curve
            
        rolling_max = equity_series.cummax()
        drawdowns = (rolling_max - equity_series) / rolling_max
        
        plt.figure(figsize=(12, 6))
        
        if timestamps is not None:
            plt.plot(timestamps, drawdowns)
            plt.xlabel('Date')
        else:
            plt.plot(drawdowns)
            plt.xlabel('Trade Number')
            
        plt.ylabel('Drawdown (%)')
        plt.title(title)
        plt.grid(True)
        
        # Save plot
        plt.savefig(f"{self.results_dir}/drawdown.png")
        plt.close()
    
    def plot_monthly_returns(self, equity_curve, timestamps):
        """Plot monthly returns heatmap"""
        if isinstance(equity_curve, list):
            equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps))
        else:
            equity_series = pd.Series(equity_curve.values, index=pd.DatetimeIndex(timestamps))
        
        # Calculate daily returns
        daily_returns = equity_series.pct_change().dropna()
        
        # Group by month and calculate monthly returns
        monthly_returns = daily_returns.groupby([lambda x: x.year, lambda x: x.month]).apply(lambda x: (1 + x).prod() - 1)
        monthly_returns = monthly_returns.unstack()
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap="RdYlGn", center=0)
        plt.title("Monthly Returns")
        plt.xlabel("Month")
        plt.ylabel("Year")
        
        # Save plot
        plt.savefig(f"{self.results_dir}/monthly_returns.png")
        plt.close()
    
    def plot_trade_distribution(self, trades):
        """Plot trade return distribution"""
        if not trades:
            return
            
        returns = [t.get('roi', 0) * 100 for t in trades]  # Convert to percentage
        
        plt.figure(figsize=(12, 6))
        sns.histplot(returns, kde=True, bins=20)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.title('Trade Return Distribution')
        
        # Save plot
        plt.savefig(f"{self.results_dir}/trade_distribution.png")
        plt.close()
    
    def compare_strategies(self, strategy_reports):
        """Compare multiple strategies"""
        if not strategy_reports:
            return
            
        # Extract metrics for comparison
        comparison = {}
        for name, report in strategy_reports.items():
            comparison[name] = report['metrics']
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison).T
        
        # Save comparison
        comparison_df.to_csv(f"{self.results_dir}/strategy_comparison.csv")
        
        # Plot key metrics
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                plt.figure(figsize=(10, 6))
                comparison_df[metric].plot(kind='bar')
                plt.title(f"Strategy Comparison - {metric}")
                plt.ylabel(metric)
                plt.grid(True, axis='y')
                plt.tight_layout()
                
                # Save plot
                plt.savefig(f"{self.results_dir}/comparison_{metric}.png")
                plt.close()
        
        return comparison_df

# Example usage
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # Sample equity curve
    equity_curve = [10000]
    for i in range(100):
        # Convert to int to avoid type error
        equity_curve.append(int(equity_curve[-1] * (1 + np.random.normal(0.001, 0.01))))
    
    # Sample trades
    trades = []
    for i in range(20):
        roi = np.random.normal(0.02, 0.05)
        profit = roi * 500  # Assuming $500 per trade
        trades.append({
            'entry_time': datetime.now() - timedelta(days=20-i),
            'exit_time': datetime.now() - timedelta(days=19-i),
            'entry_price': 100,
            'exit_price': 100 * (1 + roi),
            'roi': roi,
            'profit': profit
        })
    
    # Generate report
    report = analyzer.generate_report(equity_curve, trades, "Sample Strategy")
    
    # Plot equity curve
    analyzer.plot_equity_curve(equity_curve)
    
    # Plot drawdown
    analyzer.plot_drawdown(equity_curve)
    
    # Plot trade distribution
    analyzer.plot_trade_distribution(trades)