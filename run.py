#!/usr/bin/env python
"""
Main entry point for the Pairwise Alpha Trading System.
This script provides a command-line interface to run different components of the system.
"""

import argparse
import os
import sys
from strategy import Strategy, DEFAULT_CONFIG
from ensemble_strategy import EnsembleStrategy
from real_time_trader import RealTimeTrader
import subprocess
import time

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("Starting dashboard...")
    subprocess.Popen(["streamlit", "run", "dashboard.py"])

def run_trader(interval=5):
    """Run the real-time trader"""
    print(f"Starting real-time trader with {interval} minute updates...")
    trader = RealTimeTrader()
    trader.start(update_interval_minutes=interval)
    
    try:
        while True:
            time.sleep(60)
            print(f"Trader running... Last update: {trader.last_update}")
    except KeyboardInterrupt:
        trader.stop()
        print("Trading stopped by user")

def run_backtest():
    """Run backtest on historical data"""
    print("Running backtest...")
    
    # Standard strategy backtest
    print("\n=== Standard Strategy Backtest ===")
    strategy = Strategy()
    os.system("python strategy.py")
    
    # Ensemble strategy backtest
    print("\n=== Ensemble Strategy Backtest ===")
    ensemble = EnsembleStrategy()
    
    # Get data and train model
    from fetch_data import DataFetcher
    data_fetcher = DataFetcher()
    
    target_data = data_fetcher.get_recent_data(DEFAULT_CONFIG["target"]["symbol"], DEFAULT_CONFIG["target"]["timeframe"])
    
    # Get anchor data
    anchor_data = None
    if target_data is not None:
        anchor_data = {}
        for anchor in DEFAULT_CONFIG["anchors"]:
            symbol = anchor["symbol"]
            timeframe = anchor["timeframe"]
            data = data_fetcher.get_recent_data(symbol, timeframe)
            if data is not None:
                anchor_data[symbol] = data
    
    if target_data is not None and anchor_data:
        # Train and test ensemble strategy
        ensemble.train_model((target_data, anchor_data))
        signals = ensemble.generate_signals(target_data, anchor_data)
        print(f"Generated {len(signals)} signals")
        print(signals['signal'].value_counts())
    else:
        print("Insufficient data for ensemble strategy backtest")

def train_model():
    """Train the ensemble model"""
    print("Training ensemble model...")
    ensemble = EnsembleStrategy()
    
    # Get data
    from fetch_data import DataFetcher
    data_fetcher = DataFetcher()
    
    target_data = data_fetcher.get_recent_data(DEFAULT_CONFIG["target"]["symbol"], DEFAULT_CONFIG["target"]["timeframe"])
    
    # Get anchor data
    anchor_data = None
    if target_data is not None:
        anchor_data = {}
        for anchor in DEFAULT_CONFIG["anchors"]:
            symbol = anchor["symbol"]
            timeframe = anchor["timeframe"]
            data = data_fetcher.get_recent_data(symbol, timeframe)
            if data is not None:
                anchor_data[symbol] = data
    
    if target_data is not None and anchor_data:
        # Train model
        ensemble.train_model((target_data, anchor_data))
        print("Model training complete")
    else:
        print("Insufficient data for model training")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pairwise Alpha Trading System")
    parser.add_argument("command", choices=["dashboard", "trader", "backtest", "train"], 
                        help="Command to run")
    parser.add_argument("--interval", type=int, default=5,
                        help="Update interval in minutes for trader (default: 5)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if args.command == "dashboard":
        run_dashboard()
    elif args.command == "trader":
        run_trader(args.interval)
    elif args.command == "backtest":
        run_backtest()
    elif args.command == "train":
        train_model()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()