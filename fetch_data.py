import pandas as pd
import time
import threading
from datetime import datetime, timedelta
import os
import numpy as np

class DataFetcher:
    def __init__(self, update_interval_minutes=60):
        self.update_interval_minutes = update_interval_minutes
        self.last_update = None
        self.update_thread = None
        self.running = False
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_recent_data(self, symbol, interval):
        """Get recent data for a symbol and interval"""
        try:
            file_path = f"{self.data_dir}/{symbol}_{interval}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=True)
                return df
            else:
                # Generate sample data if file doesn't exist
                return self._generate_sample_data(symbol, interval)
        except Exception as e:
            print(f"Error getting data for {symbol}_{interval}: {e}")
            return None
    
    def _generate_sample_data(self, symbol, interval):
        """Generate sample data for testing"""
        # Set seed based on symbol for consistent results
        np.random.seed(sum(ord(c) for c in symbol))
        
        # Generate timestamps
        end_date = datetime.now()
        if interval == "1h":
            start_date = end_date - timedelta(days=30)
            freq = "H"
        elif interval == "4h":
            start_date = end_date - timedelta(days=120)
            freq = "4H"
        else:  # Default to daily
            start_date = end_date - timedelta(days=365)
            freq = "D"
            
        timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate price data
        base_price = 100 if symbol == "BTC" else 50 if symbol == "ETH" else 20
        price = base_price
        prices = []
        
        for _ in range(len(timestamps)):
            # Random walk with drift
            price = price * (1 + np.random.normal(0.0002, 0.02))
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': [np.random.uniform(1000, 10000) * base_price for _ in prices]
        })
        
        # Save to file
        file_path = f"{self.data_dir}/{symbol}_{interval}.csv"
        df.to_csv(file_path, index=False)
        
        return df
    
    def start_background_updates(self):
        """Start background data updates"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _update_loop(self):
        """Background loop to update data periodically"""
        while self.running:
            try:
                self.update_data()
                self.last_update = datetime.now()
            except Exception as e:
                print(f"Error updating data: {e}")
            
            # Sleep until next update
            time.sleep(self.update_interval_minutes * 60)
    
    def update_data(self):
        """Update data from sources"""
        # This would normally fetch data from APIs
        # For now, we'll just use sample data if it exists
        pass

def save_data(data, filename):
    """Save data to CSV file"""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save data to CSV
    data.to_csv(f"data/{filename}", index=False)
    return f"data/{filename}"