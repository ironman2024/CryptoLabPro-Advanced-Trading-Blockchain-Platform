import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import time
import json

class DataProcessor:
    """Advanced data processing for trading strategies"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self, symbol, timeframe, lookback_days=30):
        """Load data from file or download if needed"""
        file_path = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        
        # Check if file exists and is recent
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check if data is recent enough
            if len(df) > 0 and datetime.now() - df['timestamp'].max() < timedelta(hours=1):
                return df
        
        # If not, download data
        return self.download_data(symbol, timeframe, lookback_days)
    
    def download_data(self, symbol, timeframe, lookback_days=30):
        """Download data from API"""
        # This is a placeholder - in a real implementation, you would use a proper API
        # For now, we'll just return sample data
        print(f"Downloading data for {symbol}_{timeframe}...")
        
        # Create sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq=timeframe)
        
        # Generate prices
        np.random.seed(42)  # For reproducibility
        price = 100
        prices = []
        for _ in range(len(dates)):
            price = price * (1 + np.random.normal(0, 0.01))
            prices.append(price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        })
        
        # Save to file
        file_path = f"{self.data_dir}/{symbol}_{timeframe}.csv"
        df.to_csv(file_path, index=False)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for analysis"""
        df = df.copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def merge_data(self, target_df, anchor_dfs):
        """Merge target and anchor dataframes"""
        result = target_df.copy()
        
        for symbol, df in anchor_dfs.items():
            # Rename columns to avoid conflicts
            df_renamed = df.copy()
            for col in df_renamed.columns:
                if col != 'timestamp':
                    df_renamed.rename(columns={col: f"{col}_{symbol}"}, inplace=True)
            
            # Merge on timestamp
            result = pd.merge(result, df_renamed, on='timestamp', how='inner')
        
        return result
    
    def split_data(self, df, train_ratio=0.8):
        """Split data into training and testing sets"""
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        return train_df, test_df
    
    def create_features(self, df):
        """Create features for machine learning"""
        df = df.copy()
        
        # Price features
        df['return_1d'] = df['close'].pct_change(1)
        df['return_3d'] = df['close'].pct_change(3)
        df['return_5d'] = df['close'].pct_change(5)
        
        # Volatility features
        df['volatility_3d'] = df['return_1d'].rolling(3).std()
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change(1)
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # Price relative to moving averages
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        
        df['price_ma5_ratio'] = df['close'] / df['ma5']
        df['price_ma10_ratio'] = df['close'] / df['ma10']
        df['price_ma20_ratio'] = df['close'] / df['ma20']
        df['price_ma50_ratio'] = df['close'] / df['ma50']
        
        # Moving average crossovers
        df['ma5_cross_ma10'] = (df['ma5'] > df['ma10']).astype(int)
        df['ma10_cross_ma20'] = (df['ma10'] > df['ma20']).astype(int)
        df['ma20_cross_ma50'] = (df['ma20'] > df['ma50']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def create_labels(self, df, forward_periods=5, threshold=0.01):
        """Create labels for supervised learning"""
        df = df.copy()
        
        # Calculate future returns
        df['future_return'] = df['close'].shift(-forward_periods) / df['close'] - 1
        
        # Create labels
        df['label'] = 0  # Hold
        df.loc[df['future_return'] > threshold, 'label'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'label'] = -1  # Sell
        
        # Drop NaN values
        df = df.dropna()
        
        return df

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load data
    btc_data = processor.load_data("BTC", "1h", lookback_days=30)
    eth_data = processor.load_data("ETH", "1h", lookback_days=30)
    
    # Preprocess data
    btc_data = processor.preprocess_data(btc_data)
    eth_data = processor.preprocess_data(eth_data)
    
    # Merge data
    merged_data = processor.merge_data(btc_data, {"ETH": eth_data})
    
    # Create features and labels
    featured_data = processor.create_features(merged_data)
    labeled_data = processor.create_labels(featured_data)
    
    # Split data
    train_data, test_data = processor.split_data(labeled_data)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")
    print(f"Label distribution: {labeled_data['label'].value_counts()}")