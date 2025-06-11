import pandas as pd
import time
import threading
from datetime import datetime
import os

class DataFetcher:
    def __init__(self, update_interval_minutes=60):
        self.update_interval_minutes = update_interval_minutes
        self.last_update = None
        self.update_thread = None
        self.running = False
    
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