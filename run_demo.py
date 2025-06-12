#!/usr/bin/env python
"""
Run script for the blockchain demo application.
"""

import os
import subprocess
import sys

def main():
    """Main entry point for running the blockchain demo."""
    print("Starting Blockchain Demo...")
    
    # Run the Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "blockchain_demo.py"])

if __name__ == "__main__":
    main()