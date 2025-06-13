import subprocess
import sys

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    print("\n🚀 Setup complete! You can now run:")
    print("- python fetch_data.py     # To download OHLCV data")
    print("- streamlit run streamlit_app.py  # To launch the GUI")