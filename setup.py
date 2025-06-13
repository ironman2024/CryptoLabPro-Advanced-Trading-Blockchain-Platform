import subprocess
import sys
import platform
import os

def check_talib():
    """Check if TA-Lib is installed and install it if needed"""
    try:
        import talib
        print("✅ TA-Lib is already installed!")
        return True
    except ImportError:
        system = platform.system().lower()
        
        print("ℹ️ TA-Lib not found. Attempting to install...")
        
        if system == "linux":
            try:
                # Try apt-get first (Debian/Ubuntu)
                subprocess.check_call(["sudo", "apt-get", "update"])
                subprocess.check_call(["sudo", "apt-get", "install", "-y", "ta-lib"])
                return True
            except:
                try:
                    # Try yum (CentOS/RHEL)
                    subprocess.check_call(["sudo", "yum", "install", "-y", "ta-lib"])
                    return True
                except:
                    print("⚠️ Could not install TA-Lib via package manager.")
                    print("ℹ️ Falling back to source installation...")
                    return install_talib_from_source()
        
        elif system == "darwin":  # macOS
            try:
                subprocess.check_call(["brew", "install", "ta-lib"])
                return True
            except:
                print("⚠️ Could not install TA-Lib via Homebrew.")
                print("ℹ️ Falling back to source installation...")
                return install_talib_from_source()
        
        elif system == "windows":
            print("ℹ️ On Windows, please download and install TA-Lib from:")
            print("http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip")
            print("Then run: pip install TA-Lib")
            return False

def install_talib_from_source():
    """Install TA-Lib from source"""
    try:
        # Download and extract source
        subprocess.check_call([
            "wget", "http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz"
        ])
        subprocess.check_call(["tar", "-xzf", "ta-lib-0.4.0-src.tar.gz"])
        
        # Build and install
        os.chdir("ta-lib")
        subprocess.check_call(["./configure", "--prefix=/usr"])
        subprocess.check_call(["make"])
        subprocess.check_call(["sudo", "make", "install"])
        os.chdir("..")
        
        # Clean up
        subprocess.check_call(["rm", "-rf", "ta-lib", "ta-lib-0.4.0-src.tar.gz"])
        
        # Set library path
        os.environ["LD_LIBRARY_PATH"] = "/usr/lib"
        return True
    except Exception as e:
        print(f"⚠️ Error installing TA-Lib from source: {e}")
        print("ℹ️ The system will use the fallback Python implementation.")
        return False

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("Installing required packages...")
        
        # Try to install/build TA-Lib first
        talib_installed = check_talib()
        
        # Install other requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        if talib_installed:
            print("✅ All dependencies installed successfully with TA-Lib!")
        else:
            print("✅ Dependencies installed with fallback TA-Lib implementation")
            print("ℹ️ Note: Using pure Python implementation for technical indicators")
            print("   This may be slower but provides the same functionality")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    print("\n🚀 Setup complete! You can now run:")
    print("- python fetch_data.py     # To download OHLCV data")
    print("- streamlit run streamlit_app.py  # To launch the GUI")