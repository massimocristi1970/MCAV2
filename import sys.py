import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'app'))

# Now run the main app
if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"])