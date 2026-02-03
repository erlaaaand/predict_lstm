#!/usr/bin/env python3
"""
Run script for Stock Prediction Application
"""

import subprocess
import sys


def main():
    """Run the Streamlit application."""
    try:
        subprocess.run([
            "streamlit", "run", "main.py",
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit is not installed. Please run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
