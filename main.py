"""Ray Demo - Entry Point"""

import sys
import os

# Ensure the demo directory is on sys.path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import run_server

if __name__ == "__main__":
    run_server()
