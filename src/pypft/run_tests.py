"""
Cross-platform test runner for PyPFT
"""

import sys
import subprocess

def main():
    """Run all tests using unittest discover"""
    # Runs `python -m unittest discover -s tests`
    retcode = subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
    sys.exit(retcode)

if __name__ == "__main__":
    main()
