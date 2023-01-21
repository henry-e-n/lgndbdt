import os
import sys

module_path = [os.path.dirname(os.path.abspath(__file__))]
for path in module_path:
    if path not in sys.path:
        sys.path.append(path)

print("Welcome to lgndbdt ...")
print("This package was created by Henry Nachman for use by the LEGEND Collaboration")
from setup import VERSION
print(f"Current version is v{VERSION}")

print("""Summary__________
This package is designed to train and validate a 
multi-site background discrimination, interpretable Boosted 
Decision Tree Model from raw HPGe data using pulse shape parameters.

The pipeline consists of 4 major components:
1. Energy Calibration
2. Waveform Extraction
3. Pulse Shape Parameter Extraction
4. BDT Training and Validation """)
# Adds the package directory to the current path so that submodules can be appropriately added
