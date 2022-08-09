"""
import os
import sys

module_path = [os.path.dirname(os.path.abspath(__file__))]
for path in module_path:
    if path not in sys.path:
        sys.path.append(path)

# Adds the package directory to the current path so that submodules can be appropriately added
"""