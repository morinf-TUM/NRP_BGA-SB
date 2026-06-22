"""Pytest configuration for nrp tests."""

import sys
from pathlib import Path

# Add project root to path so nrp package is importable
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
