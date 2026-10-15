"""Make the validation helpers importable when suites are run from any directory."""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_ROOT))
