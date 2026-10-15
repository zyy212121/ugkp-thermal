"""Locations shared by source validation tools and their regression suites."""
from pathlib import Path
import os

VALIDATION_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = VALIDATION_ROOT.parent
CASES_ROOT = Path(os.environ.get("UGKP_CASES_ROOT", str(SOURCE_ROOT.parent / "ugkp-thermal-cases"))).expanduser().resolve()
