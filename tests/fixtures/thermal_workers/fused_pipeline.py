"""Compatibility entry point; validation uses the common S1/L2 pipeline."""
from pathlib import Path
import subprocess
import sys
raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("s1_pipeline.py")), *sys.argv[1:]]))
