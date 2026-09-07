#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys


script = Path(__file__).resolve().parents[2] / "draw.py"
sys.argv = [str(script), "--only", "temperature", *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
