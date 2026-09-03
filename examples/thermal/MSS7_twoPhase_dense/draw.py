#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

script = Path(__file__).resolve().parent / "assets/postprocessing/mss7_two_phase_postprocess.py"
sys.argv = [str(script), "--case", str(Path(__file__).resolve().parent)]
runpy.run_path(str(script), run_name="__main__")
