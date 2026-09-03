#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[3]
script = root / "examples/thermal/results/MSS7_twoPhase_sparse/mss7_two_phase_postprocess.py"
sys.argv = [str(script), "--case", str(Path(__file__).resolve().parent)]
runpy.run_path(str(script), run_name="__main__")
