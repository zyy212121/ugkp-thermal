#!/usr/bin/env python3
from pathlib import Path
import runpy


script = Path(__file__).resolve().parent / "bentsrm_wall_model_comparison.py"
runpy.run_path(str(script), run_name="__main__")
