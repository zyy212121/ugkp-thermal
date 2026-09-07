#!/usr/bin/env python3
from pathlib import Path
import runpy,sys
case=Path(__file__).resolve().parent
script=case.parent/'MSS7_turbulent_wallModel/draw.py'
sys.argv=[str(script),'--case',str(case),*sys.argv[1:]]
runpy.run_path(str(script),run_name='__main__')
