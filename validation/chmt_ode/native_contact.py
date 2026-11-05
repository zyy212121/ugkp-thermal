"""Real moving CHMT + equilibrium ODE + shared finite contact and restart.
Uses the existing moving-wall verification fixtures and numerical assertions.
Requires a sourced OpenFOAM environment; all evidence stays in --output.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
import chmt_moving_particles_native as existing
original=existing.m.prepare
def prepare(*args,**kwargs):
    result=original(*args,**kwargs)
    case=Path(args[1]);p=case/'constant/chmtProperties'
    p.write_text(p.read_text().replace('model reactingSst;','model equilibriumOde;'))
    return result
existing.m.prepare=prepare
if __name__=='__main__':existing.main()
