"""Classical pre-step selection and paid tails, without speculative retries."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
APP=(SOURCE_ROOT / 'applications/CHMT')
def test_comax_caps_and_paid_tail(tmp_path):
    source=tmp_path/'step.cpp'
    source.write_text(r'''
#include "gpu/GasWindowProgram.H"
#include <cassert>
using namespace chmt;
int main(){
 assert(gasCoMaxStep(1,.4,.8,1)==.5);
 assert(gasCoMaxStep(1,.4,0,0.5)==.525);
 assert(gasCoMaxStep(1,.4,.1,1)==1);
 // Diffusion and source are already equivalent Co; apply target only once.
 for(double co:{.8,1.6})assert(gasCoMaxStep(1,.4,co,1)==.4/co);
 assert(boundedGasMicrostep(0,1.5,1.25,2)==1.25);
 assert(boundedGasMicrostep(0,.5,2,2)==.5);
 WindowTime time,end(.001L);int steps=0;
 while(time<end){const double dt=std::min(1e-6,boundedWindowDuration(end,time));assert(dt>0);time+=dt;++steps;}
 assert(time==end);assert(steps>=1000&&steps<=1002);
}
''')
    binary=tmp_path/'step'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(APP.parents[1]/'common'),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
