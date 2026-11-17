pass
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_fixed_and_adaptive_window_steps(tmp_path):
    source=tmp_path/"step.cpp"
    source.write_text(r'''#include "applications/CHMT/gpu/GasTimeStepControl.H"
#include <cassert>
template<class R> void check(){
 const R fixed=R(.125), cap=R(.01), maximum=R(.02), tail=R(.03125);
 // Fixed mode ignores an old adaptive seed and the adaptive CFL/maxDt caps.
 const R requested=chmt::requestedGasMicrostep(false,fixed,R(.001),cap,maximum);
 assert(requested==fixed);
 const auto end=chmt::selectGasMicrostep(false,requested,tail,R(.001),R(1.1),maximum,true);
 assert(end.accepted==tail && end.next==fixed);
 const auto next=chmt::selectGasMicrostep(false,end.next,R(1),R(.001),R(1.1),maximum,true);
 assert(next.accepted==fixed && next.next==fixed);
 // Adaptive mode is exactly the existing selector, not a revised algorithm.
 const R proposal=chmt::requestedGasMicrostep(true,fixed,R(.008),cap,maximum);
 assert(proposal==R(.008));
 const auto actual=chmt::selectGasMicrostep(true,proposal,tail,R(.006),R(1.1),maximum,true);
 const auto previous=ugkwp::selectGasWindowStep(proposal,tail,R(.006),R(1.1),maximum,true);
 assert(actual.accepted==previous.accepted && actual.next==previous.next);
}
int main(){check<float>();check<double>();}
''')
    exe=tmp_path/"step"
    subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I"+str(ROOT),str(source),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_both_backend_paths_use_shared_step_policy():
    for prefix in (Path(),Path("precision/fp32")):
        text=(ROOT/prefix/"applications/CHMT/gpu/Backend.cu").read_text()
        assert "requestedGasMicrostep(b.options.adjustTimeStep" in text
        assert "selectGasMicrostep(b.options.adjustTimeStep" in text
        assert "const bool courantRefreshed=b.options.adjustTimeStep" in text
        assert ":Real(b.options.fixedDeltaT)" in text
    frontend=(ROOT/"applications/CHMT/CHMT.C").read_text()
    assert 'time.controlDict().lookupOrDefault<Foam::Switch>("adjustTimeStep",true)' in frontend
    assert 'time.controlDict().lookup<Foam::scalar>("deltaT")' in frontend
