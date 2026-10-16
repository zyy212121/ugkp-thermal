"""CPU checks for single surface solve and explicit fraction diagnostics."""
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def test_surface_single_path_and_fraction_warning(tmp_path):
    source=tmp_path/'surface.cpp'
    source.write_text(r'''
#include "fixture.hpp"
int main(){
 auto p=physics();ugkp::diagnostics::DeviceContext diagnostics{};
 p.diagnostics=&diagnostics;p.tolerances.diagnostics=&diagnostics;
 auto h=host(p);auto in=input(h,p);SurfacePhysicsResult r;
 assert(solveSurfaceInterface(in,p,r));assert(std::isfinite(r.topTemperature));
 assert(!ugkp::diagnostics::hasFatal(&diagnostics));
 auto w=bulk(h,p);w.Y[0]=1.0000000000000002;
 auto q=conservativeGas(w,1,p);assert(std::isfinite(q.mass));
 assert(q.species[0]==q.mass);assert(diagnostics.warnings[2].count>0);
 w.temperature=-1;q=conservativeGas(w,1,p);assert(!std::isfinite(q.mass));
 double flux[Ns]{},fraction[Ns]{},out[Ns]{};fraction[0]=1;flux[0]=.01;
 for(auto& D:p.gasDiffusivity)D=.001;
 assert(interfaceComposition(flux,fraction,101325,300,.01,p,1e-12,out));
 assert(out[0]>=0&&out[0]<=1);assert(!ugkp::diagnostics::hasFatal(&diagnostics));
}
''')
    exe=tmp_path/'surface'
    subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Wno-misleading-indentation',
        '-I'+str(ROOT/'validation/tests/fixtures/surface_terminal'),'-I'+str(ROOT/'applications/CHMT'),
        '-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)

def test_removed_surface_retry_implementations():
    text=(ROOT/'applications/CHMT/ablation/InterfaceMath.H').read_text()
    assert 'alpha*=' not in text
    assert 'terminalTrial' not in text
    assert 'terminalSpecies' not in text
    assert 'if(Terminal)' not in text
    assert 'if(!valid)return false;' in text
    assert 'if(!seedValid)return false;' in text
