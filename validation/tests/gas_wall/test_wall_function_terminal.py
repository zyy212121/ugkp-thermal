"""Root APIs retain finite terminal iterates and expose physical root failures."""
import subprocess
from validation.tools.paths import SOURCE_ROOT as ROOT


def test_root_failure_is_not_a_successful_zero(tmp_path):
    src=tmp_path/'root.cpp'
    src.write_text(r'''
#include "gasNumerics/OpenFoamWallFunctions.cuh"
#include <cassert>
#include <cmath>
#include <limits>
int main(){
 // A finite model with no matching root must not become a plausible zero.
 assert(std::isnan(ugkpwall::jayatillekeThermalYPlus(1.,.41,.01)));
 assert(std::isnan(ugkpwall::jayatillekeThermalYPlus(0.)));
 auto heat=ugkpwall::jayatillekeWallHeatFluxPrecomputed(1,1000,.7,.9,.41,9.8,0,10,1,0,300,500);
 assert(!heat.valid&&std::isnan(heat.heatFlux));
 ugkp::diagnostics::DeviceContext context{};
 const auto terminal=ugkpwall::jayatillekeThermalYPlus(.7/.9,.41,9.8,&context,18,0);
 assert(std::isfinite(terminal)&&terminal>0&&!context.fatalClaim);
 assert(context.warnings[1].count==1&&context.warnings[1].sample.entityIndex==18);
 context={};assert(std::isnan(ugkpwall::jayatillekeThermalYPlus(0,.41,9.8,&context,19)));
 assert(context.fatalClaim&&context.firstFatal.entityIndex==19&&context.firstFatal.sourceSiteId==2820);
 const auto y=ugkpwall::jayatillekeThermalYPlus(.7/.9);
 assert(std::isfinite(y)&&y>0);
}
''')
    exe=src.with_suffix('')
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
