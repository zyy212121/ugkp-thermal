"""Acoustic-only resident wrapper/API coverage (runtime kernel tests live separately)."""
from validation.tools.paths import SOURCE_ROOT
import pytest

@pytest.mark.parametrize("app", ["gasUGKP", "FSH", "CHT"])
def test_courant_abi_has_no_nonacoustic_target(app):
    base = SOURCE_ROOT / "applications" / app
    api = (base / "gpu/GpuBackendApi.H").read_text()
    signature = api.split("ugkwpGpuResidentStrictComputeGasCourant", 1)[1].split(";", 1)[0]
    assert "dt" in signature and "scheduleTime" in signature
    assert "targetMaxCo" not in signature
    backend = base / ("gpu/GpuResidentStrict.cu" if app == "CHT" else "private_backend/GpuResidentStrict.cu")
    source = backend.read_text()
    assert "computeGasDiffusionNumberKernel" not in source
    assert "computeSstStabilityNumberKernel" not in source
    assert "gasDiffusionNumber" not in source
    assert "sstSourceNumber" not in source

@pytest.mark.parametrize("app", ["gasUGKP", "FSH", "CHT"])
def test_outer_growth_and_maximum_step_controls_remain(app):
    source = (SOURCE_ROOT / "applications" / app / "diluteUgkwpFoam.C").read_text()
    assert "oldDt*maxCo/max(lastMeasuredCo, SMALL)" in source
    assert "min(requestedDt, oldDt*gpuResidentMaxDeltaTGrowth)" in source
    assert "runTime.setDeltaT(min(growthLimitedDt, maxDeltaT))" in source
    assert "adjustTimeStep" in source

def test_outer_timestep_formula_numerically_preserves_caps(tmp_path):
    """Execute the unchanged production formula, rather than reimplement it."""
    import re
    import subprocess
    source = (SOURCE_ROOT / 'applications/gasUGKP/diluteUgkwpFoam.C').read_text()
    formula = re.search(r'const scalar requestedDt =\s*oldDt\*maxCo/max\(lastMeasuredCo, SMALL\);\s*const scalar growthLimitedDt =\s*min\(requestedDt, oldDt\*gpuResidentMaxDeltaTGrowth\);\s*runTime.setDeltaT\(min\(growthLimitedDt, maxDeltaT\)\);', source).group()
    code = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
using scalar=double;using std::min;using std::max;
struct Clock{double value;void setDeltaT(double x){value=x;}};
double step(double oldDt,double maxCo,double lastMeasuredCo,double gpuResidentMaxDeltaTGrowth,double maxDeltaT,bool adjustTimeStep=true){
 Clock runTime{oldDt};constexpr double SMALL=1e-15;
 if(adjustTimeStep&&lastMeasuredCo>SMALL){FORMULA}
 return runTime.value;
}
int main(){
 auto equal=[](double a,double b){assert(std::abs(a-b)<1e-13);};
 equal(step(1.,.5,2.,1.2,10.),.25); // Acoustic CFL limits.
 equal(step(1.,.5,.01,1.2,10.),1.2); // Growth limits.
 equal(step(1.,.5,.01,1.2,1.1),1.1); // Maximum dt limits.
 equal(step(1.,.5,2.,1.2,10.,false),1.); // Fixed dt unchanged.
 equal(step(1.,.5,0.,1.2,10.),1.); // Existing zero-Co behavior.
}
'''.replace('FORMULA',formula)
    path=tmp_path/'outer.cpp';path.write_text(code);exe=tmp_path/'outer'
    subprocess.run(['g++','-std=c++17',str(path),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
