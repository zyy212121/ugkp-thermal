"""Run the host stage body to verify gradient freshness and preflight reuse."""
import re
import subprocess
from test_gas_advance_view import PREAMBLE, ROOT


def test_every_wall_stage_prepares_current_gradients_once(tmp_path):
    source = re.sub(r'<<<[\s\S]*?>>>', '', (ROOT / 'common/GpuGasAdvance.cuh').read_text())
    preamble = PREAMBLE.replace(
        'using GasHostPolicy=GasHostWithWallEnergy<double,double,accumulateWall>;',
        r'''
#include <algorithm>
struct GasHostPolicy:GasHostWithWallEnergy<double,double,accumulateWall>{
 template<class S>static int prepareBoundaryLayer(S*,double,double){
  const auto gradients=std::count(trace.begin(),trace.end(),"computeGasPrimitiveGradientsKernel");
  const auto profiles=std::count(trace.begin(),trace.end(),"profile");
  if(gradients!=profiles+1){std::fprintf(stderr,"profile used stale or repeated gradients\n");std::abort();}
  trace.push_back("profile");return 0;
 }
};
''')
    extra = r'''
#include "gasTransport/GasStateView.H"
struct SpeciesTag{static constexpr int speciesCount=2;ugkwp::GasMode mode=ugkwp::GasMode::MixtureFrozen;};
struct Host:IndependentHost{SpeciesTag gasSpecies;ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;};
template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*,bool=false){}
struct Policy{
 static const double* stageVolumes(Host*,bool){return nullptr;}
 static int validate(Host*,bool){return 0;}
};
'''
    main = r'''
int count(const char* value){return std::count(trace.begin(),trace.end(),value);}
int main(){Host h;Payload device;h.deviceState=&device;h.hostTurbulenceModel=3;h.gasBoundaryLayer.enabled=true;
 for(int rk=1;rk<=3;++rk){
  h.hostGasTimeIntegrator=rk;trace.clear();ugkwp::invalidateGasBoundaryLayerPreparation(h);
  if(prepareGasTrialTransport<Policy>(&h,.01,2.))return 1;
  if(count("profile")!=1||count("computeGasPrimitiveGradientsKernel")!=1)return 2;
  if(advanceGasFluxStage(&h,.01,2.))return 3;
  if(count("profile")!=rk||count("computeGasPrimitiveGradientsKernel")!=rk)return 4;
 }
 // A changed interval must invalidate both the profile and its gradient reuse.
 trace.clear();ugkwp::invalidateGasBoundaryLayerPreparation(h);
 if(prepareGasTrialTransport<Policy>(&h,.01,2.))return 5;
 if(advanceGasEulerSubstage(&h,.005,.005,2.))return 6;
 if(count("profile")!=2||count("computeGasPrimitiveGradientsKernel")!=2)return 7;
}
'''
    path = tmp_path / 'stage_gradients.cpp'
    path.write_text(preamble + extra + source + main)
    executable = tmp_path / 'stage_gradients'
    built = subprocess.run(['g++', '-std=c++17', '-I' + str(ROOT / 'common'),
        str(path), '-o', str(executable)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + f' exit={result.returncode}'
