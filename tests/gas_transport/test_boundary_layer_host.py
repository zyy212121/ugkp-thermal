"""Production cache scheduling, not a duplicated cache implementation."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_profile_preflight_reuse_rk_and_trial_invalidation(tmp_path):
    code=r'''
#include "gasTransport/GasBoundaryLayerHost.H"
#include <cassert>
struct Host{ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;int solves=0;};
struct Policy{static int prepareBoundaryLayer(Host*s,double,double){++s->solves;return 0;}};
int main(){Host s;s.gasBoundaryLayer.enabled=true;
ugkwp::invalidateGasBoundaryLayerPreparation(s); // new trial / post-chemistry state
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.1,2.,true)==0&&s.solves==1);
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.1,2.,false)==0&&s.solves==1); // first Euler reuses
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.1,2.1,false)==0&&s.solves==2); // RK fresh
ugkwp::invalidateGasBoundaryLayerPreparation(s); // rollback and retry
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.05,2.,true)==0&&s.solves==3);
ugkwp::invalidateGasBoundaryLayerPreparation(s); // material input changed after prepare
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.05,2.,false)==0&&s.solves==4);
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.05,2.,true)==0&&s.solves==5);
assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.025,2.,false)==0&&s.solves==6); // different interval
s.gasBoundaryLayer.enabled=false;assert(ugkwp::prepareGasBoundaryLayerStage<Policy>(&s,.1,2.,false)==0&&s.solves==6);
}
'''
    source=tmp_path/'probe.cpp';binary=tmp_path/'probe';source.write_text(code)
    build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(source),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr

def test_actual_trial_reuses_first_profile_and_retries_large_outer_interval(tmp_path):
    import re
    from test_gas_advance_view import PREAMBLE
    source=re.sub(r'<<<[\s\S]*?>>>','',(ROOT/'common/GpuGasAdvance.cuh').read_text())
    pre=PREAMBLE.replace('SIMPLE_KERNEL(computeGasPrimitiveGradientsKernel)',
        'template<class S>void computeGasPrimitiveGradientsKernel(S*,bool=false){trace.push_back("gas-gradients");}')
    pre=pre.replace('using GasHostPolicy=GasHostWithWallEnergy<double,double,accumulateWall>;',r'''
struct GasHostPolicy:GasHostWithWallEnergy<double,double,accumulateWall>{
 template<class S>static int prepareBoundaryLayer(S*,double,double){trace.push_back("profile");return 0;}};
''')
    extras=r'''
#include "gasTransport/GasStateView.H"
#include <algorithm>
struct SpeciesTag{static constexpr int speciesCount=2;ugkwp::GasMode mode=ugkwp::GasMode::MixtureChemistry;};
struct MixedDevice{SpeciesTag gasSpecies;ugkwp::GasSstAuditState<double> gasSstAudit;};
template<class S>void prepareGasSstAuditKernel(S*){trace.push_back("audit-volume");}
struct MixedHost:IndependentHost{SpeciesTag gasSpecies;MixedDevice*deviceState;ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;};
template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*){trace.push_back("chemistry");}
template<class S>void computeGasCourantFieldKernel(S*,double){trace.push_back("fresh-mass");}
template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){}
template<class S>void computeGasDiffusionNumberKernel(S*,double,double){}
template<class S>void computeSstStabilityNumberKernel(S*,double,double){trace.push_back("sst-bound");}
struct NativeAuditHost:MixedHost{ugkwp::GasSstAuditState<double> gasSstAudit;};
struct Policy{
 static inline int attempts=0,rollbacks=0;static inline bool limit=false;static inline double covered=0,current=0;
 static int begin(MixedHost*){++attempts;return 0;}static int applySources(MixedHost*,double dt){current=dt;return 0;}
 static int validate(MixedHost*){return 0;}static int commit(MixedHost*){covered+=current;return 0;}
 static void rollback(MixedHost*){++rollbacks;}static int beginInterval(MixedHost*){covered=0;return 0;}
 static void rollbackInterval(MixedHost*){covered=0;}static int commitInterval(MixedHost*){return 0;}
 static bool retryable(MixedHost*){return true;}
 static int validateTimeStep(MixedHost*,double dt){return limit&&dt>.02500000001;}
 static double targetMaxCo(MixedHost*){return .5;}static const double*stageVolumes(MixedHost*,bool){return nullptr;}
 static int captureChemistryAudit(MixedHost*,bool){return 0;}
};
'''
    main=r'''
int count(const char*s){return std::count(trace.begin(),trace.end(),s);}
int prefixes(const char*s){int n=0;for(auto&t:trace)if(t.find(s)==0)++n;return n;}
int main(){MixedHost h;MixedDevice d;h.deviceState=&d;h.hostTurbulenceModel=3;h.gasBoundaryLayer.enabled=true;
for(int rk=1;rk<=3;++rk){h.hostGasTimeIntegrator=rk;trace.clear();if(advanceGasTrial<Policy>(&h,.01,2.))return 1;
if(count("profile")!=rk)return 2;if(count("audit-volume")!=1)return 10;if(count("chemistry")!=2)return 3;
if(prefixes("updateLegacyGasBoundaryMirrorKernel")!=2)return 4;
if(count("updateRiemannBoundaryMirrorKernel")!=rk+1)return 5;
if(std::find(trace.begin(),trace.end(),"chemistry")>std::find(trace.begin(),trace.end(),"profile"))return 6;
}
for(int enabled=0;enabled<2;++enabled){NativeAuditHost native;native.deviceState=&d;native.hostTurbulenceModel=3;native.gasBoundaryLayer.enabled=true;native.gasSstAudit.enabled=enabled;trace.clear();
if(advanceGasTrial<Policy>(&native,.01,2.))return 11;if(count("audit-volume")!=enabled||count("profile")!=1)return 12;}
h.hostGasTimeIntegrator=1;Policy::limit=true;Policy::attempts=Policy::rollbacks=0;trace.clear();
GasRequestedIntervalControls<double> controls;controls.minimumSubstep=1e-8;
if(advanceGasRequestedInterval<Policy>(&h,.1,2.,controls))return 7;
if(Policy::rollbacks<1||Policy::attempts>12||std::abs(Policy::covered-.1)>1e-12)return 8;
if(count("profile")!=Policy::attempts)return 9; // every rejected/retried state has a fresh prepare
}
'''
    cpp=tmp_path/'stage.cpp';binary=tmp_path/'stage';cpp.write_text(pre+extras+source+main)
    build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(cpp),'-o',str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr+f' exit={run.returncode}'
