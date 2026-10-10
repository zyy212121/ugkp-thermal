"""Execute the actual host protocol with recorded CUDA launches, not a GPU.

Only CUDA launch punctuation is translated; host ordering, branches, arguments,
error handling and graph-time rebinding are the production header bodies.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

PREAMBLE = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <type_traits>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaStreamCaptureModeThreadLocal=0,cudaGraphNodeTypeKernel=0;
using cudaStream_t=void*; using cudaGraph_t=void*; using cudaGraphExec_t=void*;
using cudaGraphNode_t=int; using cudaGraphNodeType=int;
struct cudaKernelNodeParams { void* func=nullptr; void** kernelParams=nullptr; void** extra=nullptr; };
std::vector<std::string> trace;
int launchCount=0,failLaunch=0;
int cudaGetLastError(){return ++launchCount==failLaunch?91:0;}
void setLastError(const char*msg,int){trace.push_back(msg);}
void setLastErrorText(const char*msg){trace.push_back(msg);}
int cudaGraphExecDestroy(cudaGraphExec_t){return 0;} int cudaGraphDestroy(cudaGraph_t){return 0;}
int cudaStreamCreate(cudaStream_t*s){*s=reinterpret_cast<void*>(1);return 0;}
int cudaStreamDestroy(cudaStream_t){return 0;}
int cudaStreamBeginCapture(cudaStream_t,int){return 0;}
int cudaStreamEndCapture(cudaStream_t,cudaGraph_t*g){*g=reinterpret_cast<void*>(2);return 0;}
std::vector<cudaKernelNodeParams> graphNodes;
int cudaGraphGetNodes(cudaGraph_t,cudaGraphNode_t*nodes,std::size_t*n){if(nodes){for(std::size_t j=0;j<*n;++j)nodes[j]=int(j);}else *n=graphNodes.size();return 0;}
int cudaGraphNodeGetType(cudaGraphNode_t,cudaGraphNodeType*t){*t=cudaGraphNodeTypeKernel;return 0;}
int cudaGraphKernelNodeGetParams(cudaGraphNode_t n,cudaKernelNodeParams*p){*p=graphNodes[n];return 0;}
int cudaGraphInstantiate(cudaGraphExec_t*e,cudaGraph_t,void*,void*,int){*e=reinterpret_cast<void*>(3);return 0;}
void*expectedDevice=nullptr;double expectedTime=0;
int cudaGraphExecKernelNodeSetParams(cudaGraphExec_t,cudaGraphNode_t,cudaKernelNodeParams*p){if(*static_cast<void**>(p->kernelParams[0])!=expectedDevice||*static_cast<double*>(p->kernelParams[1])!=expectedTime)std::abort();trace.push_back("graph-time");return 0;}
int cudaGraphLaunch(cudaGraphExec_t,void*){trace.push_back("graph-launch");return 0;}
#ifdef LEGACY_REFERENCE
struct DeviceState;
using Payload=DeviceState;
#define HOST_TYPE DeviceState
#else
struct Payload{};
#define HOST_TYPE IndependentHost
#endif
struct HOST_TYPE {
 int fixedCellBlockThreads=128,fixedFaceBlockThreads=128,nCells=4,nFaces=5;
 int hostGasTimeIntegrator=1,hostTurbulenceModel=0,hostGasFluxScheme=2,hasPeriodicFaces=0;
 Payload*deviceState=nullptr;cudaStream_t gasCaptureStream=nullptr;
 cudaGraph_t gasGraph=nullptr;cudaGraphExec_t gasGraphExec=nullptr;
 double gasGraphDt=-1;std::vector<cudaGraphNode_t> gasGraphTimeNodes;
 std::vector<cudaKernelNodeParams> gasGraphTimeParams;
};
#ifdef LEGACY_REFERENCE
using IndependentHost=DeviceState;
#endif
#ifdef LEGACY_REFERENCE
#define STATE_TEMPLATE
#define STATE DeviceState
#else
#define STATE_TEMPLATE template<class State>
#define STATE State
#endif
#define SIMPLE_KERNEL(Name) STATE_TEMPLATE void Name(STATE*){trace.push_back(#Name);}
SIMPLE_KERNEL(recoverGasPrimitivesKernel)
SIMPLE_KERNEL(recoverSstPrimitivesKernel)
SIMPLE_KERNEL(updateRiemannBoundaryMirrorKernel)
SIMPLE_KERNEL(applySstWallFunctionStateKernel)
SIMPLE_KERNEL(computeGasHllcAdcSensorKernel)
SIMPLE_KERNEL(computeGasPrimitiveGradientsKernel)
SIMPLE_KERNEL(computeSstGradientsKernel)
SIMPLE_KERNEL(computeGasGradientLimiterKernel)
SIMPLE_KERNEL(computeGasEddyViscosityKernel)
SIMPLE_KERNEL(enforcePeriodicGasFluxAntisymmetryKernel)
SIMPLE_KERNEL(applyGasFluxPositivityScaleKernel)
SIMPLE_KERNEL(computeSstFaceFluxKernel)
SIMPLE_KERNEL(enforcePeriodicSstFluxAntisymmetryKernel)
SIMPLE_KERNEL(saveGasConservativeStateKernel)
#define TIMED_KERNEL(Name) STATE_TEMPLATE void Name(STATE*,double dt){char b[100];std::snprintf(b,sizeof(b),#Name " %.17g",dt);trace.push_back(b);}
TIMED_KERNEL(computeGasFluxPositivityScaleKernel)
TIMED_KERNEL(applySstFluxAndSourceKernel)
TIMED_KERNEL(applyGasFluxDivergenceByCellKernel)
TIMED_KERNEL(updateWaveTransmissivePressureBoundaryKernel)
STATE_TEMPLATE void updateLegacyGasBoundaryMirrorKernel(STATE*,double dt){char b[100];std::snprintf(b,sizeof(b),"updateLegacyGasBoundaryMirrorKernel %.17g",dt);trace.push_back(b);
#ifdef LEGACY_REFERENCE
 graphNodes.push_back({reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel),nullptr,nullptr});
#else
 graphNodes.push_back({reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel<State>),nullptr,nullptr});
#endif
}
#ifdef LEGACY_REFERENCE
template<bool IncludeTurbulence>
#else
template<bool IncludeTurbulence,class State>
#endif
void computeGasInternalFaceFluxKernel(STATE*,double dt){char b[100];std::snprintf(b,sizeof(b),"face-flux %d %.17g",IncludeTurbulence,dt);trace.push_back(b);}
STATE_TEMPLATE void blendGasConservativeStateKernel(STATE*,double a,double b){char msg[100];std::snprintf(msg,sizeof(msg),"blend %.17g %.17g",a,b);trace.push_back(msg);}
int accumulateWall(HOST_TYPE*,double dt){char b[100];std::snprintf(b,sizeof(b),"wall-ledger %.17g",dt);trace.push_back(b);return 0;}
#include "GpuGasHostPolicy.cuh"
using GasHostPolicy=GasHostWithWallEnergy<double,double,accumulateWall>;
'''

MAIN = r'''
int main(){
 for(int rk=1;rk<=3;++rk)for(int turbulence:{0,3})for(int periodic:{0,1})for(int scheme:{2,7}){
 IndependentHost h;Payload payload;h.deviceState=&payload;h.hostGasTimeIntegrator=rk;h.hostTurbulenceModel=turbulence;h.hasPeriodicFaces=periodic;h.hostGasFluxScheme=scheme;
 trace.clear();launchCount=0;graphNodes.clear();if(advanceGasFluxStage(&h,.3,2.)||finaliseGasBoundaryStage(&h,.3,2.))std::abort();
 std::printf("case %d %d %d %d\n",rk,turbulence,periodic,scheme);for(const auto&s:trace)std::puts(s.c_str());
 }
 IndependentHost h;Payload payload;h.deviceState=&payload;trace.clear();launchCount=0;failLaunch=2;
 if(advanceGasFluxStage(&h,.3,2.)!=1||trace.back()!="updateLegacyGasBoundaryMirrorKernel before gas advance")std::abort();
 failLaunch=0;trace.clear();launchCount=0;graphNodes.clear();expectedDevice=&payload;expectedTime=2.;
 if(advancePureGasGraph(&h,.3,2.)!=0||h.gasGraphTimeNodes.size()!=2)std::abort();
 graphNodes.clear();trace.clear();expectedTime=5.;if(advancePureGasGraph(&h,.3,5.)!=0)std::abort();
 if(trace!=std::vector<std::string>{"graph-time","graph-time","graph-launch"})std::abort();
 std::puts("PASS errors and graph-time rebinding");
}
'''

def compile_and_run(tmp_path, source, legacy=False):
    # Explicitly emulate launches; no numerical expression or host branch changes.
    host = re.sub(r"<<<[\s\S]*?>>>", "", source)
    path = tmp_path / "host.cpp"
    (tmp_path / "gas_advance_emulated.hpp").write_text(host)
    path.write_text(PREAMBLE + '\n#include "gas_advance_emulated.hpp"\n' + MAIN)
    exe = tmp_path / "host"
    flags = ["-DLEGACY_REFERENCE"] if legacy else []
    result = subprocess.run(["g++", "-std=c++17", "-O0", *flags, "-I"+str(ROOT / "common"), str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout

def test_shared_host_protocol_accepts_separate_gas_device_view(tmp_path):
    result = compile_and_run(tmp_path, (ROOT / "common/GpuGasAdvance.cuh").read_text())
    assert result == (HERE / "legacy_gas_host_trace.txt").read_text()

def test_host_policies_do_not_require_an_application_device_state(tmp_path):
    source = tmp_path / "policies.cpp"
    source.write_text(r'''
#include "GpuGasHostPolicy.cuh"
struct GasHost { int hostGasTimeIntegrator; double acceptedEnergy; };
int ledger(GasHost* h,double dt){h->acceptedEnergy+=dt;return 0;}
using NoWall=GasHostWithoutWallEnergy<double,float>;
using Wall=GasHostWithWallEnergy<double,float,ledger>;
int main(){GasHost h{2,0};if(NoWall::firstLedgerDt(&h,1)!=0||NoWall::accumulateWallEnergy(&h,1))return 1;
if(Wall::firstLedgerDt(&h,.6)!=.3||Wall::accumulateWallEnergy(&h,.3)||h.acceptedEnergy!=.3)return 2;
h.hostGasTimeIntegrator=3;if(Wall::firstLedgerDt(&h,.6)!=.6*(1./6.)||Wall::finalLedgerDt(.6)!=.6*(2./3.))return 3;}
''')
    exe=tmp_path / "policies"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I"+str(ROOT / "common"),str(source),"-o",str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    assert subprocess.run([str(exe)]).returncode==0

def test_transaction_entry_rolls_back_failed_common_stage(tmp_path):
    source=(ROOT/'common/GpuGasAdvance.cuh').read_text()
    host=re.sub(r"<<<[\s\S]*?>>>","",source)
    main=r'''
struct TrialPolicy {
 static int begin(IndependentHost*){trace.push_back("begin");return 0;}
 static int applySources(IndependentHost*,double){trace.push_back("sources");return 0;}
 static int validate(IndependentHost*){trace.push_back("validate");return reject;}
 static int commit(IndependentHost*){trace.push_back("commit");return 0;}
 static void rollback(IndependentHost*){trace.push_back("rollback");}
 static int reject;
};int TrialPolicy::reject=0;
int main(){IndependentHost h;Payload p;h.deviceState=&p;
if(advanceGasTrial<TrialPolicy>(&h,.1,2.))return 1;
if(trace.front()!="begin"||trace.back()!="commit"||trace[trace.size()-2]!="validate")return 2;
trace.clear();TrialPolicy::reject=1;if(advanceGasTrial<TrialPolicy>(&h,.1,2.)!=1||trace.back()!="rollback")return 3;
for(auto&s:trace)if(s=="commit")return 4;
trace.clear();TrialPolicy::reject=0;failLaunch=launchCount+1;
if(advanceGasTrial<TrialPolicy>(&h,.1,2.)!=1||trace.back()!="rollback")return 5;
for(auto&s:trace)if(s=="sources"||s=="commit")return 6;
}
'''
    (tmp_path/'protocol.hpp').write_text(host)
    p=tmp_path/'trial.cpp';p.write_text(PREAMBLE+'\n#include "protocol.hpp"\n'+main)
    exe=tmp_path/'trial'
    build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(p),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    assert subprocess.run([str(exe)]).returncode==0

def test_requested_interval_retries_without_partial_time_or_budget_publication(tmp_path):
    source=re.sub(r"<<<[\s\S]*?>>>","",(ROOT/'common/GpuGasAdvance.cuh').read_text())
    main=r'''
struct Host:IndependentHost{double budget=0,trialBase=0,intervalBase=0,lastDt=0,clock=2;int commits=0;};
struct Policy{
 static int begin(Host*h){h->trialBase=h->budget;return 0;}
 static int applySources(Host*h,double dt){h->lastDt=dt;h->budget+=dt;return 0;}
 static int validate(Host*h){return h->lastDt>.2?1:0;}
 static int commit(Host*h){++h->commits;return 0;}
 static void rollback(Host*h){h->budget=h->trialBase;}
 static int beginInterval(Host*h){h->intervalBase=h->budget;return 0;}
 static int commitInterval(Host*){return 0;}
 static void rollbackInterval(Host*h){h->budget=h->intervalBase;}
 static bool retryable(Host*){return true;}
};
int main(){Host h;Payload p;h.deviceState=&p;GasRequestedIntervalControls<double> controls;
if(advanceGasRequestedInterval<Policy>(&h,.6,2.,controls))return 1;
if(std::abs(h.budget-.6)>1e-14||h.commits<3||h.clock!=2)return 2;
controls.maximumAttempts=1;double before=h.budget;
if(advanceGasRequestedInterval<Policy>(&h,.6,2.6,controls)!=1||h.budget!=before||h.clock!=2)return 3;
controls.maximumAttempts=100;controls.minimumSubstep=.3;
if(advanceGasRequestedInterval<Policy>(&h,.6,2.6,controls)!=1||h.budget!=before)return 4;
}
'''
    (tmp_path/'protocol.hpp').write_text(source)
    cpp=tmp_path/'interval.cpp';cpp.write_text('#include <cmath>\n'+PREAMBLE+'\n#include "protocol.hpp"\n'+main)
    exe=tmp_path/'interval';build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    assert subprocess.run([str(exe)]).returncode==0

def test_chemistry_composition_checks_post_half_cfl_and_preserves_order(tmp_path):
    source=re.sub(r"<<<[\s\S]*?>>>","",(ROOT/'common/GpuGasAdvance.cuh').read_text())
    extras=r'''
#include "gasTransport/GasStateView.H"
struct SpeciesTag {static constexpr int speciesCount=2;ugkwp::GasMode mode=ugkwp::GasMode::MixtureChemistry;};
struct MixedHost:IndependentHost{SpeciesTag gasSpecies;double*V=nullptr;};
template<class S,class T>void advanceGasChemistryKernel(S*,T dt,const double*){trace.push_back(dt==.05?"half-chemistry":"BAD-INTERVAL");}
template<class S>void computeGasCourantFieldKernel(S*,double){trace.push_back("wave-speed");}
template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){trace.push_back("courant");}
template<class S>void computeGasDiffusionNumberKernel(S*,double,double){trace.push_back("diffusion-bound");}
struct Policy {
 static int begin(MixedHost*){trace.push_back("begin");return 0;}
 static int applySources(MixedHost*,double){trace.push_back("sources");return 0;}
 static int validate(MixedHost*){trace.push_back("validate");return 0;}
 static int commit(MixedHost*){trace.push_back("commit");return 0;}
 static void rollback(MixedHost*){trace.push_back("rollback");}
 static const double*stageVolumes(MixedHost*h,bool){return h->V;}
 static int captureChemistryAudit(MixedHost*,bool after){trace.push_back(after?"post-audit":"pre-audit");return 0;}
 static double targetMaxCo(MixedHost*){return .5;}
 static int validateTimeStep(MixedHost*,double){trace.push_back("check-CFL");return reject;}
 static int reject;
};int Policy::reject=0;
'''
    main=r'''
int main(){MixedHost h;Payload p;h.deviceState=&p;if(advanceGasTrial<Policy>(&h,.1,2.))return 1;
int first=-1,last=-1,cfl=-1,flux=-1,n=0;for(int i=0;i<int(trace.size());++i){if(trace[i]=="half-chemistry"){if(n++==0)first=i;last=i;}if(trace[i]=="check-CFL")cfl=i;if(trace[i].find("face-flux")==0)flux=i;}
if(n!=2||!(first<cfl&&cfl<flux&&flux<last)||trace.back()!="commit")return 2;
trace.clear();Policy::reject=1;if(advanceGasTrial<Policy>(&h,.1,2.)!=1||trace.back()!="rollback")return 3;
for(auto&x:trace)if(x.find("face-flux")==0||x=="post-audit"||x=="commit")return 4;
}
'''
    (tmp_path/'protocol.hpp').write_text(source)
    cpp=tmp_path/'chemistry.cpp';cpp.write_text(PREAMBLE+extras+'\n#include "protocol.hpp"\n'+main)
    exe=tmp_path/'chemistry';build=subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stdout+build.stderr
    assert subprocess.run([str(exe)]).returncode==0

def test_shared_sst_preflight_consumes_fresh_mass_predictor(tmp_path):
    """Execute the production preflight; SST may not consume stale face scratch."""
    source = re.sub(r'<<<[\s\S]*?>>>', '', (ROOT/'common/GpuGasAdvance.cuh').read_text())
    preamble = PREAMBLE.replace('SIMPLE_KERNEL(computeGasPrimitiveGradientsKernel)',
        'template<class S>void computeGasPrimitiveGradientsKernel(S*,bool=false){trace.push_back("gas-gradients");}')
    extras = r'''
#include "gasTransport/GasStateView.H"
struct SpeciesTag{static constexpr int speciesCount=2;ugkwp::GasMode mode=ugkwp::GasMode::MixtureFrozen;};
struct MixedDevice{SpeciesTag gasSpecies;};
struct MixedHost:IndependentHost{SpeciesTag gasSpecies;MixedDevice*deviceState;};
template<class S,class T>void advanceGasChemistryKernel(S*,T,const double*){}
template<class S>void computeGasCourantFieldKernel(S*,double){trace.push_back("fresh-mass");}
template<class S>void computeGasConvectiveCourantByCellKernel(S*,double){}
template<class S>void computeGasDiffusionNumberKernel(S*,double,double){}
template<class S>void computeSstStabilityNumberKernel(S*,double,double){trace.push_back("sst-bound");}
struct Policy{
 static int validate(MixedHost*){return 0;}static int validateTimeStep(MixedHost*,double){return 0;}
 static double targetMaxCo(MixedHost*){return .5;}static const double*stageVolumes(MixedHost*,bool){return nullptr;}
 static int captureChemistryAudit(MixedHost*,bool){return 0;}
};
'''
    main = r'''
#include <algorithm>
int main(){MixedHost h;MixedDevice d;h.deviceState=&d;h.hostTurbulenceModel=3;
if(prepareGasTrialTransport<Policy>(&h,.01))return 1;
auto position=[](const char*name){return std::find(trace.begin(),trace.end(),name)-trace.begin();};
for(const auto& line:trace)std::puts(line.c_str());
if(position("fresh-mass")>=position("computeSstGradientsKernel"))return 2;
if(position("computeSstGradientsKernel")>=position("sst-bound"))return 3;
if(position("computeGasGradientLimiterKernel")>=position("fresh-mass"))return 4;
}
'''
    cpp = tmp_path/'preflight.cpp';cpp.write_text(preamble+extras+source+main)
    exe = tmp_path/'preflight'
    subprocess.run(['g++','-std=c++17','-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
