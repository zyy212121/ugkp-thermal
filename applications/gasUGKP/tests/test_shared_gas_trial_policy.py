"""Real native ownership policy with host-memory CUDA-copy shim, no GPU claim."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def test_trial_and_requested_interval_rollback_restore_accepted_state(tmp_path):
    scalar_fields="rho rhoUx rhoUy rhoUz rhoE Ux Uy Uz p Tgas gasFluxPositivityScale gasDiffusionNumber sstSourceNumber rhoK rhoOmega k omega nut gasBoundaryRho gasBoundaryUx gasBoundaryUy gasBoundaryUz gasBoundaryP gasBoundaryT V".split()
    declarations="\n".join("double* "+name+"=nullptr;" for name in scalar_fields)
    assignments="\n".join(f"double {name}[2]={{1,1}};state.{name}={name};" for name in scalar_fields)
    code=r'''
#include "common/gasTransport/GasBuildConfig.H"
#include "common/gasTransport/GasStateView.H"
#include "common/gasTransport/GasCapabilities.H"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
using cudaError_t=int;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToDevice=2;
int allocations=0,deviceCopyFailures=0;
int cudaFree(void*p){if(p){--allocations;std::free(p);}return 0;}
int cudaMemcpy(void*d,const void*s,std::size_t n,int kind){if(kind==cudaMemcpyDeviceToDevice && deviceCopyFailures>0){--deviceCopyFailures;return 1;}if(n)std::memcpy(d,s,n);return 0;}
int cudaMemset(void*d,int v,std::size_t n){if(n)std::memset(d,v,n);return 0;}
int cudaDeviceSynchronize(){return 0;}
void setLastErrorText(const char*){}
void setLastError(const char*,int){}
template<class T>int allocate(T*&p,std::size_t n,const char*){p=n?static_cast<T*>(std::malloc(n*sizeof(T))):nullptr;if(p)++allocations;return 0;}
template<class T>int copyToDevice(T*d,const T*s,std::size_t n,const char*){if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
template<class T>int copyToHost(T*d,const T*s,std::size_t n,const char*){if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"
struct DeviceState{
 ugkwp::GasSpeciesState<double,2> gasSpecies;
 SharedGasTrialStorage gasTrial;
 DeviceState* deviceState=nullptr;
 int nCells=2,nFaces=2,fixedCellBlockThreads=128,hostTurbulenceModel=0;
 bool gasModelPoisoned=false;
 __DECLARATIONS__
};
int applyGasGravitySource(DeviceState*,int,int,double){return 0;}
#include "applications/gasUGKP/private_backend/SharedGasStorage.cuh"
#include "applications/gasUGKP/private_backend/SharedGasTrialPolicy.cuh"
int main(){
 DeviceState state;
 __ASSIGNMENTS__
 double partial[4]={.4,.4,.6,.6},sound[2]={1,1},cp[2]={1,1},r[2]={1,1};
 int cellStatus[2]={0,0},faceStatus[2]={0,0};
 state.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;state.gasSpecies.rho=partial;
 state.gasSpecies.soundSpeed=sound;state.gasSpecies.heatCapacity=cp;state.gasSpecies.gasConstant=r;
 state.gasSpecies.cellStatus=cellStatus;state.gasSpecies.faceStatus=faceStatus;
 assert(SharedGasTrialPolicy::beginInterval(&state)==0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.rho[0]=1.1;partial[0]=.5;state.gasBoundaryP[0]=2;
 assert(SharedGasTrialPolicy::validate(&state)==0);
 assert(!SharedGasTrialPolicy::retryable(&state));
 assert(SharedGasTrialPolicy::commit(&state)==0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.rho[0]=9;partial[0]=-1;state.gasBoundaryP[0]=8;
 cellStatus[0]=int(ugkwp::GasTransportCode::NegativeInventory);
 assert(SharedGasTrialPolicy::validate(&state)!=0);
 assert(SharedGasTrialPolicy::retryable(&state));
 SharedGasTrialPolicy::rollback(&state);
 assert(state.rho[0]==1.1 && partial[0]==.5 && state.gasBoundaryP[0]==2 && cellStatus[0]==0);
 SharedGasTrialPolicy::rollbackInterval(&state);
 assert(state.rho[0]==1 && partial[0]==.4 && state.gasBoundaryP[0]==1);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.gasFluxPositivityScale[0]=2;
 assert(SharedGasTrialPolicy::validateTimeStep(&state,.1)!=0);
 assert(SharedGasTrialPolicy::retryable(&state));
 SharedGasTrialPolicy::rollback(&state);
 ugkwp::ChemistryStatus chemistry[2];
 chemistry[0].code=ugkwp::ChemistryCode::NewtonFailure;
 chemistry[1].code=ugkwp::ChemistryCode::InvalidModel;
 state.gasSpecies.chemistryStatus=chemistry;
 assert(sharedGasTrialDetail::chemistryStatus(&state)!=0);
 assert(!SharedGasTrialPolicy::retryable(&state));
 state.gasSpecies.chemistryStatus=nullptr;
 state.hostTurbulenceModel=3;
 for(int i=0;i<2;++i){state.gasFluxPositivityScale[i]=.1;state.gasDiffusionNumber[i]=.1;state.sstSourceNumber[i]=.1;}
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.sstSourceNumber[0]=2;
 assert(SharedGasTrialPolicy::validateTimeStep(&state,.1)!=0);
 assert(SharedGasTrialPolicy::retryable(&state));
 SharedGasTrialPolicy::rollback(&state);
 assert(state.sstSourceNumber[0]==.1);
 assert(SharedGasTrialPolicy::validateTimeStep(&state,.1)==0);
 for(int interval=0;interval<2;++interval){
  // Reset only to construct the independent failure scenario in this fixture.
  state.gasModelPoisoned=false;state.rho[0]=1;
  assert(SharedGasTrialPolicy::beginInterval(&state)==0);
  assert(SharedGasTrialPolicy::begin(&state)==0);
  state.rho[0]=1.1;
  deviceCopyFailures=1;
  if(interval)SharedGasTrialPolicy::rollbackInterval(&state);
  else SharedGasTrialPolicy::rollback(&state);
  assert(state.gasModelPoisoned && !SharedGasTrialPolicy::retryable(&state));
  assert(SharedGasTrialPolicy::beginInterval(&state)!=0);
  assert(SharedGasTrialPolicy::begin(&state)!=0);
 }
 releaseSharedGasTrialStorage(state.gasTrial);
 assert(allocations==0);
}
'''.replace('__DECLARATIONS__',declarations).replace('__ASSIGNMENTS__',assignments)
    source=tmp_path/"probe.cpp";source.write_text(code);binary=tmp_path/"probe"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr

def test_native_mixture_calls_common_requested_interval_and_commits_clock_afterward():
    backend=(ROOT/"applications/gasUGKP/private_backend/GpuResidentStrict.cu").read_text()
    front=(ROOT/"applications/gasUGKP/diluteUgkwpFoam.C").read_text()
    assert 'advanceGasRequestedInterval<SharedGasTrialPolicy>' in backend
    assert '#include "SharedGasTrialPolicy.cuh"' in backend
    mixture=front[front.index('if (sharedGasModel.active())'):]
    assert mixture.index('resident.advanceOneStep') < mixture.index('runTime++;')
    assert 'resident.advanceOneStep(runTime.deltaTValue(), runTime.value());' in mixture
