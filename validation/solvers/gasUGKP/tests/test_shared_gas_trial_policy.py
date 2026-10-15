"""Real native ownership policy with host-memory CUDA-copy shim, no GPU claim."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT=SOURCE_ROOT

@pytest.mark.parametrize("profiles",[False,True])
def test_trial_and_requested_interval_rollback_restore_accepted_state(tmp_path,profiles):
    scalar_fields="rho rhoUx rhoUy rhoUz rhoE Ux Uy Uz p Tgas gasFluxPositivityScale gasDiffusionNumber sstSourceNumber rhoK rhoOmega k omega nut gasBoundaryRho gasBoundaryUx gasBoundaryUy gasBoundaryUz gasBoundaryP gasBoundaryT V".split()
    declarations="\n".join("double* "+name+"=nullptr;" for name in scalar_fields)
    assignments="\n".join(f"double {name}[2]={{1,1}};state.{name}={name};" for name in scalar_fields)
    code=r'''
#include "common/gasTransport/GasBuildConfig.H"
#include "common/gasTransport/GasStateView.H"
#include "common/gasTransport/GasCapabilities.H"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
using cudaError_t=int;
const int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToDevice=2;
int allocations=0,deviceCopyFailures=0,hostCopyFailAfter=-1;
int cudaFree(void*p){if(p){--allocations;std::free(p);}return 0;}
int cudaMemcpy(void*d,const void*s,std::size_t n,int kind){if(kind==cudaMemcpyDeviceToDevice && deviceCopyFailures>0){--deviceCopyFailures;return 1;}if(n)std::memcpy(d,s,n);return 0;}
int cudaMemset(void*d,int v,std::size_t n){if(n)std::memset(d,v,n);return 0;}
int cudaDeviceSynchronize(){return 0;}
std::string lastError;void setLastErrorText(const char*p){lastError=p;}
void setLastError(const char*,int){}
template<class T>int allocate(T*&p,std::size_t n,const char*){p=n?static_cast<T*>(std::malloc(n*sizeof(T))):nullptr;if(p)++allocations;return 0;}
template<class T>int copyToDevice(T*d,const T*s,std::size_t n,const char*){if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
template<class T>int copyToHost(T*d,const T*s,std::size_t n,const char*){if(hostCopyFailAfter==0){hostCopyFailAfter=-1;return 1;}if(hostCopyFailAfter>0)--hostCopyFailAfter;if(n)std::memcpy(d,s,n*sizeof(T));return 0;}
#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"
struct DeviceState;
namespace sharedGasTrialDetail {
template<bool Capture,class State> int snapshotWallRecords(State*,SharedGasTrialSnapshot&,bool);
}
struct DeviceState{
 int (*gasBoundaryLayerSnapshot)(DeviceState*,SharedGasTrialSnapshot&,bool)=
     &sharedGasTrialDetail::snapshotWallRecords<false,DeviceState>;
 ugkwp::GasSpeciesState<double,2> gasSpecies;
 SharedGasTrialStorage gasTrial;
 ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
 ugkwp::GasBoundaryLayerModelState<double,2> gasBoundaryLayerModel;
 double gasBoundaryLayerStageTime=0;
 DeviceState* deviceState=nullptr;
 int nCells=2,nFaces=2,fixedCellBlockThreads=128,hostTurbulenceModel=0;
 bool gasModelPoisoned=false;
 __DECLARATIONS__
};
int applyGasGravitySource(DeviceState*,int,int,double,bool=false){return 0;}
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
 ugkwp::GasBoundaryLayerExchange<double> wallExchange[1];ugkwp::GasBoundaryLayerSstClosure<double> wallClosure[1];
 int wallStatus[1]{},wallFace[1]{1};double wallSpecies[2]{};
 ugkwp::gaswall::WallInput<double,2> wallInput[1];ugkwp::gaswall::WallOutput<double,2> wallOutput[1];ugkwp::gaswall::WallStatus coreStatus[1];
 state.gasBoundaryLayer.enabled=true;state.gasBoundaryLayer.count=1;state.gasBoundaryLayer.exchange=wallExchange;
 state.gasBoundaryLayer.sst=wallClosure;state.gasBoundaryLayer.status=wallStatus;state.gasBoundaryLayer.speciesFlux=wallSpecies;
 state.gasBoundaryLayerModel.input=wallInput;state.gasBoundaryLayerModel.output=wallOutput;state.gasBoundaryLayerModel.status=coreStatus;state.gasBoundaryLayerModel.faces=wallFace;
 double profile[24*9];for(int i=0;i<24*9;++i)profile[i]=i+.25;
 if(__PROFILES__){state.gasBoundaryLayerModel.profile=profile;
   state.gasBoundaryLayerSnapshot=&sharedGasTrialDetail::snapshotWallRecords<true,DeviceState>;}
 else assert(!state.gasBoundaryLayerModel.profile);
 state.gasBoundaryLayer.preparedTime=2;state.gasBoundaryLayer.preparedInterval=.25;
 assert(SharedGasTrialPolicy::beginInterval(&state)==0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.rho[0]=1.1;partial[0]=.5;state.gasBoundaryP[0]=2;
 cellStatus[0]=int(ugkwp::GasTransportCode::NegativeInventory);
 faceStatus[0]=int(ugkwp::GasTransportCode::SourceStepLimit);
 assert(SharedGasTrialPolicy::validate(&state)==0);
 assert(SharedGasTrialPolicy::commit(&state)==0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.rho[0]=9;partial[0]=-1;state.gasBoundaryP[0]=8;
 if(__PROFILES__)for(double& v:profile)v=-7;
 wallOutput[0].conductiveHeatFlux=123;wallInput[0].temperature=555;wallExchange[0].ready=true;state.gasBoundaryLayerStageTime=9;state.gasBoundaryLayer.preparedTime=9;state.gasBoundaryLayer.preparedInterval=.01;
 cellStatus[0]=int(ugkwp::GasTransportCode::InvalidStorage);
 assert(SharedGasTrialPolicy::validate(&state)!=0);
 assert(!SharedGasTrialPolicy::retryable(&state));
 SharedGasTrialPolicy::rollback(&state);
 assert(state.rho[0]==1.1 && partial[0]==.5 && state.gasBoundaryP[0]==2);
 assert(wallOutput[0].conductiveHeatFlux==0&&wallInput[0].temperature==0&&!wallExchange[0].ready&&state.gasBoundaryLayerStageTime==0);
 if(__PROFILES__)for(int i=0;i<24*9;++i)assert(profile[i]==i+.25);
 const std::size_t baseBytes=sizeof(wallExchange)+sizeof(wallClosure)+sizeof(wallStatus)
   +sizeof(wallSpecies)+sizeof(wallInput)+sizeof(wallOutput)+sizeof(coreStatus);
 assert(state.gasTrial.trial.wallRecordBytes==baseBytes+(__PROFILES__?sizeof(profile):0));
 assert(!state.gasBoundaryLayer.preparedFirstStage);
 assert(state.gasBoundaryLayer.preparedTime==2&&state.gasBoundaryLayer.preparedInterval==.25);
 if(__PROFILES__)profile[0]=-99;
 SharedGasTrialPolicy::rollbackInterval(&state);
 if(__PROFILES__)assert(profile[0]==.25);
 assert(state.rho[0]==1 && partial[0]==.4 && state.gasBoundaryP[0]==1);
 assert(cellStatus[0]==0 && faceStatus[0]==0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.gasFluxPositivityScale[0]=2;state.gasDiffusionNumber[0]=1e300;state.sstSourceNumber[0]=NAN;
 assert(SharedGasTrialPolicy::validateTimeStep(&state,.1)!=0);
 assert(SharedGasTrialPolicy::retryable(&state));
 cellStatus[0]=int(ugkwp::GasTransportCode::ChemistryFailure);
 assert(SharedGasTrialPolicy::validate(&state)==0);
 cellStatus[0]=int(ugkwp::GasTransportCode::InvalidStorage);
 assert(SharedGasTrialPolicy::validate(&state)!=0);
 assert(SharedGasTrialPolicy::begin(&state)==0);
 faceStatus[0]=wallStatus[0]=int(ugkwp::GasTransportCode::BoundaryLayerFailure);
 coreStatus[0].code=ugkwp::gaswall::WallCode::NonConvergence;coreStatus[0].iteration=17;coreStatus[0].residual=.125;
 assert(sharedGasTrialDetail::transportStatus(&state,true)!=0&&!SharedGasTrialPolicy::retryable(&state));
 assert(lastError.find("boundaryLayer")!=std::string::npos&&lastError.find("face=1")!=std::string::npos);
 assert(SharedGasTrialPolicy::continueTerminal(&state));
 for(int failedRead=0;failedRead<5;++failedRead){hostCopyFailAfter=failedRead;
  assert(SharedGasTrialPolicy::validate(&state,true)!=0&&!SharedGasTrialPolicy::continueTerminal(&state));}
 hostCopyFailAfter=-1;assert(SharedGasTrialPolicy::validate(&state,true)!=0&&SharedGasTrialPolicy::continueTerminal(&state));
 wallStatus[0]=0;faceStatus[0]=0;
 // Reconfigure optional chemistry storage between same-sized intervals.
 ugkwp::ChemistryStatus chemicalStatus[2];
 state.gasSpecies.chemistryStatus=chemicalStatus;
 assert(SharedGasTrialPolicy::begin(&state)==0);
 assert(state.gasTrial.trial.chemistryStatus);
 chemicalStatus[0].code=ugkwp::ChemistryCode::NewtonFailure;
 SharedGasTrialPolicy::rollback(&state);
 assert(chemicalStatus[0].code==ugkwp::ChemistryCode::Success);
 state.gasSpecies.chemistryStatus=nullptr;
 assert(SharedGasTrialPolicy::begin(&state)==0);
 state.rho[0]=99;deviceCopyFailures=1;
 SharedGasTrialPolicy::rollback(&state);
 assert(state.gasModelPoisoned && !SharedGasTrialPolicy::retryable(&state));
 assert(SharedGasTrialPolicy::beginInterval(&state)!=0);
 assert(SharedGasTrialPolicy::begin(&state)!=0);
 releaseSharedGasTrialStorage(state.gasTrial);
 assert(allocations==0);
}
'''.replace('__DECLARATIONS__',declarations).replace('__ASSIGNMENTS__',assignments).replace('__PROFILES__','true' if profiles else 'false')
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


def test_real_server_error_response_keeps_restored_handle_downloadable(tmp_path):
    import re
    own=(VALIDATION_ROOT / 'solvers/gasUGKP/tests/test_shared_gas_trial_policy.py').read_text()
    pre=re.search("code=r'''(.*?)'''",own,re.S).group(1).split('int main(){')[0]
    gas='rho rhoUx rhoUy rhoUz rhoE Ux Uy Uz p Tgas'.split()
    solids='epsS rhoUsx rhoUsy rhoUsz rhoEs rhoDs rhoHp Usx Usy Usz theta Tp dMeanCell'.split()
    other='gasFluxPositivityScale gasDiffusionNumber sstSourceNumber rhoK rhoOmega k omega nut gasBoundaryRho gasBoundaryUx gasBoundaryUy gasBoundaryUz gasBoundaryP gasBoundaryT V'.split()
    fields=gas+solids+other
    pre=pre.replace('__DECLARATIONS__','\n'.join('double* '+x+'=nullptr;' for x in fields))
    pre+='\nDeviceState*asState(void*p){return static_cast<DeviceState*>(p);}\nint validateState(DeviceState*s,const char*){return s->gasModelPoisoned?1:0;}\n'
    native=(ROOT/'applications/gasUGKP/private_backend/GpuResidentStrict.cu').read_text()
    download=native[native.index('extern "C" int ugkwpGpuResidentStrictDownloadFields\n'):native.index('extern "C" int ugkwpGpuResidentStrictDownloadEpsGPrev\n')]
    server=(ROOT/'applications/gasUGKP/private_backend/GpuBackendServer.C').read_text()
    handlers=server[server.index('bool handleGasOnly('):server.index('bool handleDownloadGasBoundary\n')]
    code=pre+download+r'''
struct AdvanceArgs{double dt=.1,simulationTime=0;};struct RequestHeader{};struct ServerState{void*backend;int nCells=2;};
int responseStatus=0;std::vector<std::vector<double>>published;
int wireBytes(const AdvanceArgs&){return 16;}bool validatePayload(int,const RequestHeader&,std::size_t){return true;}
bool readObject(int,AdvanceArgs&){return true;}bool responseHeader(int,int status,std::uint64_t){responseStatus=status;return true;}
double*ptr(std::vector<double>&v){return v.data();}bool writeVector(int,const std::vector<double>&v){published.push_back(v);return true;}
int ugkwpGpuResidentStrictAdvanceGasOnly(void*handle,double,double){auto*s=asState(handle);
 if(SharedGasTrialPolicy::begin(s))return 1;
 for(auto field:sharedGasTrialDetail::cellFields(s))if(field.pointer)for(std::size_t i=0;i<field.count;++i)field.pointer[i]=-777;
 s->gasSpecies.cellStatus[0]=int(ugkwp::GasTransportCode::InvalidStorage);
 const int error=SharedGasTrialPolicy::validate(s);assert(error);SharedGasTrialPolicy::rollback(s);return error;
}
'''+handlers+'\nint main(){DeviceState state;state.deviceState=&state;\n'
    for i,name in enumerate(fields):code+=f'double {name}[2]={{{i+1},{i+2}}};state.{name}={name};\n'
    code+=r'''
 double species[4]={.4,.5,.6,.5},sound[2]={300,301},cp[2]={1000,1001},R[2]={287,288};int cells[2]{},faces[2]{};
 state.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;state.gasSpecies.rho=species;state.gasSpecies.soundSpeed=sound;state.gasSpecies.heatCapacity=cp;state.gasSpecies.gasConstant=R;state.gasSpecies.cellStatus=cells;state.gasSpecies.faceStatus=faces;
 ServerState server{&state};RequestHeader request;
 assert(handleGasOnly(0,request,server)&&responseStatus!=0&&server.backend==&state&&!state.gasModelPoisoned);
 assert(handleDownloadFields(0,request,server)&&responseStatus==0&&published.size()==23);
 for(int field=0;field<23;++field)for(int c=0;c<2;++c)assert(published[field][c]==field+1+c);
 assert(species[0]==.4&&species[1]==.5&&species[2]==.6&&species[3]==.5&&sound[0]==300&&cp[1]==1001&&R[1]==288);
 assert(SharedGasTrialPolicy::begin(&state)==0&&SharedGasTrialPolicy::validate(&state)==0);
 releaseSharedGasTrialStorage(state.gasTrial);assert(allocations==0);
}
'''
    source=tmp_path/'error-download.cpp';source.write_text(code);binary=tmp_path/'error-download'
    b=subprocess.run(['g++','-std=c++17','-I'+str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True);assert b.returncode==0,b.stderr
    r=subprocess.run([str(binary)],capture_output=True,text=True);assert r.returncode==0,r.stderr
