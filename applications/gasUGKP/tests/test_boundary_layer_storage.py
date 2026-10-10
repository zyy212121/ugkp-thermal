"""Execute native wall adapter allocation/validation with host-memory CUDA shim."""
from pathlib import Path
import subprocess
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT


def test_boundary_layer_native_storage_is_atomic_and_scratch_bounded(tmp_path):
    pre=ADAPTER_PROBE.split('int main(){',1)[0]
    pre=pre.replace('#include <cassert>', '#include "common/gasTransport/GasBoundaryLayerModelState.H"\n#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"\n#include <cassert>')
    pre=pre.replace(' double* rho=nullptr;', ''' double* rho=nullptr;
 ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
 ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
 ugkwp::GasSstAuditState<double> gasSstAudit;
 SharedGasTrialStorage gasTrial;
 double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
 double gasBoundaryLayerStageTime=0;
 int *faceOwner=nullptr,*faceNeighbour=nullptr,*gasBoundaryKind=nullptr;
 double *V=nullptr,*magSf=nullptr;
''')
    body=pre+r'''
static int deviceQueries=0;
const int cudaDevAttrMultiProcessorCount=1;
int cudaGetDevice(int*p){*p=0;return 0;}
int cudaDeviceGetAttribute(int*p,int,int){++deviceQueries;*p=80;return 0;}
int cudaMemGetInfo(std::size_t*f,std::size_t*t){*f=*t=std::size_t(8)<<30;return 0;}
int syncDeviceState(DeviceState*s,const char*){return cudaMemcpy(s->deviceState,s,sizeof(*s),cudaMemcpyHostToDevice);}
#include "applications/gasUGKP/private_backend/BoundaryLayerStorage.cuh"
#include "applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh"
int main(){DeviceState state,device;state.deviceState=&device;
int owner[]={0,1},neighbour[]={-1,-1},kind[]={2,0};double volume[]={1,1},area[]={1,1},diffusion[]={.1,.1};
state.faceOwner=owner;state.faceNeighbour=neighbour;state.gasBoundaryKind=kind;state.V=volume;state.magSf=area;
state.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;state.gasSpecies.diffusivity=diffusion;state.gasSpeciesUploaded=true;
ugkwpGpuIpc::BoundaryLayerConfigV1 a;a.model=0;a.speciesCount=2;a.wallCount=1;a.quadratureCount=1;a.matchingCount=1;a.workspaceSlots=64;
int faces[]={0},offset[]={0,1},match[]={1};double geometry[]={1,0,0,.5,1.5,1,.5},qd[]={.5},qw[]={1},mw[]={1},flux[]={.2,.1};
auto configure=[&](){return ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&state,&a,faces,offset,offset,match,geometry,qd,qw,mw,flux);};
double oldSnapshot[1]{};
for(int level=0;level<2;++level){auto& snapshot=level?state.gasTrial.interval:state.gasTrial.trial;snapshot.cells=oldSnapshot;
assert(configure()!=0);assert(!state.gasBoundaryLayer.enabled&&allocations==0&&snapshot.cells==oldSnapshot);snapshot.cells=nullptr;}
geometry[5]=2;assert(configure()!=0);assert(!state.gasBoundaryLayer.enabled&&allocations==0);geometry[5]=1;
failAfter=5;assert(configure()!=0);assert(!state.gasBoundaryLayer.enabled&&allocations==0);failAfter=-1;
assert(configure()==0);assert(state.gasBoundaryLayer.enabled&&device.gasBoundaryLayer.enabled);
assert(state.gasBoundaryLayerModel.workspaceCount==0&&!state.gasBoundaryLayerModel.workspace&&deviceQueries==0);assert(state.gasBoundaryLayer.faceSlot[0]==0&&state.gasBoundaryLayer.faceSlot[1]==-1);
assert(state.gasBoundaryLayer.ownerSlot[0]==0&&state.gasBoundaryLayer.ownerSlot[1]==-1);
assert(state.gasBoundaryLayer.exchange[0].massReady&&!state.gasBoundaryLayer.exchange[0].ready);
assert(std::abs(state.gasBoundaryLayer.exchange[0].mass+.3)<1e-14);assert(!state.gasSstAudit.enabled);
const int live=allocations;assert(configure()!=0&&allocations==live);
releaseBoundaryLayerStorage(state);assert(allocations==0);
a.model=1;a.nodes=96;a.workspaceSlots=0;state.hostTurbulenceModel=3;assert(configure()==0);
assert(deviceQueries==1&&state.gasBoundaryLayerModel.workspaceCount==1&&state.gasBoundaryLayerModel.workspaceCapacity==128&&state.gasBoundaryLayerModel.workspace);
const int unauditedAllocations=allocations;assert(!state.gasSstAudit.enabled&&!state.gasSstAudit.sourceK);
releaseBoundaryLayerStorage(state);assert(allocations==0);
a.budgetAudit=1;assert(configure()==0);assert(state.gasSstAudit.enabled&&state.gasSstAudit.sourceK&&allocations==unauditedAllocations+13);
state.gasBoundaryLayer.exchange[0].ready=true;state.gasSstAudit.transportK[0]=7;state.gasSstAudit.sourceOmega[0]=11;
double diagnostics[ugkwpGpuIpc::boundaryLayerDiagnosticScalars+4];
assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&state,1,2,diagnostics)==0);
assert(diagnostics[16]==1&&diagnostics[17]==0&&diagnostics[18]==0&&diagnostics[19]==7&&diagnostics[22]==11);
releaseBoundaryLayerStorage(state);assert(allocations==0);
a.budgetAudit=0;assert(configure()==0);state.gasBoundaryLayer.exchange[0].ready=true;
assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&state,1,2,diagnostics)==0);
assert(diagnostics[16]==0);for(int j=17;j<25;++j)assert(std::isnan(diagnostics[j]));
releaseBoundaryLayerStorage(state);assert(allocations==0);
}
'''
    source=tmp_path/'probe.cpp';exe=tmp_path/'probe';source.write_text(body)
    build=subprocess.run(['g++','-std=c++17','-O0','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
