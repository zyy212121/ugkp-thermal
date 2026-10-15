"""Actual native adapter with counted host CUDA plumbing and injected failures."""
import subprocess
from test_shared_gas_adapter import ADAPTER_PROBE, ROOT
from test_boundary_layer_storage import snapshot_release_probe


def adapter_probe_source():
    pre=ADAPTER_PROBE.split('int main(){',1)[0]
    pre=pre.replace('#include <cassert>', '#include "common/gasTransport/GasBoundaryLayerModelState.H"\n#include "applications/gasUGKP/private_backend/SharedGasTrialFields.H"\n#include <cassert>')
    pre=pre.replace(' double* rho=nullptr;', ''' double* rho=nullptr;
 ugkwp::GasBoundaryLayerState<double> gasBoundaryLayer;
 ugkwp::GasBoundaryLayerModelState<double,ugkwp::compiledGasSpecies> gasBoundaryLayerModel;
 SharedGasTrialStorage gasTrial;
 double *gasWallQuadratureDistance=nullptr,*gasWallQuadratureWeight=nullptr;
 double gasBoundaryLayerStageTime=0;
 int *faceOwner=nullptr,*faceNeighbour=nullptr,*gasBoundaryKind=nullptr;
 double *V=nullptr,*magSf=nullptr;
''')
    pre=pre.replace('static int allocations=0,','static int reads=0;static int allocations=0,')
    pre=pre.replace('int copyToHost(T*d,const T*s,std::size_t n,const char*){','int copyToHost(T*d,const T*s,std::size_t n,const char*){++reads;')
    code=pre+snapshot_release_probe()+r'''
static int queryCalls=0,queryFailure=0;
const int cudaDevAttrMultiProcessorCount=1;
int cudaGetDevice(int*p){++queryCalls;*p=0;return queryFailure==1;}
int cudaDeviceGetAttribute(int*p,int,int){++queryCalls;*p=80;return queryFailure==2;}
int cudaMemGetInfo(std::size_t*f,std::size_t*t){++queryCalls;*f=*t=0;return queryFailure==3;}
#include "applications/gasUGKP/private_backend/BoundaryLayerStorage.cuh"
#include "applications/gasUGKP/private_backend/BoundaryLayerAdapter.cuh"
int main(){DeviceState s,device;s.deviceState=&device;
int owner[]={0,1},neighbour[]={1,-1},kind[]={0,0};double volume[]={99,99},area[]={2,1},diffusion[]={.1,.1};
s.faceOwner=owner;s.faceNeighbour=neighbour;s.gasBoundaryKind=kind;s.V=volume;s.magSf=area;
s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.diffusivity=diffusion;s.gasSpeciesUploaded=true;
ugkwpGpuIpc::BoundaryLayerConfigV1 a;a.model=1;a.speciesCount=3;a.version=99;a.wallCount=1;a.quadratureCount=1;a.matchingCount=33;
a.nodes=2;a.workspaceSlots=5000;a.relativeTolerance=-1;a.absoluteTolerance=-2;a.stretch=.5;a.maxIterations=0;
int faces[]={0},qo[]={0,1},mo[]={0,33},mc[33];double mw[33];for(int i=0;i<33;++i){mc[i]=1;mw[i]=.1;}
double g[]={2,0,0,.5,1.5,1,std::numeric_limits<double>::quiet_NaN()},qd[]={.7},qw[]={2},flux[]={-.2,.1,999};
auto configure=[&](){return ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&s,&a,faces,qo,mo,mc,g,qd,qw,mw,flux);};
assert(configure()==0);assert(reads==2);assert(queryCalls==3);
assert(s.gasBoundaryLayerModel.workspaceCount==1&&s.gasBoundaryLayerModel.workspaceCapacity==24&&s.gasBoundaryLayerModel.workspace);
assert(s.gasBoundaryLayerModel.matchingOffsets[1]==33&&s.gasBoundaryLayerModel.matchingWeights[32]==.1);
assert(s.gasBoundaryLayerModel.input[0].massFlux[0]==-.2&&s.gasBoundaryLayerModel.input[0].massFlux[1]==.1);
assert(std::isnan(s.gasBoundaryLayerModel.input[0].quadrature.firstMoment));
assert(std::abs(s.gasBoundaryLayer.exchange[0].mass-.2)<1e-14);
const int live=allocations;auto*old=s.gasBoundaryLayerModel.input;
// A failed real allocation/copy preserves the existing fully published view.
failAfter=2;assert(configure()!=0&&allocations==live&&s.gasBoundaryLayerModel.input==old);failAfter=-1;
for(int failure:{0,2,6,7,8,9}){copyFailAfter=failure;assert(configure()!=0&&allocations==live&&s.gasBoundaryLayerModel.input==old);}
for(int failure:{0,1}){hostCopyFailAfter=failure;assert(configure()!=0&&allocations==live&&s.gasBoundaryLayerModel.input==old);}
for(int failure=1;failure<=3;++failure){queryFailure=failure;assert(configure()!=0&&allocations==live&&s.gasBoundaryLayerModel.input==old);}queryFailure=0;
// An out-of-backing face index is intentionally not executed: it now reaches
// the supplied addressing consumer without a metadata admission check.
a.nodes=129;assert(configure()!=0);a.nodes=2;
a.speciesCount=1;assert(configure()==0&&s.gasBoundaryLayerModel.input[0].massFlux[1]==.1);a.speciesCount=3;
a.model=99;assert(configure()!=0);a.model=1;
// Matching indices are uploaded verbatim here; no matching-cell consumer runs.
mc[32]=99;assert(configure()==0&&s.gasBoundaryLayerModel.matchingCells[32]==99);mc[32]=1;
flux[0]=std::numeric_limits<double>::quiet_NaN();assert(configure()==0&&std::isnan(s.gasBoundaryLayerModel.input[0].massFlux[0]));flux[0]=-.2;
// Diagnostics publish actual requested species; missing consumed storage fails.
s.gasBoundaryLayer.exchange[0].ready=true;
s.gasBoundaryLayerModel.output[0].wallSpeciesFlux[0]=.7;s.gasBoundaryLayerModel.output[0].reactionIntegral[0]=.8;
s.gasBoundaryLayerModel.output[0].integratedKSource=std::numeric_limits<double>::infinity();
double diagnostic[31];assert(!ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,1,1,diagnostic));
assert(diagnostic[25]==.7&&diagnostic[26]==.8&&std::isinf(diagnostic[9]));
auto* output=s.gasBoundaryLayerModel.output;s.gasBoundaryLayerModel.output=nullptr;
assert(ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,1,1,diagnostic));s.gasBoundaryLayerModel.output=output;
// Reconfiguration replaces actual arrays and invalidates obsolete snapshots.
for(auto*p:{&s.gasTrial.trial}){assert(!allocateSharedGasZero(p->cells,2));assert(!allocateSharedGasZero(p->wallRecords,3));p->valid=true;}
old=s.gasBoundaryLayerModel.input;g[3]=.75;g[4]=2.5;g[5]=7;
assert(configure()==0&&allocations==live);assert(s.gasBoundaryLayerModel.input!=old);
assert(device.gasBoundaryLayerModel.input==s.gasBoundaryLayerModel.input&&device.gasBoundaryLayerModel.input[0].matchingDistance==2.5);
for(auto*p:{&s.gasTrial.trial})assert(!p->cells&&!p->wallRecords&&!p->valid);
// An empty wall request releases all wall-owned resources.
a.wallCount=a.quadratureCount=a.matchingCount=0;int empty[]={0};
assert(!ugkwpGpuResidentStrictConfigureBoundaryLayerV1(&s,&a,nullptr,empty,empty,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr));
assert(!s.gasBoundaryLayer.enabled&&!device.gasBoundaryLayer.enabled&&allocations==0);
assert(!ugkwpGpuResidentStrictDownloadBoundaryLayerV1(&s,0,2,nullptr));
releaseBoundaryLayerStorage(s);assert(!allocations);
}
'''
    return code


def test_actual_adapter_replaces_storage_without_geometry_admission(tmp_path):
    code=adapter_probe_source()
    src=tmp_path/'adapter.cpp';src.write_text(code);exe=tmp_path/'adapter'
    built=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    ran=subprocess.run([str(exe)],capture_output=True,text=True)
    assert ran.returncode==0,ran.stdout+ran.stderr


import pytest


@pytest.fixture(scope='module')
def host_allocation_probe(tmp_path_factory):
    code=adapter_probe_source()
    code=code.replace('#include <cassert>',r'''
#include <cassert>
#include <new>
static int hostFailure=-1,hostCalls=0;
void* operator new(std::size_t size){
 if(hostFailure>=0){++hostCalls;if(hostFailure--==0){hostFailure=-1;throw std::bad_alloc();}}
 if(void*p=std::malloc(size?size:1))return p;throw std::bad_alloc();
}
void operator delete(void*p) noexcept {std::free(p);}
void operator delete(void*p,std::size_t) noexcept {std::free(p);}
''',1)
    prefix,main=code.split('int main(){',1)
    main=main[:main.index('assert(configure()==0);')]
    code=prefix+'int main(int argc,char**argv){'+main+r'''
assert(argc==3);const int failure=std::atoi(argv[1]);const bool configured=std::atoi(argv[2]);
if(configured){
 assert(configure()==0);
 for(auto*p:{&s.gasTrial.trial}){
  assert(!allocateSharedGasZero(p->cells,2));assert(!allocateSharedGasZero(p->wallRecords,3));
  p->cells[0]=123;p->wallRecords[0]=77;p->wallRecordBytes=3;p->valid=true;
 }
}
const auto old=s;const auto oldDevice=device;const int live=allocations;
g[3]=.75;g[4]=2.5;g[5]=7;
hostFailure=failure;hostCalls=0;int rc=0;bool escaped=false;
try{rc=configure();}catch(const std::bad_alloc&){escaped=true;}
std::fprintf(stderr,"configured=%d host_fault=%d calls=%d escaped=%d live_before=%d live_after=%d\n",
 int(configured),failure,hostCalls,int(escaped),live,allocations);
if(escaped){releaseBoundaryLayerStorage(s);releaseSharedGasTrialStorage(s.gasTrial);
 std::fprintf(stderr,"after resident cleanup=%d\n",allocations);return 2;}
assert(rc!=0&&last.find("host allocation")!=std::string::npos);
assert(hostCalls==failure+1&&allocations==live);
assert(std::memcmp(&s,&old,sizeof(s))==0&&std::memcmp(&device,&oldDevice,sizeof(device))==0);
if(configured){
 assert(s.gasBoundaryLayerModel.input[0].ownerDistance==.5);
 for(auto*p:{&s.gasTrial.trial})assert(p->valid&&p->cells[0]==123&&p->wallRecords[0]==77&&p->wallRecordBytes==3);
}
// The same request remains executable once the genuine allocation failure is gone.
assert(configure()==0&&allocations==16);
assert(s.gasBoundaryLayerModel.input[0].ownerDistance==.75&&device.gasBoundaryLayerModel.input==s.gasBoundaryLayerModel.input);
for(auto*p:{&s.gasTrial.trial})assert(!p->cells&&!p->wallRecords&&!p->valid);
releaseBoundaryLayerStorage(s);releaseSharedGasTrialStorage(s.gasTrial);assert(allocations==0);
}
'''
    path=tmp_path_factory.mktemp('wall_host_allocation');src=path/'allocation.cpp';binary=path/'allocation'
    src.write_text(code)
    built=subprocess.run(['g++','-std=c++17','-O1','-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(binary)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    return binary


@pytest.mark.parametrize('configured',[False,True])
@pytest.mark.parametrize('failed_allocation',range(6),ids=['owners','areas','face_slots','owner_slots','inputs','exchanges'])
def test_real_host_allocation_failure_preserves_native_ownership(host_allocation_probe,configured,failed_allocation):
    run=subprocess.run([str(host_allocation_probe),str(failed_allocation),str(int(configured))],capture_output=True,text=True)
    print(run.stderr)
    assert run.returncode==0,run.stdout+run.stderr
