"""Execute native ledger owners with recorded CUDA boundaries, not GPU claims."""
from pathlib import Path
import importlib.util
import re
import subprocess
import json
ROOT=Path(__file__).resolve().parents[1]
BASE='a9af8ecbdfe94baf61ac6b5d9fa75a4e23b0c92b'
spec=importlib.util.spec_from_file_location('native_helpers',ROOT/'tests/test_native_cleanup.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def source(path, revision=None):
    return subprocess.check_output(['git','show',revision+':'+path],cwd=ROOT,text=True) if revision else (ROOT/path).read_text()


def native(revision=None, restore_stub=True):
    header=source('applications/CHT/gpu/GpuWallEnergy64.H',revision).replace('#include <cuda_runtime.h>','')
    body=source('applications/CHT/gpu/GpuWallEnergy64.cu',revision).replace('#include "GpuWallEnergy64.H"','')
    body=re.sub(r'(\w+)\s*<<<.*?>>>',r'(++launches, \1)',body,flags=re.S)
    stub = '\nextern \"C\" int ugkpPrepareParticleWallEnergyRestore(void*){assert(false);return 1;}\n' if restore_stub and 'ugkpPrepareParticleWallEnergyRestore' in header else ''
    return h.CUDA+header+'\nint invalidations=0;int ugkpInvalidateGasWallEnergyGraph(void*){++invalidations;return 0;}\n'+body+stub


def test_repeat_selection_preserves_energy_and_empty_route(tmp_path):
    h.run_cpp(tmp_path,native()+r'''
int main(){GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=1;
 assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,0,nullptr)==0);assert(allocations==0&&copies==0);
 int a[]={1,3};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,a)==0);
 double* energy=host.gasWallEnergy;energy[0]=-0.;energy[1]=17;energy[2]=19;energy[3]=23;energy[4]=std::numeric_limits<double>::quiet_NaN();double saved[5];std::memcpy(saved,energy,sizeof saved);
 int b[]={2};counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,b)==0);
 assert(host.gasWallEnergy==energy&&std::memcmp(saved,energy,sizeof saved)==0&&std::signbit(energy[0]));
 assert(host.nEnabledFaces==1&&device.nEnabledFaces==1&&host.enabledFaceIds[0]==2&&host.gasWallEnergyMask[1]==0&&host.gasWallEnergyMask[2]==1);
 assert(memsets==0&&allocations==0&&h2d==3&&d2h==0);
 counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,0,nullptr)==0);assert(host.nEnabledFaces==0&&device.nEnabledFaces==0&&energy[3]==23&&h2d==2&&allocations==0&&memsets==0);
 int duplicate[]={0,2,2,3,4,4};counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,6,duplicate)==0);assert(host.nEnabledFaces==6&&device.enabledFaceIds==host.enabledFaceIds&&std::memcmp(saved,energy,sizeof saved)==0&&allocations==1&&memsets==0&&h2d==3);
 ugkpReleaseGasWallEnergy64(&host);
}
''')


def test_old_first_configuration_exact_bits_and_counts(tmp_path):
    main=r'''
#include <iostream>
#include <iomanip>
int main(){GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=1;
 assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,0,nullptr)==0);assert(allocations==0&&copies==0&&memsets==0);
 int faces[]={1,3};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,faces)==0);
 std::cout<<"META "<<sizeof(host)<<" "<<copiedBytes<<"\n";
 std::cout<<"COST "<<allocations<<" "<<h2d<<" "<<d2h<<" "<<memsets<<" "<<zeroedBytes<<"\n";
 std::cout<<"DATA "<<host.nEnabledFaces<<" "<<device.nEnabledFaces<<std::hexfloat;
 for(int i=0;i<5;++i)std::cout<<" "<<host.gasWallEnergy[i]<<" "<<int(host.gasWallEnergyMask[i]);
 for(int i=0;i<2;++i)std::cout<<" "<<host.enabledFaceIds[i];std::cout<<"\n";
 ugkpReleaseGasWallEnergy64(&host);}
'''
    results={}
    for name,revision in [('old',BASE),('current',None)]:
        d=tmp_path/name;d.mkdir();results[name]=h.run_cpp(d,native(revision)+main)
    assert [x for x in results['old'].splitlines() if not x.startswith('META')]==[x for x in results['current'].splitlines() if not x.startswith('META')]
    assert 'META 80 93' in results['old'] and 'META 96 109' in results['current']
    assert 'COST 4 3 0 1 40' in results['current']
    (tmp_path/'parity.json').write_text(json.dumps(results,indent=2))


def tracked_native():
    s=native()
    s=s.replace('#include <vector>','#include <vector>\n#include <map>\nstd::map<void*,size_t> live;int frees=0,failFree=0;',1)
    s=s.replace('*p=std::malloc(n);return *p?0:1;','*p=std::malloc(n);if(*p)live[*p]=n;return *p?0:1;')
    s=s.replace('int cudaFree(void* p){std::free(p);return 0;}', 'int cudaFree(void* p){if(!p)return 0;++frees;if(frees==failFree)return 1;assert(live.erase(p)==1);std::free(p);return 0;}')
    return s


def test_replacement_failure_keeps_owners_and_retry(tmp_path):
    h.run_cpp(tmp_path,tracked_native()+r'''
int main(){
 for(int copyFault=1;copyFault<=3;++copyFault)for(bool cleanupFault:{false,true}){
  GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=1;
  int a[]={1};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);auto* energy=host.gasWallEnergy;energy[1]=19;
  int b[]={2,3};counters();frees=0;failCopy=copyFault;failFree=cleanupFault?1:0;
  assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)!=0);assert(host.gasWallEnergy==energy&&energy[1]==19&&live.count(energy));
  // A failed publication keeps both potential device owners reachable.
  if(copyFault==3){assert(host.retiredFaceIds&&live.size()==5&&live.count(device.enabledFaceIds));}
  else assert(live.size()==size_t(cleanupFault?5:4));
  counters();frees=0;failFree=0;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)==0);
  assert(host.retiredFaceIds==nullptr&&live.size()==4&&device.enabledFaceIds==host.enabledFaceIds&&host.gasWallEnergy[1]==19);
  ugkpReleaseGasWallEnergy64(&host);assert(live.empty());
 }
 for(int fault=1;fault<=4;++fault){GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=1;int a[]={1};counters();failAlloc=fault;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)!=0);ugkpReleaseGasWallEnergy64(&host);assert(live.empty());}
 for(int fault=1;fault<=3;++fault){GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=1;int a[]={1};counters();failCopy=fault;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)!=0);counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);ugkpReleaseGasWallEnergy64(&host);assert(live.empty());}
}
''')


def graph_native():
    s=tracked_native().replace('int invalidations=0;int ugkpInvalidateGasWallEnergyGraph(void*){++invalidations;return 0;}','')
    s=s.replace('namespace\n{','void recordAccumulation(double*,const double*,const int*,int,double);\nnamespace\n{',1)
    s=s.replace('(++launches, accumulateGasWallEnergy64Kernel)','(++launches, recordAccumulation)')
    # The helpers below record the CUDA graph API; production capture/reuse and
    # invalidation function bodies execute verbatim.
    graph=r'''
struct Accum {double*energy;const double*flux;const int*faces;int count;double dt;};
struct Graph {std::vector<Accum> work;};using cudaGraph_t=Graph*;using cudaGraphExec_t=Graph*;using cudaGraphNode_t=int;
struct cudaKernelNodeParams {void*func=nullptr;void**kernelParams=nullptr;void**extra=nullptr;};
using cudaGraphNodeType=int;constexpr int cudaGraphNodeTypeKernel=1,cudaStreamCaptureModeThreadLocal=0;
struct DeviceState{GpuWallEnergyState wallEnergy;DeviceState*deviceState=this;cudaGraphExec_t gasGraphExec=nullptr;cudaGraph_t gasGraph=nullptr;cudaStream_t gasCaptureStream=nullptr;double gasGraphDt=-1;int nFaces=5;std::vector<int>gasGraphTimeNodes;std::vector<cudaKernelNodeParams>gasGraphTimeParams;};
struct GasHostPolicy{using Time=double;};template<class T>void updateLegacyGasBoundaryMirrorKernel(T*,double){}
Graph*capturing=nullptr;int captures=0,graphLaunches=0,destroyExecs=0,destroyGraphs=0,graphCall=0,failGraph=0;
int graphError(){return ++graphCall==failGraph?1:0;}
void setLastError(const char*,int){}void setLastErrorText(const char*){}
int cudaGraphExecDestroy(Graph*g){if(graphError())return 1;++destroyExecs;delete g;return 0;}
int cudaGraphDestroy(Graph*g){if(graphError())return 1;++destroyGraphs;delete g;return 0;}
int cudaStreamCreate(void**s){if(graphError())return 1;*s=reinterpret_cast<void*>(1);return 0;}
int cudaStreamDestroy(void*){return graphError();}
int cudaStreamBeginCapture(void*,int){if(graphError())return 1;++captures;capturing=new Graph;return 0;}
int cudaStreamEndCapture(void*,Graph**g){*g=capturing;capturing=nullptr;return graphError();}
int cudaGraphGetNodes(Graph*,int*n,size_t*c){if(graphError())return 1;if(n){n[0]=1;n[1]=2;}*c=2;return 0;}
int cudaGraphNodeGetType(int,int*t){*t=cudaGraphNodeTypeKernel;return graphError();}
int cudaGraphKernelNodeGetParams(int,cudaKernelNodeParams*p){p->func=reinterpret_cast<void*>(updateLegacyGasBoundaryMirrorKernel<DeviceState>);return graphError();}
int cudaGraphInstantiate(Graph**e,Graph*g,void*,void*,int){if(graphError())return 1;*e=new Graph(*g);return 0;}
int cudaGraphExecKernelNodeSetParams(Graph*,int,const cudaKernelNodeParams*){return graphError();}
void execute(const Accum&a){for(int i=0;i<a.count;++i){blockIdx.x=i;blockDim.x=1;threadIdx.x=0;accumulateGasWallEnergy64Kernel(a.energy,a.flux,a.faces,a.count,a.dt);}blockIdx.x=0;}
int cudaGraphLaunch(Graph*g,void*){if(graphError())return 1;++graphLaunches;for(const auto&a:g->work)execute(a);return 0;}
void recordAccumulation(double*e,const double*f,const int*ids,int n,double dt){Accum a{e,f,ids,n,dt};if(capturing)capturing->work.push_back(a);else execute(a);}
int advanceGasFluxStage(DeviceState*s,double dt,double){return ugkpAccumulateGasWallEnergy64(&s->wallEnergy,dt,s->gasCaptureStream);}
int finaliseGasBoundaryStage(DeviceState*,double,double){return 0;}
'''
    helper=h.block(source('applications/CHT/gpu/GpuResidentStrict.cu'),'extern "C" int ugkpInvalidateGasWallEnergyGraph')
    helper=helper.replace('DeviceState* s = asState(handle);','DeviceState* s = static_cast<DeviceState*>(handle);')
    advance=h.block(source('common/GpuGasAdvance.cuh'),'int advancePureGasGraph')
    return s+graph+helper+'\ntemplate<class GasHostState>\n'+advance


def test_actual_graph_recapture_uses_current_raw_selection(tmp_path):
    h.run_cpp(tmp_path,graph_native()+r'''
int main(){DeviceState host,device;host.deviceState=&device;host.wallEnergy.deviceState=&device.wallEnergy;char error[2048]{};host.wallEnergy.errorBuffer=error;host.wallEnergy.nFaces=5;host.wallEnergy.nInternalFaces=1;
 // A graph captured before the first nonempty configuration must also refresh.
 assert(advancePureGasGraph(&host,.25,0.)==0&&captures==1);
 int a[]={1};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);
 for(int i=0;i<5;++i){host.wallEnergy.gasWallEnergy[i]=10+i;host.wallEnergy.gasWallFlux[i]=4*(i+1);}
 assert(advancePureGasGraph(&host,.25,1.)==0&&captures==2);assert(host.wallEnergy.gasWallEnergy[1]==13);
 assert(advancePureGasGraph(&host,.25,2.)==0&&captures==2);assert(host.wallEnergy.gasWallEnergy[1]==15);
 int b[]={2,3};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)==0);assert(advancePureGasGraph(&host,.25,3.)==0&&captures==3);
 assert(host.wallEnergy.gasWallEnergy[1]==15&&host.wallEnergy.gasWallEnergy[2]==15&&host.wallEnergy.gasWallEnergy[3]==17);
 assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,0,nullptr)==0);assert(advancePureGasGraph(&host,.25,4.)==0&&captures==4&&host.wallEnergy.gasWallEnergy[2]==15);
 // Sequential shim checks captured list consumption only; actual duplicate
 // GPU entries retain the original non-atomic write race and may lose updates.
 int duplicate[]={2,2};assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,duplicate)==0);assert(advancePureGasGraph(&host,.25,5.)==0&&captures==5&&host.wallEnergy.gasWallEnergy[2]==21);
 assert(ugkpInvalidateGasWallEnergyGraph(&host)==0);ugkpReleaseGasWallEnergy64(&host.wallEnergy);assert(live.empty());
}
''')


def test_actual_graph_destroy_failure_precedes_mutation(tmp_path):
    h.run_cpp(tmp_path,graph_native()+r'''
int main(){for(int failed:{1,2}){DeviceState host,device;host.deviceState=&device;host.wallEnergy.deviceState=&device.wallEnergy;char error[2048]{};host.wallEnergy.errorBuffer=error;host.wallEnergy.nFaces=5;host.wallEnergy.nInternalFaces=1;
 int a[]={1};counters();failGraph=0;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);host.wallEnergy.gasWallEnergy[1]=17;host.wallEnergy.gasWallFlux[1]=0;
 assert(advancePureGasGraph(&host,.25,0.)==0);auto*faces=host.wallEnergy.enabledFaceIds;int b[]={2,3};counters();graphCall=0;failGraph=failed;
 assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)!=0);assert(host.wallEnergy.enabledFaceIds==faces&&host.wallEnergy.nEnabledFaces==1&&host.wallEnergy.gasWallEnergy[1]==17&&copies==0&&allocations==0);
 failGraph=0;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)==0);assert(ugkpInvalidateGasWallEnergyGraph(&host)==0);ugkpReleaseGasWallEnergy64(&host.wallEnergy);assert(live.empty());}}
''')


def test_retirement_and_initialisation_failures_are_real_errors(tmp_path):
    h.run_cpp(tmp_path,tracked_native()+r'''
int main(){GpuWallEnergyState host,device;char error[2048]{};host.deviceState=&device;host.errorBuffer=error;host.nFaces=5;host.nInternalFaces=99;int a[]={1};
 counters();failMemset=1;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)!=0);assert(live.empty());
 counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);host.gasWallEnergy[1]=17;
 int b[]={2,3};counters();failAlloc=1;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)!=0);assert(live.size()==4&&host.gasWallEnergy[1]==17);
 counters();frees=0;failFree=1;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)!=0);assert(host.retiredFaceIds&&live.size()==5&&device.enabledFaceIds==host.enabledFaceIds);
 counters();frees=0;failFree=1;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)!=0);assert(host.retiredFaceIds&&live.size()==5);
 counters();frees=0;failFree=0;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,2,b)==0);assert(live.size()==4&&host.gasWallEnergy[1]==17);
 counters();assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,nullptr)!=0&&copies==0);ugkpReleaseGasWallEnergy64(&host);assert(live.empty());}
''')


def test_actual_frontend_repeated_and_foreign_patch_selection(tmp_path):
    method=h.block(source('applications/CHT/gpu/GpuResidentStrict.H'),'    void configureGasWallEnergyLedger')
    h.run_cpp(tmp_path,r'''
#include <cassert>
#include <vector>
#include <stdexcept>
using label=int;
struct labelList:std::vector<int>{using std::vector<int>::vector;int*begin(){return data();}const int*begin()const{return data();}void transfer(labelList&other){swap(other);}};
struct polyPatch{int first,count;int size()const{return count;}int start()const{return first;}};
struct fvMesh{std::vector<polyPatch>patches;const auto&boundaryMesh()const{return patches;}int nFaces()const{return 9;}};
#define forAll(v,i) for(int i=0;i<int((v).size());++i)
int calls=0,fault=0;std::vector<int>supplied;
int ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(void*,int n,const int*p){++calls;supplied.assign(p,p+n);return fault;}
struct Resident{void*handle_=this;bool gasWallEnergyConfigured_=false;const fvMesh*residentMesh_=nullptr;int gasWallEnergyMeshFaces_=0;labelList gasWallEnergyFaceIds_,gasWallEnergyPatchSizes_;
void requireInitialised(const char*){}void check(int e,const char*){if(e)throw std::runtime_error("backend error");}
'''+method+r'''
};int main(){Resident r;fvMesh foreign{{{1,2},{5,1}}};r.configureGasWallEnergyLedger(foreign,labelList{0});assert((supplied==std::vector<int>{1,2}));r.configureGasWallEnergyLedger(foreign,labelList{1,1});assert((supplied==std::vector<int>{5,5})&&r.gasWallEnergyPatchSizes_.size()==2);
r.configureGasWallEnergyLedger(foreign,labelList{});assert(!r.gasWallEnergyConfigured_&&supplied.empty());fault=1;bool failed=false;try{r.configureGasWallEnergyLedger(foreign,labelList{0});}catch(const std::runtime_error&){failed=true;}assert(failed&&calls==4);}
''')


def test_actual_recapture_errors_propagate(tmp_path):
    h.run_cpp(tmp_path,graph_native()+r'''
int main(){int count=0;for(int failed=0;failed<=count;++failed){DeviceState host,device;host.deviceState=&device;host.wallEnergy.deviceState=&device.wallEnergy;char error[2048]{};host.wallEnergy.errorBuffer=error;host.wallEnergy.nFaces=5;host.wallEnergy.nInternalFaces=1;
 int a[]={1};failGraph=0;assert(ugkwpGpuResidentStrictConfigureGasWallEnergyLedger(&host,1,a)==0);host.wallEnergy.gasWallFlux[1]=0;
 graphCall=0;failGraph=failed;int rc=advancePureGasGraph(&host,.25,0.);if(!failed){assert(rc==0);count=graphCall;}else assert(rc!=0);
 failGraph=0;assert(ugkpInvalidateGasWallEnergyGraph(&host)==0);ugkpReleaseGasWallEnergy64(&host.wallEnergy);assert(live.empty());}assert(count>10);}
''')
