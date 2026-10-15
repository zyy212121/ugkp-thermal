"""Actual SST Configure and initializer; CUDA boundaries emulated, no GPU claim."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import subprocess
import re
import json
import pytest
ROOT=SOURCE_ROOT
BASE='91a51f9e1269a2c9d37a9fb900cdd2f3be30687d'
spec=importlib.util.spec_from_file_location('h',(VALIDATION_ROOT / 'tests/test_native_cleanup.py'));h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

def path(app):return f'applications/{app}/'+('gpu' if app=='CHT' else 'private_backend')+'/GpuResidentStrict.cu'
def source(app,rev=None):return subprocess.check_output(['git','show',rev+':'+path(app)],cwd=ROOT,text=True) if rev else (ROOT/path(app)).read_text()
def inventory(app):
    alloc=h.block(source(app,BASE),'int allocateFields(')
    return re.findall(r'allocate\(s->(\w+), (nc|nf), "(cudaMalloc SST [^"]+)"\)',alloc)

def code(app,rev=None):
    src=source(app,rev);slots=inventory(app);assert len(slots)==22
    scalar=src[src.index('    int sstConfigured ='):src.index('    GpuReal particleDiameterFallback' if app=='CHT' else '    double particleDiameterFallback')]
    scalar=scalar.replace('GpuReal','double').replace('int sstConfigured =', 'alignas(8) int sstConfigured =')
    members='\n'.join(('int' if n.endswith('Mode') else 'double')+'* '+n+'=nullptr;' for n,_,_ in slots)
    cuda=h.CUDA.replace('#include <vector>','#include <vector>\n#include <map>\n#include <string>\n#include <iostream>\n#include <cstddef>\nstd::map<void*,size_t>live;std::vector<std::string>events;')
    cuda=cuda.replace('++copies;if(copies==failCopy)','++copies;events.push_back("copy "+std::to_string(kind)+" "+std::to_string(n));if(copies==failCopy)')
    cuda=cuda.replace('++allocations;if(allocations==failAlloc)','++allocations;events.push_back("alloc "+std::to_string(n));if(allocations==failAlloc)')
    cuda=cuda.replace('*p=std::malloc(n);return *p?0:1;','*p=std::malloc(n);if(*p){live[*p]=n;std::memset(*p,0xa5,n);}return *p?0:1;')
    cuda=cuda.replace('int cudaFree(void* p){std::free(p);return 0;}','int cudaFree(void*p){if(p){assert(live.erase(p));std::free(p);}return 0;}')
    cuda=cuda.replace('++syncs;return','++syncs;events.push_back("sync");return').replace('void counters(){','void counters(){events.clear();')
    common=r'''
#include "GpuPrecisionTypes.H"
#include "common/gasNumerics/GpuSstAlgebra.cuh"
#include "common/gasNumerics/OpenFoamWallFunctions.cuh"
#define __device__
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) (x)
#include "common/operators/clampMin.cuh"
constexpr double OfSmall=2.22044604925031308085e-16;
struct DeviceState{DeviceState*deviceState=nullptr;int nCells=2,nFaces=3,hostTurbulenceModel=3,fixedCellBlockThreads=1;unsigned short particleStuckInitialisationPending=0;
SCALARS
unsigned long long unrelated=0x12345678;double* rho=nullptr;double* nut=nullptr;double rhoMin=.01,gasPrClamped=.7,turbulentPrandtl=.9;
MEMBERS
unsigned long long trailing=0xabcdef;
};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s||!s->deviceState;}void setLastErrorText(const char*){}void setLastError(const char*,int){}
template<class T>int allocate(T*&p,size_t n,const char*){p=nullptr;if(!n)return 0;return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),1):0;}
'''.replace('SCALARS',scalar).replace('MEMBERS',members)
    init=h.block((ROOT/'common/operators/computeGasPrimitiveGradientsKernel.cuh').read_text(),'template<class GasState>\n__global__ void initialiseSstConservativeStateKernel')
    launch=r'''
void launchInitialise(DeviceState*s,int grid,int block){++launches;events.push_back("launch");if(launches==failLaunch)return;blockDim.x=block;for(int b=0;b<grid;++b)for(int t=0;t<block;++t){blockIdx.x=b;threadIdx.x=t;initialiseSstConservativeStateKernel(s);}}
'''
    helpers=h.block(src,'int syncSstConfiguration(')+'\n'
    if 'int prepareSstOwners(' in src:helpers+=h.block(src,'int prepareSstOwners(')+'\n'
    config=h.block(src,'extern "C" int ugkwpGpuResidentStrictConfigureSst').replace('initialiseSstConservativeStateKernel<<<grid, block>>>(s->deviceState);','launchInitialise(s->deviceState,grid,block);')
    posts='void allocateOriginal(DeviceState&s){const size_t nc=s.nCells,nf=s.nFaces;int rc=0;'+''.join(f'rc|=allocate(s.{n},{count},"{name}");' for n,count,name in slots)+'assert(rc==0);}\n'
    posts+='void cleanup(DeviceState&s){'+''.join(f'cudaFree(s.{n});s.{n}=nullptr;' for n,_,_ in slots)+'}\n'
    posts+='std::vector<void*> pointers(DeviceState&s){return {'+','.join('s.'+n for n,_,_ in slots)+'};}\n'
    posts+=r'''
struct Input{double k[2]={.3,.7},omega[2]={2,3},distance[2]={.5,.6},bk[3]={.1,.2,.3},bw[3]={1,2,3};int km[3]={0,1,2},wm[3]={2,1,0};};
int configure(DeviceState&s,Input&i){return ugkwpGpuResidentStrictConfigureSst(&s,.85,1,.5,.856,.075,.0828,.09,.55555,.44,.31,1,10,1e-12,1e-6,.25,0,.41,9.8,.09,i.k,i.omega,i.distance,i.km,i.wm,i.bk,i.bw);}
void setup(DeviceState&s,DeviceState&d){static double rho[]={1.25,.75};static double nut[]={97,98};s.rho=rho;s.nut=nut;s.deviceState=&d;d=s;d.unrelated=111;d.trailing=222;}
void complete(DeviceState&s,DeviceState&d,Input&i){assert(pointers(d)==pointers(s));for(int c=0;c<2;++c){assert(s.rhoK[c]==s.rho[c]*i.k[c]);assert(s.rhoOmega[c]==s.rho[c]*i.omega[c]);assert(s.rhoKInitial[c]==s.rhoK[c]&&s.rhoOmegaInitial[c]==s.rhoOmega[c]&&s.nut[c]==0);}assert(d.sstJayatillekeP!=0&&d.sstThermalYPlus!=0);assert(d.unrelated==111&&d.trailing==222);}
'''
    return cuda+common+init+launch+helpers+config+posts

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_late_sst_canonical_owners_and_actual_initializer(tmp_path,app):
    h.run_cpp(tmp_path,code(app)+r'''
int main(){DeviceState s,d;Input i;setup(s,d);s.hostTurbulenceModel=0;counters();assert(configure(s,i)==0);assert(allocations==22&&copies==9&&launches==1&&syncs==1&&memsets==0);complete(s,d,i);assert(!(s.particleStuckInitialisationPending&4096));auto owners=pointers(s);counters();i.k[0]=2;assert(configure(s,i)==0);assert(pointers(s)==owners&&allocations==0&&copies==8&&launches==1&&syncs==1);complete(s,d,i);cleanup(s);assert(live.empty());}
''')

def partial(c):return c.replace('if(copies==failCopy)return 1;','if(copies==failCopy){if(n)std::memcpy(d,s,(n+1)/2);return 1;}')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_old_complete_first_repeat_exact_fields_and_work(tmp_path,app):
    main=r'''
void word(double v){unsigned long long b;std::memcpy(&b,&v,8);std::cout<<" "<<b;}
void output(DeviceState&s,DeviceState&d){for(auto*p:{s.rhoK,s.rhoOmega,s.rhoKInitial,s.rhoOmegaInitial,s.k,s.omega,s.sstWallDistance,s.nut})for(int c=0;c<2;++c)word(p[c]);for(auto*p:{s.sstBoundaryK,s.sstBoundaryOmega})for(int f=0;f<3;++f)word(p[f]);for(int f=0;f<3;++f)std::cout<<" "<<s.sstBoundaryKMode[f]<<" "<<s.sstBoundaryOmegaMode[f];word(d.sstJayatillekeP);word(d.sstThermalYPlus);std::cout<<"\n";}
int main(){DeviceState s,d;Input i;allocateOriginal(s);setup(s,d);for(int turn=0;turn<2;++turn){if(turn){i.k[0]=9;i.omega[1]=7;s.rhoKInitial[0]=-55;s.nut[1]=88;}counters();assert(configure(s,i)==0);complete(s,d,i);assert(allocations==0&&memsets==0&&copies==8&&launches==1&&syncs==1);std::cout<<"DATA";output(s,d);std::cout<<"COST "<<allocations<<" "<<copies<<" "<<copiedBytes<<" "<<launches<<" "<<syncs<<" "<<memsets<<"\n";for(auto&e:events)std::cout<<e<<"\n";}cleanup(s);assert(live.empty());}
'''
    result={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();result[label]=h.run_cpp(d,code(app,rev)+main)
    assert result['old']==result['current'];(tmp_path/'old-success-parity.json').write_text(json.dumps(result,indent=2))

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_every_allocation_copy_launch_sync_failure_retries_actual_tail(tmp_path,app):
    result=h.run_cpp(tmp_path,partial(code(app))+r'''
int main(){for(int fault=1;fault<=22;++fault){DeviceState s,d;Input i;setup(s,d);s.hostTurbulenceModel=0;s.particleStuckInitialisationPending=3;counters();failAlloc=fault;assert(configure(s,i)!=0&&allocations==22&&copies==0&&live.size()==21);assert((s.particleStuckInitialisationPending&4096)!=0);auto before=pointers(s);counters();assert(configure(s,i)==0&&allocations==1&&copies==9);auto after=pointers(s);for(size_t n=0;n<before.size();++n)if(before[n])assert(before[n]==after[n]);assert(s.particleStuckInitialisationPending==3);complete(s,d,i);cleanup(s);assert(live.empty());}
 for(int fault=1;fault<=11;++fault){DeviceState s,d;Input i;setup(s,d);s.hostTurbulenceModel=0;counters();if(fault<=9)failCopy=fault;else if(fault==10)failLaunch=1;else failSync=1;
 assert(configure(s,i)!=0&&allocations==22&&live.size()==22);if(fault==1)assert(s.particleStuckInitialisationPending&4096);else assert(!(s.particleStuckInitialisationPending&4096));if(fault==11)assert(s.rhoKInitial[0]==s.rho[0]*i.k[0]);auto before=pointers(s);
 counters();assert(configure(s,i)==0&&allocations==0&&copies==(fault==1?9:8));assert(before==pointers(s));complete(s,d,i);cleanup(s);assert(live.empty());}
 std::cout<<"FAULTS 22 allocation + 9 partial-copy + launch + sync = 33\n";}
''');(tmp_path/'faults.log').write_text(result)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_bad_numbers_use_real_initializer_and_missing_input_stays_error(tmp_path,app):
    c=code(app).replace('(&s,.85,1,.5,','(&s,-.85,1,.5,').replace('1e-6,.25,0,.41','1e-6,.25,99,.41')
    bad=tmp_path/'bad';bad.mkdir()
    h.run_cpp(bad,c+r'''
int main(){DeviceState s,d;Input i;setup(s,d);s.hostTurbulenceModel=17;i.k[0]=std::nan("17");i.k[1]=-3;i.omega[0]=std::numeric_limits<double>::infinity();i.distance[0]=-1;i.km[0]=99;i.bk[1]=std::nan("91");assert(configure(s,i)==0);assert(s.k[0]==s.sstKMin&&s.k[1]==s.sstKMin&&s.omega[0]==s.sstOmegaMin);assert(s.rhoKInitial[0]==s.rho[0]*s.sstKMin&&s.nut[1]==0);assert(s.sstWallDistance[0]==-1&&s.sstBoundaryKMode[0]==99&&std::memcmp(s.sstBoundaryK+1,i.bk+1,8)==0);assert(d.sstCoefficients.alphaK1==-.85&&d.sstWallTreatment==99);cleanup(s);assert(live.empty());}
''')

    missing=tmp_path/'missing';missing.mkdir()
    h.run_cpp(missing,code(app).replace('i.k,i.omega,i.distance,','nullptr,i.omega,i.distance,')+r'''
int main(){DeviceState s,d;Input i;setup(s,d);s.hostTurbulenceModel=0;counters();assert(configure(s,i)!=0&&allocations==0&&copies==0&&launches==0);}
''')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_zero_size_does_not_add_publication_and_keeps_launch_error(tmp_path,app):
    main=r'''
int main(){DeviceState s,d;Input i;s.nCells=s.nFaces=0;setup(s,d);counters();failLaunch=1;assert(configure(s,i)!=0);assert(allocations==0&&memsets==0&&copies==1&&launches==1&&syncs==0);std::cout<<copies<<" "<<copiedBytes<<" "<<s.particleStuckInitialisationPending<<"\n";}
'''
    result={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();result[label]=h.run_cpp(d,code(app,rev)+main)
    assert result['old']==result['current']

def header(app,rev=None):
    spec=importlib.util.spec_from_file_location('consumer',(VALIDATION_ROOT / 'tests/test_native_consumer_admission.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    src=c.source(f'applications/{app}/gpu/GpuResidentStrict.H',rev)
    pre=c.HEADER_PRELUDE.replace('const Boundary&boundaryMesh()const{return patches;}','const Boundary&boundaryMesh()const{return patches;}const Boundary&boundary()const{return patches;}')
    old=h.block(pre,'struct volScalarField')+';'
    pre=pre.replace(old,r'''
struct fvPatchScalarField:scalarField{using scalarField::scalarField;bool fixed=false,io=false;bool fixesValue()const{return fixed;}const scalarField&refValue()const{return *this;}};
using inletOutletFvPatchScalarField=fvPatchScalarField;
template<class T>bool isA(const fvPatchScalarField&p){return p.io;}
template<class T>T&refCast(const fvPatchScalarField&p){return static_cast<T&>(p);}
struct volScalarField{scalarField values;List<fvPatchScalarField>patches;size_t size()const{return values.size();}const scalarField&primitiveField()const{return values;}const List<fvPatchScalarField>&boundaryField()const{return patches;}};
''')
    signature=h.block(source('FSH',BASE),'extern "C" int ugkwpGpuResidentStrictConfigureSst').split('{',1)[0]
    names=['alphaK1','alphaK2','alphaOmega1','alphaOmega2','beta1','beta2','betaStar','gamma1','gamma2','a1','b1','c1','kMin','omegaMin','maxSourceNumber','wallTreatment','wallKappa','wallE','wallCmu']
    boundary=signature+'{++nativeCalls;arguments={'+','.join('double('+n+')' for n in names)+'};for(int c=0;c<2;++c){arguments.push_back(k[c]);arguments.push_back(omega[c]);arguments.push_back(wallDistance[c]);}for(int f=0;f<6;++f){arguments.push_back(boundaryKMode[f]);arguments.push_back(boundaryOmegaMode[f]);arguments.push_back(boundaryK[f]);arguments.push_back(boundaryOmega[f]);}return nativeFault;}\n'
    wrapper=r'''
struct Resident{void*handle_=this;const fvMesh*residentMesh_=nullptr;void requireInitialised(const char*)const{}void check(int status,const char*)const{if(status)throw std::runtime_error("native failure");}
'''+h.block(src,'    void configureSst')+'};\n'
    wrapper+=r'''
void configure(Resident&r,fvMesh&m,volScalarField&k,volScalarField&w,volScalarField&distance,int wall){r.configureSst(m,k,w,distance,.85,1,.5,.856,.075,.0828,.09,.55555,.44,.31,1,10,1e-12,1e-6,.25,wall,.41,9.8,.09);}
void fill(volScalarField&k,volScalarField&w,volScalarField&distance){k.values={.3,.7};w.values={2,3};distance.values={.5,.6};k.patches={{.1,.2},{.3,.4}};w.patches={{3,4},{5,6}};k.patches[1].io=true;w.patches[1].fixed=true;}
'''
    return h.CUDA+pre+boundary+wrapper

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_frontend_old_boundary_math_exact_and_new_identity_admission(tmp_path,app):
    main=r'''
int main(){fvMesh mesh;Resident r;r.residentMesh_=&mesh;volScalarField k,w,d;fill(k,w,d);for(int wall:{0,1}){configure(r,mesh,k,w,d,wall);std::cout<<"DATA";for(double v:arguments){unsigned long long b;std::memcpy(&b,&v,8);std::cout<<" "<<b;}std::cout<<"\n";}assert(nativeCalls==2);}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=h.run_cpp(d,header(app,rev)+main)
    assert results['old']==results['current'];(tmp_path/'frontend-parity.json').write_text(json.dumps(results,indent=2))
    d=tmp_path/'admitted';d.mkdir();h.run_cpp(d,header(app)+r'''
int main(){fvMesh mesh,foreign;Resident r;r.residentMesh_=&foreign;volScalarField k,w,d;fill(k,w,d);k.values.push_back(99);configure(r,mesh,k,w,d,99);assert(nativeCalls==1&&arguments[15]==99);nativeFault=1;bool failed=false;try{configure(r,mesh,k,w,d,99);}catch(const std::runtime_error&){failed=true;}assert(failed&&nativeCalls==2);}
''')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_other_lifecycle_operations_preserve_sst_publication_bit(tmp_path,app):
    spec=importlib.util.spec_from_file_location('stuck',(VALIDATION_ROOT / 'tests/test_stuck_owner_lifecycle.py'));stuck=importlib.util.module_from_spec(spec);spec.loader.exec_module(stuck)
    h.run_cpp(tmp_path,stuck.code(app)+r'''
int main(){DeviceState s,d;s.deviceState=&d;s.particleStuckInitialisationPending=4096;unsigned char mask[]={0,1,0,0};assert(configure(s,mask)==0);assert(s.particleStuckInitialisationPending==4096);unsigned char off[4]{};assert(configure(s,off,0)==0);assert(s.particleStuckInitialisationPending==4096);cleanup(s);assert(live.empty());}
''',[ROOT/f'applications/{app}/thermal'])
