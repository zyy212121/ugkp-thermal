"""Actual resident ABI bodies with typed host allocation/copy boundaries, not CUDA."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import re
import json
import subprocess
import pytest
from test_native_cleanup import block, CUDA, run_cpp
ROOT=SOURCE_ROOT
BASE='5f1a39ac96e2e7ff9a0640a4db344d9cccd34245'

def source(app,rev=None):
    p=f'applications/{app}/'+('gpu' if app=='CHT' else 'private_backend')+'/GpuResidentStrict.cu'
    return subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()

def body(app,name,rev=None):return block(source(app,rev),'extern "C" int ugkwpGpuResidentStrict'+name)
def args(code):
    signature=code[code.index('(')+1:code.index(')')]
    return [(x.strip().rsplit(' ',1)[0],x.strip().rsplit(' ',1)[1]) for x in signature.split(',')]

def prelude():return CUDA+r'''
#include <algorithm>
#include <iostream>
using GpuReal=double;using GpuTime=double;
#include "common/GpuParticleRestartTheta.H"
#define GPU_R(x) (x)
#define GPU_TINY(x) (x)
constexpr double OfSmall=1e-15;
void setLastErrorText(const char*){}void setLastError(const char*,int){}
namespace Foam{namespace gpuThermal{constexpr int coldWallAxialNodeCount=8,coldWallRadialRingCount=8,coldWall2DNodeCount=64,coldWall2DRadialNodeCount=8,particleWallDeposited=1,particleWallTransientRebound=2,particleWallTransientDeposit=3;}}
'''

def create_fixture(app,rev=None):
    code=body(app,'Create',rev);src=source(app,rev)+(ROOT/'common/GpuAutomaticCsrScheduleFields.inl').read_text();fields=sorted(set(re.findall(r's->(\w+)',code)))
    declarations=[]
    for field in fields:
        match=re.search(r'^\s*((?:unsigned long long|unsigned short|double|GpuReal|int|bool))\s+'+field+r'\s*=',src,re.M)
        assert match,field
        declarations.append(f'{match[1]} {field}{{}};')
    parameters=args(code);defaults={'particleGasHeatTransferModelId':'1','gammaGas':'1.4','gasMu':'.1','gasPr':'.7','packingFraction':'.6','particleBlockThreads':'128','reductionBlockThreads':'128','gasRobustFallback':'1','csrCellLocalPathEnabled':'1','gasFluxScheme':'2','gasReconstruction':'1','gasLimiter':'1','gasTimeIntegrator':'1','turbulenceModel':'0','csrHeavyReductionMode':'0','csrWarpAggregatedBinning':'0','jammingPressureEnabled':'0'}
    declarations2=''.join(f'{typ} {name}={defaults.get(name.removesuffix("Wire"),"2")};' for typ,name in parameters[:-1])
    call='ugkwpGpuResidentStrictCreate('+','.join(n for _,n in parameters[:-1])+',&handle)'
    dump=''.join(f'std::cout<<" {f}="<<std::hexfloat<<s->{f};' for f in fields)
    return prelude()+'struct DeviceState{'+''.join(declarations)+'};\n'+r'''
int allocateCalls=0,occupancyCalls=0,releases=0,allocationError=0,occupancyError=0;
int allocateFields(DeviceState*){++allocateCalls;return allocationError;}
int configureLaunchOccupancy(DeviceState*){++occupancyCalls;return occupancyError;}
void releaseState(DeviceState*s){++releases;delete s;}
'''+code+'\nint main(){'+declarations2+'void*handle=nullptr;\n'+r'''
assert(CALL==0);auto*s=static_cast<DeviceState*>(handle);DUMP;std::cout<<"\n";delete s;
#ifdef COUNTEREXAMPLE
gammaGas=2.;assert(CALL==0);s=static_cast<DeviceState*>(handle);assert(s->gammaGas==2.&&s->gasCp==2.*Rgas);delete s;
gasMu=-3.;gasFluxScheme=27;gasRobustFallback=0;assert(CALL==0);s=static_cast<DeviceState*>(handle);assert(s->gasMu==-3.&&s->gasFluxScheme==27&&s->gasRobustFallback==0);delete s;
#endif
allocationError=1;assert(CALL!=0&&handle==nullptr);allocationError=0;occupancyError=1;assert(CALL!=0&&handle==nullptr&&releases==1);
}
'''.replace('CALL',call).replace('DUMP',dump).replace('gammaGas=2.', 'gammaGasWire=2.' if app=='CHT' else 'gammaGas=2.').replace('2.*Rgas)', '2.*RgasWire)' if app=='CHT' else '2.*Rgas)').replace('gasMu=-3.', 'gasMuWire=-3.' if app=='CHT' else 'gasMu=-3.')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_create_preserves_fields_and_admits_supplied_parameters(tmp_path,app):
    outputs=[]
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();outputs.append(run_cpp(d,create_fixture(app,rev)))
    assert outputs[0]==outputs[1]
    d=tmp_path/'continued';d.mkdir();run_cpp(d,'#define COUNTEREXAMPLE\n'+create_fixture(app))

def restart_fixture(app,rev=None):
    code=body(app,'UploadParticleRestartMirror',rev).replace('std::isfinite(', 'checkedFinite(')
    code=re.sub(r'<<<.*?>>>','',code,flags=re.S)
    fields=args(code)[2:];types={name:typ.replace('const ','').replace('*','').strip() for typ,name in fields}
    widths={name:(64 if name=='pCold2DNodeSpecificEnthalpy' else 8 if name in ['pColdNodeSpecificEnthalpy','pColdRingSolidMass','pCold2DRingContactAge'] else 1) for _,name in fields}
    state=''.join(f'{types[n]} {n}Storage[{2*widths[n]}]{{}};{types[n]}*{n}={n}Storage;' for _,n in fields)
    data=''.join(f'{types[n]} {n}[{2*widths[n]}];for(int j=0;j<{2*widths[n]};++j){n}[j]=static_cast<{types[n]}>({k+1}+j*.125);' for k,(_,n) in enumerate(fields))
    data+='for(int i=0;i<2;++i){pStuck[i]=0;pStuckFaceId[i]=-1;pDepositionArea[i]=pContactDuration[i]=pContactMaximumArea[i]=pContactPeakFraction[i]=0;}'
    call='ugkwpGpuResidentStrictUploadParticleRestartMirror(&s,n,'+','.join(n for _,n in fields)+')'
    inactive='ugkwpGpuResidentStrictUploadParticleRestartMirror(&s,n,'+','.join('nullptr' if n.startswith('pCold') else n for _,n in fields)+')'
    missing='ugkwpGpuResidentStrictUploadParticleRestartMirror(&s,n,'+','.join('nullptr' if n=='px' else n for _,n in fields)+')'
    checks=''.join(f'assert(std::memcmp(s.{n},{n},sizeof({n}))==0);' for _,n in fields)
    dump=''.join(f'for(auto x:s.{n}Storage)std::cout<<std::hexfloat<<double(x)<<" ";std::cout<<"\\n";' for _,n in fields)
    return prelude()+'struct DeviceState{'+state+r'''
double pContactAgeStorage[2]{};double*pContactAge=pContactAgeStorage;DeviceState*deviceState=this;int particleCapacity=2,nFaces=4,particleWorkGrid=1,particleBlockThreads=128,count=0,wallCount=0;
int*particleCountDevice=&count,*wallBoundParticleCountDevice=&wallCount;
int preBaseDirectoryReady=1,splitPreDirectoryActive=1,coldWallSolidificationEnabled=1,coldWall2DEnabled=1;bool particlesMayBePresent=false;
};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){if(n&&(!d||!s))return 1;return n?cudaMemcpy(d,s,n*sizeof(T),1):0;}
int finiteChecks=0;template<class T>bool checkedFinite(T x){++finiteChecks;return std::isfinite(x);}
int rebuildCalls=0,rebuildError=0;int rebuildResidentParticleMomentsFromParticles(DeviceState*s,int n){++rebuildCalls;assert(n==s->count);return rebuildError;}
void rebuildWallBoundParticleDirectoryKernel(DeviceState*s){++launches;s->wallCount=0;for(int i=0;i<s->count;++i)if(s->pStuck[i])++s->wallCount;}
'''+code+'\nint main(){DeviceState s;int n=2;'+data+r'''
counters();assert(CALL==0);CHECKS;DUMP;std::cout<<"COST "<<copies<<" "<<copiedBytes<<" "<<memsets<<" "<<zeroedBytes<<"\n";std::cout<<"SCANS "<<finiteChecks<<"\n";
#ifdef COUNTEREXAMPLE
pCold2DNodeSpecificEnthalpy[0]=-1.;pm[0]=-2.;pTheta[1]=-3.;s.particleCapacity=1;counters();assert(CALL==0);CHECKS;assert(s.count==2&&launches==1);
s.coldWallSolidificationEnabled=s.coldWall2DEnabled=0;counters();assert(INACTIVE==0);assert(MISSING!=0);s.coldWallSolidificationEnabled=s.coldWall2DEnabled=1;
#endif
counters();assert(CALL==0);const int copyCount=copies;for(int fail=1;fail<=copyCount;++fail){counters();failCopy=fail;assert(CALL!=0);}
counters();failMemset=1;assert(CALL!=0);counters();failLaunch=1;assert(CALL!=0);counters();rebuildError=1;assert(CALL!=0);rebuildError=0;
n=0;counters();assert(CALL==0&&s.count==0&&launches==0);
}
'''.replace('CALL',call).replace('CHECKS',checks).replace('DUMP',dump).replace('INACTIVE',inactive).replace('MISSING',missing)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_restart_preserves_all_payloads_and_continues_backed_metadata(tmp_path,app):
    outputs=[]
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();outputs.append(run_cpp(d,restart_fixture(app,rev)))
    assert [x for x in outputs[0].splitlines() if not x.startswith('SCANS')]==[x for x in outputs[1].splitlines() if not x.startswith('SCANS')]
    assert 'SCANS 0' in outputs[1] and 'SCANS 0' not in outputs[0]
    (tmp_path/'restart-parity.json').write_text(json.dumps(dict(old=outputs[0],current=outputs[1]),indent=2))
    d=tmp_path/'continued';d.mkdir();run_cpp(d,'#define COUNTEREXAMPLE\n'+restart_fixture(app))

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_mesh_upload_preserves_all_arrays_and_periodic_flag(tmp_path,app):
 code=body(app,'UploadMesh');fields=args(code)[1:];decl='';data='';checks=''
 for k,(typ,n) in enumerate(fields):
  t=typ.replace('const ','').replace('*','').strip();decl+=f'{t} {n}Store[4]{{}};{t}*{n}={n}Store;';data+=f'{t} {n}[4];for(int i=0;i<4;++i){n}[i]=static_cast<{t}>({k}+i*.25);';checks+=f'assert(std::memcmp(s.{n},{n},sizeof {n})==0);'
 call='ugkwpGpuResidentStrictUploadMesh(&s,'+','.join(n for _,n in fields)+')'
 run_cpp(tmp_path,prelude()+'struct DeviceState{'+decl+'int nCells=4,nFaces=4,nInternalFaces=2,nCellPlanes=4,hasPeriodicFaces=0;};'+r'''
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){if(n&&(!d||!s))return 1;return cudaMemcpy(d,s,n*sizeof(T),1);}
'''+code+'int main(){DeviceState s;'+data+r'''
facePeriodicPair[0]=-1;facePeriodicPair[1]=-1;facePeriodicPair[2]=3;facePeriodicPair[3]=2;faceNeighbour[2]=1;faceNeighbour[3]=0;
assert(CALL==0&&s.hasPeriodicFaces==1);CHECKS;
facePeriodicPair[2]=2;facePeriodicDx[2]=NAN;counters();assert(CALL==0&&s.hasPeriodicFaces==1);CHECKS;
const int count=copies;for(int i=1;i<=count;++i){counters();failCopy=i;assert(CALL!=0);}
for(int i=0;i<4;++i)facePeriodicPair[i]=-1;counters();assert(CALL==0&&s.hasPeriodicFaces==0);}
'''.replace('CALL',call).replace('CHECKS',checks))

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_gas_only_body_keeps_stage_outputs_and_errors(tmp_path,app):
 code=re.sub(r'<<<.*?>>>','',body(app,'AdvanceGasOnly'),flags=re.S)
 run_cpp(tmp_path,prelude()+r'''
#define UGKP_DEVELOPMENT_PROBES
struct DeviceState{DeviceState*deviceState=this;int particleCapacity=3,nBoundarySources=2,gravityEnabled=1,fixedCellBlockThreads=2,nCells=2;double rhoU[2]{1,2};};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
int stages=0,finals=0,error=0;int tuneFixedWorkBlockThreads(DeviceState*,double,double){return error==1;}
int advanceGasFluxStage(DeviceState*s,double dt,double time){++stages;if(error==2)return 1;for(auto&x:s->rhoU)x+=dt+time;return 0;}
void applyGasGravityKernel(DeviceState*s,double dt){++launches;for(auto&x:s->rhoU)x+=2*dt;}
void applyGasOnlyGravityKernel(DeviceState*s,double dt){applyGasGravityKernel(s,dt);}
void recoverGasPrimitivesKernel(DeviceState*){++launches;}
int finaliseGasBoundaryStage(DeviceState*s,double dt,double time){++finals;if(error==3)return 1;for(auto&x:s->rhoU)x+=dt-time;return 0;}
'''+code+r'''
int main(){DeviceState s;counters();assert(ugkwpGpuResidentStrictAdvanceGasOnly(&s,2,-3)==0&&stages==1&&finals==1&&s.rhoU[0]==9&&s.rhoU[1]==10);
for(int e=1;e<=3;++e){counters();error=e;assert(ugkwpGpuResidentStrictAdvanceGasOnly(&s,2,-3)!=0);}error=0;counters();failLaunch=1;assert(ugkwpGpuResidentStrictAdvanceGasOnly(&s,2,-3)!=0);}
''')

def test_actual_affine_repeated_surplus_copies_only_consumed_prefix(tmp_path):
 code=re.sub(r'<<<.*?>>>','',body('CHT','ApplyParticleRadiationAffineTemperature'),flags=re.S)
 run_cpp(tmp_path,prelude()+r'''
namespace Foam{namespace gpuThermal{struct RadiationAffineTemperatureUpdate{double scale,offset;};}}
using Update=Foam::gpuThermal::RadiationAffineTemperatureUpdate;
struct DeviceState{DeviceState*deviceState=this;int nCells=2,particleWorkGrid=1,particleBlockThreads=128;bool particlesMayBePresent=true;Update*particleRadiationAffineUpdate=nullptr;double temperature[2]{300,400};};
DeviceState*asState(void*p){return static_cast<DeviceState*>(p);}int validateState(DeviceState*s,const char*){return !s;}
template<class T>int allocate(T*&p,size_t n,const char*){return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
template<class T>void release(T*&p){cudaFree(p);p=nullptr;}
template<class T>int copyToDevice(T*d,const T*s,size_t n,const char*){return cudaMemcpy(d,s,n*sizeof(T),1);}
void applyParticleRadiationAffineTemperatureKernel(DeviceState*s,const Update*p){++launches;for(int i=0;i<s->nCells;++i)s->temperature[i]=p[i].scale*s->temperature[i]+p[i].offset;}
'''+code+r'''
int main(){DeviceState s;Update u[4]{{2,1},{3,2},{99,99},{NAN,NAN}};
counters();assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,3,u)==0);assert(s.temperature[0]==601&&s.temperature[1]==1202&&copiedBytes==2*sizeof(Update));
counters();assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,4,u)==0);assert(s.temperature[0]==1203&&s.temperature[1]==3608&&allocations==0&&copiedBytes==2*sizeof(Update));
assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,u)!=0);counters();failCopy=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,2,u)!=0);counters();failLaunch=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,2,u)!=0);counters();failSync=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,2,u)!=0);release(s.particleRadiationAffineUpdate);counters();failAlloc=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,2,u)!=0);}
''')
