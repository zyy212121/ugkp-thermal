"""Sentinel-filled actual production-header continuation fixtures, CPU only."""
from pathlib import Path
import re
import os
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
COMMON=Path(os.environ.get('PARTICLE_TERMINAL_COMMON', ROOT/'common'))
def body(path):
    return (COMMON/path).read_text().replace('#pragma once','').replace('asm("trap;");','throw std::runtime_error("production trap");')
def function(text,name):
    a=text.index(name);a=text.rfind('\n',0,a)+1

    if text[a:].startswith(name):a=text.rfind('\n',0,a-1)+1
    end=text.index('{',a)+1;depth=1
    while depth:depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[a:end]
def run(code,tmp_path):
    cpp=tmp_path/'probe.cpp';cpp.write_text(code);exe=tmp_path/'probe'
    q=subprocess.run(['g++','-std=c++17','-O1','-g','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(COMMON),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30,cwd=tmp_path)
    assert q.returncode==0,q.stdout+q.stderr
PREFIX=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <type_traits>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __syncthreads() ((void)0)
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) (x)
#define GPU_PIPELINE_REAL double
using std::min;using std::max;
struct Dim{unsigned x=0;} blockIdx,threadIdx,blockDim{1},gridDim{1};
'''

def test_contact_configuration_keeps_finite_coverage_and_heat_disabled(tmp_path):
    run(PREFIX+body('ContactCapabilityValidation.h')+r'''
int main(){for(double coverage:{-2.,0.,.5,2.})for(int heat:{0,1})
 assert(validateContactCapabilities(1,true,true,heat,coverage)==nullptr);
 assert(validateContactCapabilities(1,false,false,1,std::numeric_limits<double>::infinity())!=nullptr);}
''',tmp_path)

@pytest.mark.parametrize('scalar,width',[('float',8),('float',7),('double',8),('int',8)])
def test_packed_option_uses_actual_compatible_type_or_scalar_copy(tmp_path,scalar,width):
    run(PREFIX+r'''
struct alignas(16) float4 {float x,y,z,w;};
namespace Foam {namespace gpuThermal {enum{particleWallMobile=0,coldWallAxialNodeCount=8,coldWallRadialRingCount=8,coldWall2DNodeCount=64,coldWall2DRadialNodeCount=8};} namespace gpuWall{template<class S>void publishWallBoundParticleIndex(S&,int){}}}
'''+body('GpuCellLocalThermalFields.cuh')+f'''
int main(){{alignas(16) {scalar} src[{width*3}],dst[{width*3}];for(int i=0;i<{width*3};++i){{src[i]=i+1;dst[i]=-9;}}
 CellLocalThermalExtraFields<false,true>::copyArray<{width},true>(src,dst,2,1);
 for(int i=0;i<{width*3};++i)assert(dst[i]==(i>={width}&&i<{width*2}?src[i+{width}]:-9));}}
''',tmp_path)

@pytest.mark.parametrize('which',['moment','pool'])
def test_segmented_actual_producer_finalizer_zero_one_multiple_lineage(tmp_path,which):
    text=body('GpuSegmented'+which.title()+'Workers.cuh')
    producer=text[text.index('template<bool'):text.index('__global__')]
    producer=producer[:producer.rfind('};')+2]
    if which=='moment':
        a=text.index('__device__ __forceinline__ void finalizeCsrMomentCell');b=text.index('__global__',a)
        code='template<class MomentReal>\n'+function(body('GpuParticleMoments.cuh'),'publishParticleMomentsCell')+producer+text[a:b]
        setup='CsrMomentOperation<> operation{&descriptor,scratch};'
        invoke='CsrMomentFinalizeOperation{scratch}.execute(s,0);'
        output='s.momRhoP[0],s.momRhoUPx[0],s.momRhoUPy[0],s.momRhoUPz[0],s.momRhoEP[0],s.momRhoPD[0],s.momRhoHpP[0],double(s.cellParticleCount[0])'
    else:
        a=text.index('struct CsrPoolFinalizeOperation');b=text.index('__global__',a)
        code=body('GpuPoolMomentOperations.cuh')+body('operators/clearPoissonThermalPoolKernel.cuh')+producer+text[a:b]
        setup='clearPoissonThermalPoolKernel(&s,.25);double probability=1;CsrPoolOperation<true> operation{&descriptor,&probability,scratch,.25};'
        invoke='CsrPoolFinalizeOperation{scratch}.execute(s,0);'
        output='s.poissonPoolMass[0],s.poissonPoolMomX[0],s.poissonPoolMomY[0],s.poissonPoolMomZ[0],s.poissonPoolEnergy[0],s.poissonPoolDiameter[0],s.poissonPoolDiameter2[0],double(s.poolThermalCount[0])'
    run(PREFIX+r"""
#define GPU_POOL_STATIC_DIRECTORY 0
#define GPU_POOL_CACHED_PROBABILITY 1
#define GPU_POOL_INITIALIZATION_WITH_PROBABILITY 1
#define GPU_POOL_SPLIT_DIRECTORY false
#define GPU_POOL_DIRECT_PARTICLE_INDEX false
#define GPU_POOL_END_TASK(s,c,first) ((first)+(s).csrCellTaskCount[c])
#define GPU_POOL_SAMPLING_READY(s,c) (++(s).ready);
struct CsrReductionTask{int cell,begin,end,source;};
struct DeviceState{int nCells=1,csrHeavyReductionEnabled=1,csrCellTaskCount[1]={},csrCellTaskOffset[1]={1},csrMultiTaskCellList[1]={0},ready=0,calls=0,cellParticleCount[2]={-888,-888},poolThermalCount[1]={-888},poissonPoolSampleTargetCount[1]={-888};
 CsrReductionTask csrReductionTasks[4]={{0,0,0,0},{0,10,11,0},{0,20,21,0},{0,30,31,0}};
 double csrHeavyPartials[40],probability=1,poissonCellCollisionProbability[1]={-999},V[1]={2},rhoMin=1e-12;
 double momRhoP[1]={-888},momRhoUPx[1]={-888},momRhoUPy[1]={-888},momRhoUPz[1]={-888},momRhoEP[1]={-888},momRhoPD[1]={-888},momRhoHpP[1]={-888};
 double poissonPoolMass[1]={-888},poissonPoolMomX[1]={-888},poissonPoolMomY[1]={-888},poissonPoolMomZ[1]={-888},poissonPoolEnergy[1]={-888},poissonPoolDiameter[1]={-888},poissonPoolDiameter2[1]={-888},poolThermalSumUx[1]={-888},poolThermalSumUy[1]={-888},poolThermalSumUz[1]={-888},poolThermalSumU2[1]={-888};};
double clampMin(double x,double y){return max(x,y);}
template<int N>void blockReduceComponentSums(double(&)[N],double*){}
double poissonCollisionProbabilityForCell(DeviceState&s,int,double){return s.probability;}
// Only the numerical per-task accumulation hook is isolated. The actual
// producer, clear, partial store, finalizer and complete publishers execute.
template<bool B>void accumulateCsrHeavyMomentTask(DeviceState&s,int,int begin,int,double(&out)[8],double*){++s.calls;for(int j=0;j<8;++j)out[j]=begin+j;}
template<bool B>void accumulateCsrHeavyPoolTask(DeviceState&s,int,int begin,int,bool,double,double(&out)[8],double*){++s.calls;for(int j=0;j<8;++j)out[j]=begin+j;}
template<bool B>void accumulateCsrSplitLogicalPoolTask(DeviceState&s,int,int begin,int,double,double(&out)[8],double*){++s.calls;for(int j=0;j<8;++j)out[j]=begin+j;}
"""+code+f'\nint main(){{for(int n:{{0,1,3}})for(int active:{{0,1}}){{DeviceState s;s.csrCellTaskCount[0]=n;s.probability=active;for(double&v:s.csrHeavyPartials)v=-999;CsrReductionTask descriptor;double scratch[8]={{}};'+setup+r"""
for(int task=1;task<1+n;++task){assert(operation.prepare(s,task));operation.execute(s,task);}
"""+invoke+f'\ndouble output[8]={{{output}}};'+f'\nconst bool computed={"true" if which=="moment" else "active"};'+r"""
for(int component=0;component<8;++component){double expected=0;for(int task=1;task<1+n;++task)if(computed)expected+=10*task+component;
"""+('if(component<7)expected/=2;\n' if which=='moment' else '')+r"""
assert(output[component]==expected);}
assert(s.calls==(computed?n:0));
for(int task=0;task<5;++task)for(int component=0;component<8;++component){double expected=-999;if(n>1&&task>=1&&task<1+n)expected=computed?10*task+component:0;assert(s.csrHeavyPartials[8*task+component]==expected);}
"""+('assert(s.cellParticleCount[1]==0);' if which=='moment' else 'assert(s.ready==(n==1?(active?1:0):1));assert(s.poissonPoolSampleTargetCount[0]==0&&s.poolThermalSumUx[0]==0&&s.poolThermalSumUy[0]==0&&s.poolThermalSumUz[0]==0&&s.poolThermalSumU2[0]==0);')+'}}\n',tmp_path)


def launch_host(text):
    return re.sub(r'<<<.*?>>>','',text,flags=re.S)

def test_auto_negative_interval_executes_existing_unsigned_modulo(tmp_path):
    source=body('GpuAutomaticCsrSchedule.cuh')
    source=re.sub(r'#include[^\n]*','',source)
    run(PREFIX+r'''
#include <cstring>
#define GPU_DIRECTORY_PARAMETER_TYPE int
#define GPU_DIRECTORY_SELECTOR kind
#define GPU_DIRECTORY_ARGUMENT kind
#define GPU_AUTO_THRESHOLD_FIELD dynamicHeavyThreshold
#define GPU_AUTO_UPDATE_POLICY(s,kind) updatePolicy(s,kind)
using cudaError_t=int;constexpr int cudaSuccess=0,cudaErrorInvalidValue=17,cudaMemcpyDeviceToHost=1;
struct DeviceState{int csrHeavyReductionMode=2,particleCapacity=10,csrHeavyAutoInterval=-1,nCells=1,dynamicHeavyThreshold=2,csrHeavyTileParticles=0,csrHeavyReductionActive=0,csrHeavyReductionEnabled=0,csrTasksReady=0,csrPreparedDirectoryKind=0;
 unsigned long long schedulingAdvanceCount=0;DeviceState*deviceState=this;int maximum=5,*csrMaximumOccupancy=&maximum,nt=0,nc=0,*csrHeavyTaskCount=&nt,*csrHeavyCellCount=&nc;};
int decisions=0,prepares=0,updates=0,errors=0;
void setLastError(const char*,int){++errors;}
int updatePolicy(DeviceState*,int){++updates;return 0;}
int cudaMemset(void*p,int v,size_t n){std::memset(p,v,n);return 0;}
int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
int cudaGetLastError(){return 0;}
void maximumDirectoryOccupancyKernel(DeviceState*,int,int*p){*p=5;}
void publishHeavyReductionDecisionKernel(DeviceState*,int active){assert(active==1);++decisions;}
int prepareCsrSegmentedReductionTasks(DeviceState*s,int,int kind){++prepares;s->csrTasksReady=1;s->csrPreparedDirectoryKind=kind;return 0;}
'''+launch_host(source)+r'''
int main(){DeviceState s;for(int j=0;j<3;++j)assert(runAutomaticCsrSchedule(&s,32,1)==0);
assert(s.schedulingAdvanceCount==3&&updates==1&&decisions==1&&prepares==1&&s.csrHeavyTileParticles==2);
s.nCells=0;assert(runAutomaticCsrSchedule(&s,32,1)==0&&s.schedulingAdvanceCount==4&&updates==1);
s.csrHeavyAutoInterval=0;assert(runAutomaticCsrSchedule(&s,32,1)==1&&errors==1);}
''',tmp_path)

@pytest.mark.parametrize('which',['pressure','packing'])
def test_signed_nonfinite_dt_host_protocol_and_real_cuda_errors(tmp_path,which):
    code=body('GpuPressurePipeline.cuh' if which=='pressure' else 'GpuMobilePackingHost.cuh')
    run(PREFIX+r'''
#include <string>
#include <vector>
using PressureTime=double;struct GasHostPolicy{using Time=double;using Weight=double;};
using cudaError_t=int;constexpr int cudaSuccess=0;struct dim3{unsigned x;dim3(int n):x(n){}};
struct DeviceState{int collisionalPressureEnabled=1,jammingPressureEnabled=1,particleWorkGrid=1,nCells=1,nFaces=1,particleBlockThreads=32,csrCellLocalPathEnabled=1,mobilePackingCooperativeGrid=1;DeviceState*deviceState=this;};
std::vector<std::string> calls;int failAt=0,queries=0;double seenDt=0;
void setLastError(const char*,int){}void setLastErrorText(const char*){}
int cudaGetLastError(){return ++queries==failAt?79:0;}
void computeCollisionalPressureFaceFluxKernel(DeviceState*){calls.push_back("flux");}
void scaleCollisionalPressureFaceFluxKernel(DeviceState*){calls.push_back("scale");}
struct Backend{static int limit(DeviceState*,double dt,int,int){seenDt=dt;calls.push_back("limit");return cudaGetLastError();}static int project(DeviceState*,double dt,int,int,int,bool,bool){seenDt=dt;calls.push_back("project");return cudaGetLastError();}static int projectUnsorted(DeviceState*s,double dt,int g,int b){return project(s,dt,g,b,32,0,0);}};
void clearMobilePackingMomentsKernel(DeviceState*){calls.push_back("clear");}
void clearMobilePackingActivityCountsKernel(DeviceState*){calls.push_back("counts");}
void accumulateMobilePackingMomentsKernel(DeviceState*){calls.push_back("accumulate");}
void normalizeMobilePackingMomentsKernel(DeviceState*){calls.push_back("normalize");}
void prepareMobilePackingProjectionKernel(DeviceState*,double dt){seenDt=dt;calls.push_back("prepare");}
void completeMobilePackingProjectionCooperativeKernel(DeviceState*,double){}
int cudaLaunchCooperativeKernel(const void*,dim3,dim3,void**args,int,void*){seenDt=*static_cast<double*>(args[1]);calls.push_back("cooperative");return cudaGetLastError();}
'''+launch_host(code)+r'''
int main(){DeviceState s;
'''+('auto invoke=[&](double dt){return launchCommonPressureKick<Backend>(&s,dt,32,32,false,false);};' if which=='pressure' else 'auto invoke=[&](double dt){return applyMobilePackingProjection(&s,dt,32);};')+r'''
assert(invoke(-.25)==0&&seenDt==-.25);const auto ordinary=calls;assert(!ordinary.empty());
calls.clear();assert(invoke(0.)==0&&calls.empty());
calls.clear();assert(invoke(std::numeric_limits<double>::infinity())==0&&calls==ordinary&&std::isinf(seenDt));
for(int fail=1;fail<=int(ordinary.size());++fail){calls.clear();queries=0;failAt=fail;assert(invoke(-.25)==1&&calls.size()==size_t(fail));}
}
''',tmp_path)

@pytest.mark.parametrize('thermal',[0,1])
def test_sampling_signed_mass_keeps_real_rng_and_weighted_moments(tmp_path,thermal):
    code=body('GpuCollisionPoolSampling.cuh')
    rng=function((COMMON/'operators/normalDevice.cuh').read_text(),'normalPairDevice')
    run(PREFIX+f'\n#define GPU_OPERATOR_THERMAL {thermal}\n'+r'''
#define GPU_RESET_CONTACT_AGE(s,i)
#define GPU_CONTACT_AGE(s,i) ((s).pContactAge[i])
#define GPU_CONTACT_TIME_ZERO 0.
constexpr double OfGreat=1e30;
namespace Foam{namespace gpuThermal{enum{particleWallTransientRebound=2,particleWallTransientDeposit=3};}}
template<class T>T finiteOr(T x,T d){return std::isfinite(x)?x:d;}
template<class T>T clampMin(T x,T l){return max(x,l);}
template<class T>T clampRange(T x,T l,T h){return min(max(x,l),h);}
template<class T>bool finiteDevice(T x){return x==x&&std::abs(x)<1e300;}
template<class T>bool nonFiniteDevice(T x){return !finiteDevice(x);}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
void atomicAdd(double*p,double x){*p+=x;}
struct DeviceState{int particleCapacity=1,nCells=1,pStatus[1]={2},pCellId[1]={0},poissonPoolSampleTargetCount[1]={2},pStuck[1]={0},pStuckFaceId[1]={-1};
 double poissonPoolMass[1]={4},poissonPoolMomX[1]={8},poissonPoolMomY[1]={12},poissonPoolMomZ[1]={16},poissonPoolEnergy[1]={76},thetaDragAlpha[1]={.5};
 double pux[1]={},puy[1]={},puz[1]={},pTheta[1]={-9},pContactAge[1]={},pd[1]={1},pm[1]={},particleDiameterMin=.1,particleDiameterMax=2,particleDiameterFallback=1;
 double pDepositionArea[1]={},pContactDuration[1]={},pContactMaximumArea[1]={},pContactPeakFraction[1]={};
 unsigned long long pRng[1]={41};double poolThermalSumUx[1]={},poolThermalSumUy[1]={},poolThermalSumUz[1]={},poolThermalSumU2[1]={};};
void appendSelectedStuckParticleIndex(DeviceState&,int){}
'''+r'''
// The production Box-Muller operator uses the production uniform generator below.
'''+function(body('operators/snapshotParticleGasCouplingStateKernel.cuh'),'mixSeed')+body('operators/uniform01Device.cuh')+rng+code+r'''
int main(){for(double mass:{-2.,0.,2.,GPU_OPERATOR_THERMAL?1.1e300:2.}){DeviceState s;s.pm[0]=mass;
 unsigned long long rng=s.pRng[0];double z0,z1,z2,z3;normalPairDevice(rng,z0,z1);normalPairDevice(rng,z2,z3);
 sampleOnePoissonPoolParticle(s,0,true);assert(s.pRng[0]==rng&&s.pTheta[0]==0);
 const double sigma=sqrt(1.5),x=sigma*z0,y=sigma*z1,z=sigma*z2;
 assert(s.pux[0]==2+x&&s.puy[0]==3+y&&s.puz[0]==4+z);
 const double weight=GPU_OPERATOR_THERMAL?mass:max(mass,0.);
 assert(s.poolThermalSumUx[0]==weight*(s.pux[0]-2));assert(s.poolThermalSumUy[0]==weight*(s.puy[0]-3));assert(s.poolThermalSumUz[0]==weight*(s.puz[0]-4));
 assert(s.poolThermalSumU2[0]==weight*sqr3(s.pux[0]-2,s.puy[0]-3,s.puz[0]-4));}
 DeviceState s;s.pm[0]=1;s.poissonPoolMomX[0]=std::numeric_limits<double>::infinity();sampleOnePoissonPoolParticle(s,0,true);assert(std::isinf(s.pux[0])&&s.pRng[0]!=41);
 DeviceState fallback;fallback.pm[0]=1;fallback.poissonPoolEnergy[0]=std::numeric_limits<double>::quiet_NaN();sampleOnePoissonPoolParticle(fallback,0,true);assert(fallback.pux[0]==2&&fallback.puy[0]==3&&fallback.puz[0]==4&&fallback.pRng[0]!=41&&fallback.poolThermalSumU2[0]==0);
 DeviceState badMass;badMass.pm[0]=std::numeric_limits<double>::quiet_NaN();sampleOnePoissonPoolParticle(badMass,0,true);assert(badMass.pRng[0]!=41);assert(GPU_OPERATOR_THERMAL?std::isnan(badMass.poolThermalSumU2[0]):badMass.poolThermalSumU2[0]==0);
}
''',tmp_path)

def test_signed_theta_contribution_publishes_actual_energy(tmp_path):
    code=body('GpuPoolParticleContribution.inl')
    run(PREFIX+r'''
struct DeviceState{double pm[1]={2},pux[1]={3},puy[1]={4},puz[1]={0},pd[1]={1},particleDiameterFallback=1;};
template<class T>T finiteOr(T x,T y){return std::isfinite(x)?x:y;}
template<class T>T clampMin(T x,T y){return max(x,y);}
bool nonFiniteDevice(double x){return !std::isfinite(x);}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
#define GPU_POOL_MASS_FALLBACK 0.
#define GPU_POOL_LOAD_LATE_THETA()
double contribution(DeviceState&s,double theta){const int i=0;
'''+code+r'''
return m*specificEnergy;}
int main(){DeviceState s;assert(contribution(s,-1)==22);assert(std::isnan(contribution(s,std::numeric_limits<double>::quiet_NaN())));}
''',tmp_path)

def test_selected_stuck_route_mobile_particle_runs_existing_mobile_correction(tmp_path):
    run(PREFIX+r'''
#define GPU_OPERATOR_THERMAL 1
constexpr double OfSmall=1e-12;
template<class T>T finiteOr(T x,T y){return std::isfinite(x)?x:y;}
template<class T>T clampMin(T x,T y){return max(x,y);}
template<class T>T clampRange(T x,T a,T b){return min(max(x,a),b);}
bool nonFiniteDevice(double x){return !std::isfinite(x)||std::abs(x)>=1e300;}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
struct DeviceState{int particleCapacity=1,nCells=1,pStatus[1]={2},pCellId[1]={0},pStuck[1]={0},poissonPoolSampleTargetCount[1]={2},route=-1;
 double poissonPoolMass[1]={2},poissonPoolMomX[1]={4},poissonPoolMomY[1]={6},poissonPoolMomZ[1]={8},poissonPoolEnergy[1]={35},poolThermalSumUx[1]={1},poolThermalSumUy[1]={2},poolThermalSumUz[1]={3},poolThermalSumU2[1]={20},thetaDragAlpha[1]={.5},pux[1]={3},puy[1]={5},puz[1]={7},pTheta[1]={-9};};
template<bool StuckPath>void finalizeOneThermalizedParticlePath(DeviceState&s,int i,double x,double y,double z,double,double,double){s.route=StuckPath;s.pux[i]=x;s.puy[i]=y;s.puz[i]=z;s.pTheta[i]=0;s.pStatus[i]=1;}
'''+body('GpuCollisionPoolCorrection.cuh')+r'''
int main(){DeviceState a,b;correctOnePoissonThermalizedParticlePath<false>(a,0,true);correctOnePoissonThermalizedParticlePath<true>(b,0,true);
assert(a.route==0&&b.route==0&&b.pStatus[0]==1&&a.pux[0]==b.pux[0]&&a.puy[0]==b.puy[0]&&a.puz[0]==b.puz[0]&&a.pTheta[0]==b.pTheta[0]);
DeviceState large;large.pux[0]=2.2e300;correctOnePoissonThermalizedParticlePath<false>(large,0,true);assert(large.route==0&&large.pStatus[0]==1&&std::isfinite(large.pux[0])&&large.pux[0]>1e300);
DeviceState invalid;invalid.pux[0]=std::numeric_limits<double>::quiet_NaN();correctOnePoissonThermalizedParticlePath<false>(invalid,0,true);assert(invalid.pStatus[0]==1&&std::isnan(invalid.pux[0]));}
''',tmp_path)


def test_dynamic_schedule_population_is_not_an_allocation_admission(tmp_path):
    run(PREFIX+r'''
struct DeviceState{int nCells=1,particleCapacity=2,reductionBlockThreads=32,multiprocessorCount=1,lightResidentBlocksPerSm=1,dynamicHeavyThreshold=-1,csrHeavyTileParticles=-1,cellParticleOffset[2]={0,10},base=5,*preBaseParticleCountDevice=&base;};
'''+body('operators/updateDynamicHeavyPolicyKernel.cuh')+r'''
int main(){DeviceState s;updateDynamicHeavyPolicyKernel(&s,1);assert(s.dynamicHeavyThreshold==32&&s.csrHeavyTileParticles==32);}
''',tmp_path)


def test_contact_area_signed_fields_compute_and_publish(tmp_path):
    code=body('operators/accumulateParticleWallRepresentedContactAreaKernel.cuh')+body('operators/rebuildWallBoundParticleDirectoryKernel.cuh')
    run(PREFIX+r'''
#include "wall/GpuFiniteWallContact.H"
#include "wall/GpuParticleWallContactHeat.H"
#include "wall/GpuColdWallSolidification.H"
namespace Foam {namespace gpuWall {template<class S>int wallBoundDirectoryCount(S&){return 1;}template<class S>int wallBoundDirectoryParticle(S&,int){return 0;}template<class S>void publishWallBoundParticleIndex(S&,int){}}}
template<class T>T finiteOr(T x,T y){return std::isfinite(x)?x:y;}
template<class T>T clampMin(T x,T l){return max(x,l);}
template<class T>T clampRange(T x,T l,T h){return min(max(x,l),h);}
template<class T>bool finiteDevice(T x){return x==x&&std::abs(x)<1e300;}
void atomicAdd(double*p,double v){*p+=v;}
struct DeviceState{int particleCapacity=1,nFaces=1,pStatus[1]={1},pStuckFaceId[1]={0},count=1,*particleCountDevice=&count;unsigned char pStuck[1]={1},particleStuckCandidateMask[1]={0};
 double pDepositionArea[1]={-2},pContactDuration[1]={1},pContactMaximumArea[1]={2},pContactPeakFraction[1]={.5},pColdFrozenArea[1]={},pCold2DFrozenArea[1]={};
 double rhoSolid=3,pd[1]={1},pm[1]={2},particleWallRepresentedContactArea[1]={0},particleWallContactAreaScale[1]={-999},particleWallMaximumCoverage=-.5,magSf[1]={2},pContactAge[1]={.5};};
double particleContactAgeStorage(DeviceState&s,int i,int){return s.pContactAge[i];}
'''+code+r'''
int main(){DeviceState s;const double expected=Foam::gpuThermal::representedDepositionContactArea(s.rhoSolid,s.pd[0],s.pDepositionArea[0],s.pm[0]);
accumulateParticleWallRepresentedContactAreaKernel(&s);assert(std::isfinite(expected)&&s.particleWallRepresentedContactArea[0]==expected);
s.particleStuckCandidateMask[0]=1;s.particleWallRepresentedContactArea[0]=2;finalizeParticleWallContactAreaScaleKernel(&s);assert(s.particleWallContactAreaScale[0]==-.5);
s.pStuck[0]=2;s.pContactDuration[0]=-1;s.pContactPeakFraction[0]=1.5;s.pDepositionArea[0]=-1;s.particleWallRepresentedContactArea[0]=0;
const double area=max(s.pContactMaximumArea[0]*Foam::gpuThermal::normalizedKinematicArea(1.,1.5),0.)+1.;
accumulateParticleWallRepresentedContactAreaKernel(&s);assert(std::abs(s.particleWallRepresentedContactArea[0]-Foam::gpuThermal::representedDepositionContactArea(s.rhoSolid,s.pd[0],area,s.pm[0]))<1e-12);
s.pContactDuration[0]=0;accumulateParticleWallRepresentedContactAreaKernel(&s);bool caught=false;
{DeviceState large;large.pm[0]=1.1e300;large.pDepositionArea[0]=2;large.rhoSolid=1;accumulateParticleWallRepresentedContactAreaKernel(&large);assert(large.particleWallRepresentedContactArea[0]>1e300&&std::isfinite(large.particleWallRepresentedContactArea[0]));}
{DeviceState large;large.particleStuckCandidateMask[0]=1;large.particleWallRepresentedContactArea[0]=1.1e300;large.particleWallMaximumCoverage=1.2e300;finalizeParticleWallContactAreaScaleKernel(&large);assert(large.particleWallContactAreaScale[0]==1);large.particleWallMaximumCoverage=std::numeric_limits<double>::quiet_NaN();finalizeParticleWallContactAreaScaleKernel(&large);assert(large.particleWallContactAreaScale[0]==1);}
s.pStuck[0]=255;caught=false;try{accumulateParticleWallRepresentedContactAreaKernel(&s);}catch(...){caught=true;}assert(caught);}
''',tmp_path)

@pytest.mark.parametrize('fsh',[0,1])
def test_contact_duration_publishes_large_finite_double_for_both_owners(tmp_path,fsh):
    source=(ROOT/'tests/fixtures/particle_tracking/wall_incident_velocity.cpp').read_text().split('void run(const char* scenario',1)[0]
    actual=body('GpuParticleTransport.cuh').replace('evaluateFiniteWallContactImpact','durationFixtureImpact').replace('throw std::runtime_error("production trap");','asm("trap;");')
    inject=r'''
namespace Foam {namespace gpuThermal{
FiniteWallContactImpact durationFixtureImpact(GpuReal t,GpuReal d,GpuReal u,GpuReal c){auto v=evaluateFiniteWallContactImpact(t,d,u,c);v.contactDurationS=1e40;return v;}
}}
'''
    source=source.replace('#include "GpuParticleTransport.cuh"',inject+actual)
    source+=r'''
int main(){DeviceState s;bool trapped=false;try{trackOneParticleLocalFaceWalk(s,0,.1);}catch(const DeviceTrap&){trapped=true;}
 require(!trapped&&s.pStatus[0]==1&&s.pContactDuration[0]==1e40&&s.pStuck[0]!=0,"double duration must complete real contact publication");
}
'''
    source='#define INCIDENT_FSH '+str(fsh)+'\n'+source

    source=source.replace('#include "GpuSommerfeldSticking.H"','#include "wall/GpuSommerfeldSticking.H"').replace('#include "GpuFiniteWallContact.H"','#include "wall/GpuFiniteWallContact.H"').replace('#include "GpuColdWallSolidification.H"','#include "wall/GpuColdWallSolidification.H"')
    run(source,tmp_path)


def test_cooperative_packing_finishes_actual_shorter_correction_list(tmp_path):
    code=function(body('gasNumerics/GpuPackingProjectionCooperative.cuh'),'completeMobilePackingProjectionCooperativeKernel')
    run(PREFIX+r'''
using GpuTime=double;using GpuReal=double;
namespace packingCg {struct grid_group{void sync(){}};grid_group this_grid(){return {};}}
struct DeviceState{int nCells=2,nFaces=1,particleCapacity=2,packingProjectionIterations=1;
 int active=2,correct=0,current=2,next=0,nParticles=2,*mobilePackingActiveCellCount=&active,*mobilePackingCorrectionCellCount=&correct,*mobilePackingFrontierCurrentCount=&current,*mobilePackingFrontierNextCount=&next,*particleCountDevice=&nParticles;
 int mobilePackingFrontierCurrent[2]={0,1},mobilePackingFrontierNext[2]={0,1},mobilePackingActiveCellList[2]={0,1},mobilePackingCorrectionCellList[2]={0,1},mobilePackingActiveCellMask[2]={1,1},mobilePackingCorrectionCellMask[2]={1,1};
 double collisionalPressure[2]={1,2},pressureKickScale[2]={-99,-99};};
int solved=0,flux=0,reconstructed=0,particles=0;
int checkedMobilePackingCount(const DeviceState&s,const int*p){if(*p<0||*p>s.nCells)throw std::runtime_error("extent");return *p;}
void expandMobilePackingActivityFrontierDevice(DeviceState&,int,const int*,int*,int*,int*,int*,int*){}
void solveActiveMobilePackingPressureJacobiDevice(DeviceState&,int i,const int*,const double*old,double*next){++solved;next[i]=old[i]+3;}
void initialiseMobilePackingCorrectionRegionDevice(DeviceState&s,int,int count){s.correct=count;}
void computeMobilePackingFaceCorrectionFluxDevice(DeviceState&s,int,const double*p,double dt){++flux;assert(p[0]==4&&p[1]==5&&dt==-.25);s.correct=1;}
void reconstructMobilePackingVelocityCorrectionDevice(DeviceState&,int i,const int*list){++reconstructed;assert(i==0&&list[i]==0);}
void applyMobilePackingCorrectionToParticleDevice(DeviceState&,int){++particles;}
template<class T>T clampRange(T x,T l,T h){return min(max(x,l),h);}
'''+code+r'''
int main(){DeviceState s;completeMobilePackingProjectionCooperativeKernel(&s,-.25);assert(solved==2&&flux==1&&reconstructed==1&&particles==2);}
''',tmp_path)


def test_descriptor_writes_ignore_capacity_label_with_full_backing(tmp_path):
    run(PREFIX+r'''
enum class CsrReductionTaskSource{fullIndexed};struct CsrReductionTask{int cell,begin,end,source;};
struct DeviceState{int csrHeavyTaskCapacity=1;CsrReductionTask csrReductionTasks[2]={{-99,-99,-99,-99},{-99,-99,-99,-99}};};
'''+body('GpuReductionTaskWrite.cuh')+r'''
int main(){DeviceState s;for(int bad:{-1}){bool caught=false;try{writeCsrReductionTask(s,bad,0,0,1,CsrReductionTaskSource::fullIndexed);}catch(...){caught=true;}assert(caught&&s.csrReductionTasks[0].cell==-99);}
writeCsrReductionTask(s,1,3,0,1,CsrReductionTaskSource::fullIndexed);assert(s.csrReductionTasks[1].cell==3);
writeCsrReductionTask(s,0,0,0,1,CsrReductionTaskSource::fullIndexed);assert(s.csrReductionTasks[0].begin==0&&s.csrReductionTasks[0].end==1);}
''',tmp_path)


def test_payload_gathers_keep_negative_address_and_ignore_capacity_label(tmp_path):
    code=body('GpuParticlePayload.cuh').split('// Both scheduling policies',1)[0]+body('operators/gatherSelectedParticlesKernel.cuh')
    run(PREFIX+r'''
struct DeviceState{int nCells=1,particleCapacity=1,compactCellOffset[2]={0,1},n=1,*compactCountDevice=&n,sortedParticleIndex[1]={},compactPStatus[1]={},pStatus[2]={2,2},pCellId[2]={0,0},compactPCellId[1]={-9};
 double px[2]={1,2},py[2]={3,4},pz[2]={5,6},pux[2]={7,8},puy[2]={9,10},puz[2]={11,12},pT[2]={13,14},pTheta[2]={15,16},pd[2]={17,18},pm[2]={19,20};
 double compactPx[1]={-9},compactPy[1]={-9},compactPz[1]={-9},compactPux[1]={-9},compactPuy[1]={-9},compactPuz[1]={-9},compactPT[1]={-9},compactPTheta[1]={-9},compactPd[1]={-9},compactPm[1]={-9};
 unsigned long long pRng[2]={21,22},pOrigId[2]={23,24},compactPRng[1]={99},compactPOrigId[1]={99};};
#include "GpuCellLocalPrimary.cuh"
template<class T>T clampRange(T x,T a,T b){return min(max(x,a),b);}
'''+code+r'''
int main(){for(int bad:{-1,1})for(int route:{0,1,2}){DeviceState s;s.sortedParticleIndex[0]=s.compactPStatus[0]=bad;bool caught=false;
try{if(route==0)gatherSelectedParticlesKernel(&s);else if(route==1)gatherCellLocalParticlePayloadKernel<false>(&s);else gatherCellLocalParticlePayloadKernel<true>(&s);}catch(...){caught=true;}assert(bad<0?(caught&&s.compactPOrigId[0]==99):(!caught&&s.compactPOrigId[0]==24&&s.compactPux[0]==8&&s.compactPTheta[0]==16&&s.compactPm[0]==20&&s.compactPStatus[0]==1));}}
''',tmp_path)


def test_owned_pool_error_predicate_uses_true_finite_theta(tmp_path):
    code=body('GpuPoolParticleContribution.inl')
    run(PREFIX+r'''
struct DeviceState{double pm[1]={1},pux[1]={0},puy[1]={0},puz[1]={0},pd[1]={1},particleDiameterFallback=1;};
bool finiteDevice(double x){return x==x&&std::abs(x)<1e300;}bool nonFiniteDevice(double x){return !finiteDevice(x);}
double finiteOr(double x,double d){return finiteDevice(x)?x:d;}double clampMin(double x,double d){return max(x,d);}double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
#define GPU_POOL_MASS_FALLBACK 0.
#define GPU_POOL_LOAD_LATE_THETA()
double contribution(DeviceState&s,double theta){const int i=0;
'''+code+r'''
return m*specificEnergy;}
int main(){DeviceState s;assert(contribution(s,1.1e300)==1.5*1.1e300);assert(std::isnan(contribution(s,std::numeric_limits<double>::quiet_NaN())));}
''',tmp_path)


def test_owned_drag_error_predicate_preserves_actual_large_finite_momentum(tmp_path):
    run(PREFIX+r'''
#define GPU_OPERATOR_TINY(x) (x)
struct DeviceState{double gravityX=1.4e301,gravityY=-1.2e301,gravityZ=4,gradPx[1]={},gradPy[1]={},gradPz[1]={},invRhoSolid=1,pux[1]={-99},puy[1]={-99},puz[1]={-99};};
bool finiteDevice(double x){return x==x&&std::abs(x)<1e300;}bool nonFiniteDevice(double x){return !finiteDevice(x);}
double finiteOr(double x,double d){return finiteDevice(x)?x:d;}double clampMin(double x,double d){return max(x,d);}
void drag(DeviceState&s){const int i=0,c=0;const double dt=.25,invTauDrag=0,ugx=0,ugy=0,ugz=0,ux=0,uy=0,uz=0;
'''+body('operators/ParticleDragMomentum.inl')+r'''
}
int main(){DeviceState s;drag(s);assert(s.pux[0]==.25*s.gravityX&&s.puy[0]==.25*s.gravityY&&s.puz[0]==1);s.gravityX=std::numeric_limits<double>::quiet_NaN();drag(s);assert(std::isnan(s.pux[0]));s.gravityX=std::numeric_limits<double>::max();s.invRhoSolid=-std::numeric_limits<double>::max();s.gradPx[0]=2;drag(s);assert(std::isinf(s.pux[0]));}
''',tmp_path)


def test_pressure_existing_convex_limiter_preserves_bounded_zero_fallback(tmp_path):
    run(PREFIX+r'''
#include "GpuPressureConvexAlgebra.cuh"
using PressureReal=double;using PressureTime=double;
bool finiteDevice(double x){return x==x&&std::abs(x)<1e300;}
struct DeviceState{double momRhoP[1]={1},momRhoUPx[1]={},momRhoUPy[1]={},momRhoUPz[1]={},momRhoEP[1]={1.1e301},V[1]={1},cellLength[1]={1},thetaMin=.1,pressureKickFraction=.2,epsSMin=1e-12,rhoSolid=3;
int nFaces=1,cellPlaneCount[1]={1},cellPlaneStart[1]={0},cellFaceId[1]={0},faceOwner[1]={0},faceNeighbour[1]={-1};double solidPressurePhiMomX[1]={},solidPressurePhiMomY[1]={},solidPressurePhiMomZ[1]={},solidPressurePhiEnergy[1]={1};};
'''+function(body('GpuPressureKickAccumulation.cuh'),'pressureLocalConvexScale')+r'''
int main(){DeviceState s;assert(pressureLocalConvexScale(s,0,1)==0);s.momRhoEP[0]=std::numeric_limits<double>::quiet_NaN();assert(pressureLocalConvexScale(s,0,1)==0);}
''',tmp_path)


def test_development_probe_mismatch_mask_is_reported_without_aborting_completed_work(tmp_path):
    run(PREFIX+r'''
#include <cstdio>
#include <string>
using cudaError_t=int;constexpr int cudaSuccess=0,ProbeStageCount=2,ProbeMaxOccurrences=2;
using DevelopmentProbeStage=int;const char* developmentProbeStageNames[2]={"one","two"};
constexpr bool developmentProbeIncludesScheduling=false;
char lastError[128]={};int writes=0,errors=0,eventCount=0,failAt=0,observedMask=0,collectError=0,writeError=0;std::string observedStatus;double observedDt=0,observedTime=0;
struct DeviceState{int nCells=2,particlesMayBePresent=1,particleCapacity=3;};
struct DevelopmentProbeSample{unsigned long long step=0;double simulationTime=0,dt=0;std::string status="ok",errorStage,errorMessage;int nCells=0,particlePath=0,particleCapacity=0,badFieldMask=0;};
struct Probe{unsigned long long advanceIndex=0,interval=1;int stageOccurrenceCount[2]={},totalStartEvent=1,totalStopEvent=2,stageStartEvents[2][2]={},stageStopEvents[2][2]={};bool stageOccurrenceExecuted[2][2]={},failOnNonFinite=true;}developmentProbe;
bool developmentProbeEnabled(){return true;}int cudaEventRecord(int,int){return ++eventCount==failAt?77:0;}
void setLastError(const char*,int){++errors;}void setLastErrorText(const char*){++errors;}
int collectDevelopmentProbeSample(DeviceState*,bool,DevelopmentProbeSample&s){s.badFieldMask=4;s.status="state_invalid";return collectError;}
int writeDevelopmentProbeSample(const DevelopmentProbeSample&s){++writes;observedMask=s.badFieldMask;observedStatus=s.status;observedDt=s.dt;observedTime=s.simulationTime;return writeError;}
'''+body('GpuDevelopmentAdvanceProbe.cuh')+r'''
int main(){DeviceState s;{DevelopmentAdvanceProbe p(&s,.25,4.);p.enter(0);assert(p.leave(0)==0);assert(p.finish(true)==0);}
assert(writes==1&&observedMask==4&&observedStatus=="state_invalid"&&errors==0&&developmentProbe.stageOccurrenceCount[0]==1&&observedDt==.25&&observedTime==4.);
developmentProbe.failOnNonFinite=false;{DevelopmentAdvanceProbe p(&s,.5,5.);assert(p.finish(true)==0);}assert(writes==2&&observedMask==4&&observedStatus=="state_invalid"&&observedDt==.5&&observedTime==5.);
collectError=1;{DevelopmentAdvanceProbe p(&s,.25,4.);assert(p.finish(true)==1);}collectError=0;assert(writes==3&&observedStatus=="error");
writeError=1;{DevelopmentAdvanceProbe p(&s,.25,4.);assert(p.finish(true)==1);}writeError=0;assert(writes==4);
{DevelopmentAdvanceProbe p(&s,.25,4.25);failAt=eventCount+1;assert(p.finish(true)==1);}assert(errors==1&&writes==5&&observedStatus=="error");}
''',tmp_path)


def test_development_probe_relative_log_uses_real_io(tmp_path):
    run(PREFIX+r'''
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
using cudaError_t=int;constexpr int cudaSuccess=0,ProbeStageCount=2,ProbeMaxOccurrences=2;
struct DeviceState{int nCells=2;};struct DevelopmentProbeDeviceSummary{int count=0;};
enum class DevelopmentProbeMode{timing,full};
struct Probe{DeviceState*owner=nullptr;DevelopmentProbeMode mode=DevelopmentProbeMode::timing;std::string modeName,logPath,runId,variant;unsigned long long interval=1;bool failOnNonFinite=false;int pid=0,totalStartEvent=0,totalStopEvent=0,stageStartEvents[2][2]={},stageStopEvents[2][2]={};FILE*log=nullptr;DevelopmentProbeDeviceSummary*deviceSummary=nullptr;std::vector<int>occupancy;}developmentProbe;
char lastError[512]={};int errors=0,events=0;
void setLastErrorText(const char*t){++errors;std::snprintf(lastError,sizeof lastError,"%s",t);}void setLastError(const char*t,int){setLastErrorText(t);}
void shutdownDevelopmentProbe(){if(developmentProbe.log)std::fclose(developmentProbe.log);developmentProbe=Probe{};}
bool developmentProbeBoolean(const char*t){return t&&*t=='1';}std::string developmentProbePathForPid(const char*t){return t;}
bool developmentProbeFullValidation(){return developmentProbe.mode==DevelopmentProbeMode::full;}
int writeDevelopmentProbeHeader(){return std::fprintf(developmentProbe.log,"actual-header\n")<0?1:0;}
int cudaEventCreate(int*p){*p=++events;return 0;}int cudaMalloc(void**p,size_t n){*p=std::malloc(n);return *p?0:1;}
'''+body('GpuDevelopmentProbeInit.cuh')+r'''
int main(){DeviceState s;setenv("UGKP_DEV_PROBE_MODE","timing",1);setenv("UGKP_DEV_PROBE_LOG","probe-relative.tsv",1);unsetenv("UGKP_DEV_PROBE_INTERVAL");
assert(initialiseDevelopmentProbe(&s)==0);assert(developmentProbe.owner==&s&&developmentProbe.occupancy.size()==2&&events==10);shutdownDevelopmentProbe();
FILE*f=std::fopen("probe-relative.tsv","r");assert(f);char line[64]={};assert(std::fgets(line,64,f));std::fclose(f);assert(std::string(line)=="actual-header\n");std::remove("probe-relative.tsv");
char cwd[4096];assert(getcwd(cwd,sizeof cwd));const std::string absolute=std::string(cwd)+"/absolute.tsv";setenv("UGKP_DEV_PROBE_LOG",absolute.c_str(),1);assert(initialiseDevelopmentProbe(&s)==0);shutdownDevelopmentProbe();assert(std::remove(absolute.c_str())==0);
setenv("UGKP_DEV_PROBE_MODE","unknown",1);assert(initialiseDevelopmentProbe(&s)==1&&std::strstr(lastError,"MODE"));setenv("UGKP_DEV_PROBE_MODE","timing",1);
unsetenv("UGKP_DEV_PROBE_LOG");assert(initialiseDevelopmentProbe(&s)==1&&std::strstr(lastError,"LOG"));
setenv("UGKP_DEV_PROBE_LOG","missing/subdirectory/probe.tsv",1);assert(initialiseDevelopmentProbe(&s)==1);assert(std::strstr(lastError,"cannot open"));
setenv("UGKP_DEV_PROBE_LOG","probe-relative.tsv",1);setenv("UGKP_DEV_PROBE_INTERVAL","0",1);assert(initialiseDevelopmentProbe(&s)==1&&std::strstr(lastError,"INTERVAL"));
setenv("UGKP_DEV_PROBE_INTERVAL","unparseable",1);assert(initialiseDevelopmentProbe(&s)==1&&std::strstr(lastError,"INTERVAL"));
}
''',tmp_path)
