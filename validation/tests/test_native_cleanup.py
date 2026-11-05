"""Actual native bodies with host CUDA/transfer boundaries; not GPU execution."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import re
import subprocess

ROOT = SOURCE_ROOT


CPP_NESTING_TOKENS = re.compile(
    r'''R"(?P<delimiter>[^\s()\\]{0,16})\(.*?\)(?P=delimiter)"|"(?:\\.|[^"\\])*"|(?<!\w)(?:u8|u|U|L)?'(?:\\.|[^'\\])*'|//[^\n]*|/\*.*?\*/|[{}]''',
    re.S,
)


def block(text, signature):
    start = text.index(signature)
    depth = 0
    for token in CPP_NESTING_TOKENS.finditer(text, start):
        value = token.group()
        if value == '{':
            depth += 1
        elif value == '}':
            depth -= 1
            if depth == 0:
                return text[start:token.end()]
    raise ValueError('unterminated C++ body for '+signature)


def test_block_keeps_cpp_literals_and_comment_braces_out_of_nesting():
    expected=r'''void target(const char* defaultValue="{") {
    const char* value="} escaped \" {";
    const char* raw=R"body( } \" { )body";
    const char brace='}';
    /* } { } */
    // }
    if (defaultValue && value && raw && brace) { consume(); }
}'''
    assert block('void prior() {}\n'+expected+'\nvoid after() {}','void target')==expected
    try:
        block('void unfinished() {','void unfinished')
    except ValueError:
        pass
    else:
        raise AssertionError('unterminated C++ body accepted')


def run_cpp(tmp_path, code, includes=()):
    source = tmp_path / 'probe.cpp'
    source.write_text(code)
    binary = tmp_path / 'probe'
    build = subprocess.run(['g++', '-std=c++17', '-O2', '-ffp-contract=off',
                            '-I'+str(ROOT), '-I'+str(ROOT/'common'),
                            *['-I'+str(p) for p in includes], str(source), '-o', str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


CUDA = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>
using cudaError_t=int;using cudaStream_t=void*;
constexpr int cudaSuccess=0,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
int copies=0,h2d=0,d2h=0,allocations=0,memsets=0,syncs=0,launches=0;
int failCopy=0,failMemset=0,failSync=0,failLaunch=0,failAlloc=0;size_t copiedBytes=0,zeroedBytes=0;
const char* cudaGetErrorString(int){return "injected CUDA error";}
int cudaMemcpy(void* d,const void* s,size_t n,int kind){++copies;if(copies==failCopy)return 1;copiedBytes+=n;if(kind==1)++h2d;else ++d2h;std::memcpy(d,s,n);return 0;}
int cudaMemset(void* d,int v,size_t n){++memsets;if(memsets==failMemset)return 1;zeroedBytes+=n;std::memset(d,v,n);return 0;}
int cudaMalloc(void** p,size_t n){++allocations;if(allocations==failAlloc)return 1;*p=std::malloc(n);return *p?0:1;}
int cudaFree(void* p){std::free(p);return 0;}
int cudaDeviceSynchronize(){++syncs;return syncs==failSync?1:0;}
int cudaGetLastError(){return launches==failLaunch?1:0;}
#define __global__
struct Dim {int x=1;};Dim gridDim,blockDim;struct Index {int x=0;};Index blockIdx,threadIdx;
void counters(){copies=h2d=d2h=allocations=memsets=syncs=launches=0;failCopy=failMemset=failSync=failLaunch=failAlloc=0;copiedBytes=zeroedBytes=0;}
'''


def test_saved_velocity_actual_include_prefix_bits_and_copy_failures(tmp_path):

    code = CUDA + r'''
struct DeviceState {int particleCapacity=5,count=4;int* particleCountDevice=&count;double a[5]{},b[5]{},c[5]{};double* puxOld=a;double* puyOld=b;double* puzOld=c;};
DeviceState* asState(void* p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState* p,const char*){return !p;}void setLastErrorText(const char*){}
template<class T>int copyToDevice(T* d,const T* s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),1):0;}
template<class T>int copyToHost(T* d,const T* s,size_t n,const char*){return n?cudaMemcpy(d,s,n*sizeof(T),2):0;}
#include "common/GpuParticleSavedVelocityRestart.inl"
int main(){
 for(int owner=0;owner<2;++owner){ // Both native owners use this exact include and double fields.
  for(int count:{0,1,3,5}){DeviceState s;for(int i=0;i<5;++i)s.a[i]=s.b[i]=s.c[i]=-99;
   double x[5]={1+std::ldexp(1.,-30),-2,3,4,5},y[5]={6,7,8,9,10},z[5]={11,12,13,14,15};counters();
   assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,count,x,y,z)==0);
   assert(d2h==0&&h2d==(count?3:0)&&copiedBytes==size_t(count)*24);
   for(int i=0;i<5;++i){assert(s.a[i]==(i<count?x[i]:-99));assert(s.b[i]==(i<count?y[i]:-99));assert(s.c[i]==(i<count?z[i]:-99));}assert(s.count==4);
   double xx[5]={-8,-8,-8,-8,-8},yy[5]={-8,-8,-8,-8,-8},zz[5]={-8,-8,-8,-8,-8};counters();
   assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,count,xx,yy,zz)==0);assert(d2h==(count?3:0)&&h2d==0&&copiedBytes==size_t(count)*24);
   for(int i=0;i<5;++i){assert(xx[i]==(i<count?x[i]:-8));assert(yy[i]==(i<count?y[i]:-8));assert(zz[i]==(i<count?z[i]:-8));}
  }
  for(int failed=1;failed<=3;++failed){DeviceState s;double x[1]={1},y[1]={2},z[1]={3};counters();failCopy=failed;
   assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,1,x,y,z)==1);assert(copies==failed);assert(s.a[0]==(failed>1?1:0));assert(s.b[0]==(failed>2?2:0));assert(s.c[0]==0);
   s.a[0]=1;s.b[0]=2;s.c[0]=3;x[0]=y[0]=z[0]=-1;counters();failCopy=failed;
   assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,1,x,y,z)==1);assert(copies==failed);assert(x[0]==(failed>1?1:-1));assert(y[0]==(failed>2?2:-1));assert(z[0]==-1);
  }
  DeviceState s;double x=std::numeric_limits<double>::infinity(),y=-0.,z=std::numeric_limits<double>::quiet_NaN();counters();
  assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,1,&x,&y,&z)==0);assert(std::isinf(s.a[0])&&std::signbit(s.b[0])&&std::isnan(s.c[0]));
  counters();assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,0,nullptr,nullptr,nullptr)==0&&copies==0);
  double xx[6]{},yy[6]{},zz[6]{};
  for(int count:{-1,6}){assert(ugkwpGpuResidentStrictUploadParticleSavedVelocity(&s,count,xx,yy,zz)!=0&&copies==0);
   assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,count,xx,yy,zz)!=0&&copies==0);}
  assert(ugkwpGpuResidentStrictDownloadParticleSavedVelocity(&s,1,nullptr,&y,&z)!=0&&copies==0);
 }
 puts("saved velocity: exact prefix and partial-copy failures; no certificates");
}
'''
    run_cpp(tmp_path, code)


def radiation_code(text=None):
    if text is None:
        text = (ROOT/'common/thermal/GpuResidentStrict.cu').read_text()
    legacy = 'struct ParticleRadiationValidationError' in text
    record = block(text, 'struct ParticleRadiationValidationError')+';\n' if legacy else ''
    code = CUDA + r'''
#include "GpuPrecisionTypes.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
using std::isfinite;
namespace Foam {namespace gpuThermal {constexpr int particleWallMobile=0;}}
''' + record + r'''
struct DeviceState {
 int count=1,nCells=1,particleWorkGrid=1,particleBlockThreads=1;int* particleCountDevice=&count;
 int pStatus[1]={1},pStuck[1]={0},pCellId[1]={0};unsigned long long pOrigId[1]={19};
 double pT[1]={300},pThermalEnthalpyResidual[1]={.0002};double TpMin=200,TpMax=4000;
 bool particlesMayBePresent=true;DeviceState* deviceState=this;
 Foam::gpuThermal::RadiationAffineTemperatureUpdate* particleRadiationAffineUpdate=nullptr;
''' + ('ParticleRadiationValidationError* particleRadiationValidationError=nullptr;\n' if legacy else '') + r'''
};
char lastError[2048];DeviceState* asState(void* p){return static_cast<DeviceState*>(p);}
int validateState(DeviceState* p,const char*){return !p;}
void setLastErrorText(const char* s){std::snprintf(lastError,sizeof(lastError),"%s",s);}
void setLastError(const char* s,int){setLastErrorText(s);}
template<class T>int allocate(T*& p,size_t n,const char*){return cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T));}
template<class T>void release(T*& p){cudaFree(p);p=nullptr;}
template<class T>int copyToDevice(T* d,const T* s,size_t n,const char*){return cudaMemcpy(d,s,n*sizeof(T),1);}
template<class T>int copyToHost(T* d,const T* s,size_t n,const char*){return cudaMemcpy(d,s,n*sizeof(T),2);}
int atomicCAS(int* p,int expected,int value){int old=*p;if(old==expected)*p=value;return old;}
ugkwp::ParticleCaloric particleCaloricDevice(const DeviceState&,int){return {};}
double particleEffectiveTemperatureDevice(const DeviceState&s,int i){return ugkwp::effectiveParticleTemperature(ugkwp::ParticleThermalCarry<double>{s.pT[i],s.pThermalEnthalpyResidual[i]},particleCaloricDevice(s,i));}
'''
    if legacy:
        code += block(text, 'enum ParticleRadiationValidationCode')+';\n'
        code += block(text, '__global__ void validateParticleRadiationAffineTemperatureKernel')+'\n'
    code += block(text, '__global__ void applyParticleRadiationAffineTemperatureKernel')+'\n'
    api = block(text, 'extern "C" int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature')
    api = re.sub(r'(\w+)\s*<<<.*?>>>', r'(++launches, \1)', api, flags=re.S)
    return (code+api).replace('asm("trap;");','throw std::runtime_error("device error");')


def test_radiation_single_actual_apply_counts_and_extensions(tmp_path):
    code = radiation_code()+r'''
int main(){
 DeviceState s;Foam::gpuThermal::RadiationAffineTemperatureUpdate update{1.,.25};counters();
 {DeviceState empty;failAlloc=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&empty,1,&update)!=0);assert(empty.particleRadiationAffineUpdate==nullptr&&copies==0&&launches==0);}
 counters();
 assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&update)==0);
 assert(allocations==1&&h2d==1&&d2h==0&&memsets==0&&launches==1&&syncs==1&&copiedBytes==16);
 for(auto u:{Foam::gpuThermal::RadiationAffineTemperatureUpdate{1.1,0}, {1.,4000.}, {-.1,400.}}){
  s.pT[0]=300;s.pThermalEnthalpyResidual[0]=.0002;auto expected=ugkwp::ParticleThermalCarry<double>{300,.0002};
  auto result=ugkwp::publishAffineRadiationThermalHeat(expected,u.scale,u.offset,s.TpMin,s.TpMax,particleCaloricDevice(s,0));assert(result.valid);
  counters();assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&u)==0);
  assert(s.pT[0]==expected.temperature&&s.pThermalEnthalpyResidual[0]==expected.specificEnthalpyResidual);
  assert(allocations==0&&h2d==1&&d2h==0&&launches==1&&syncs==1);
 }
 s.particlesMayBePresent=false;counters();assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&update)==0&&copies==0&&launches==0);s.particlesMayBePresent=true;
 counters();failCopy=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&update)!=0&&launches==0);
 counters();failLaunch=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&update)!=0&&syncs==0);
 counters();failSync=1;assert(ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(&s,1,&update)!=0&&syncs==1);
 free(s.particleRadiationAffineUpdate);puts("radiation: one actual apply and complete formerly-refused transforms");
}
'''
    run_cpp(tmp_path, code)


def test_ledger_reset_actual_owner_no_download_and_partial_errors(tmp_path):
    text = (ROOT/'applications/CHT/gpu/GpuWallEnergy64.cu').read_text()
    new = 'ugkwpGpuResidentStrictResetGasWallEnergy' in text
    text = re.sub(r'(\w+)\s*<<<.*?>>>', r'(++launches, \1)', text, flags=re.S)
    (tmp_path/'cuda_runtime.h').write_text('#pragma once\n')
    code = CUDA+text+r'''
// Configuration is not exercised by this reset-only owner fixture.
extern "C" int ugkpInvalidateGasWallEnergyGraph(void*){assert(false);return 1;}
extern "C" int ugkpPrepareParticleWallEnergyRestore(void*){assert(false);return 1;}
int main(){
 GpuWallEnergyState s;s.deviceState=&s;s.nFaces=5;s.nInternalFaces=2;s.nEnabledFaces=1;s.particleWallHeatTransferEnabled=1;
 double g[7],d[7],r[7],out[5],out2[5];unsigned char mask[5]={};s.gasWallEnergy=g+1;s.particleWallDepositedEnergy=d+1;s.particleWallReflectedEnergy=r+1;s.gasWallEnergyMask=mask;
 auto seed=[&](){for(int i=0;i<7;++i){g[i]=i+10;d[i]=i+20;r[i]=-i-30;}};
 seed();counters();assert(RESET_GAS==0);assert(RESET_PARTICLE==0);
 assert(d2h==0&&h2d==0&&allocations==0&&memsets==3&&syncs==2&&zeroedBytes==120);
 for(int i=1;i<=5;++i)assert(g[i]==0&&d[i]==0&&r[i]==0);
 assert(g[0]==10&&g[6]==16&&d[0]==20&&d[6]==26&&r[0]==-30&&r[6]==-36);
 for(int fail=1;fail<=2;++fail){seed();counters();failMemset=fail;assert(RESET_PARTICLE!=0);assert(memsets==fail&&syncs==0);for(int i=1;i<=5;++i){assert(d[i]==(fail==2?0:i+20));assert(r[i]==-i-30);}}
 seed();counters();failSync=1;assert(RESET_GAS!=0);for(int i=1;i<=5;++i)assert(g[i]==0);assert(syncs==1);
 seed();counters();failMemset=1;assert(RESET_GAS!=0&&syncs==0);for(int i=1;i<=5;++i)assert(g[i]==i+10);
 seed();counters();s.particleWallHeatTransferEnabled=0;assert(RESET_PARTICLE==0);assert(memsets==2&&syncs==1&&d[1]==0&&r[1]==0);
 puts("ledger: full-face reset only, real partial failure, no data copies");
}
'''
    code = code.replace('RESET_GAS', 'ugkwpGpuResidentStrictResetGasWallEnergy(&s)' if new else 'ugkwpGpuResidentStrictDownloadAndResetGasWallEnergy(&s,5,out)')
    code = code.replace('RESET_PARTICLE', 'ugkwpGpuResidentStrictResetParticleWallHeatLedgers(&s)' if new else 'ugkwpGpuResidentStrictDownloadAndResetParticleWallHeatLedgers(&s,5,out,out2)')
    run_cpp(tmp_path, code, [tmp_path, ROOT/'applications/CHT/gpu'])


def test_actual_frontend_reset_wrappers_preserve_unconsumed_channels(tmp_path):
    text = (ROOT/'applications/CHT/gpu/GpuResidentStrict.H').read_text()
    methods = '\n'.join(block(text, '    void '+name+'()') for name in
                        ['resetGasWallEnergy', 'resetParticleWallHeatLedgers'])
    run_cpp(tmp_path, CUDA+r'''
int gasCalls=0,particleCalls=0,failGas=0,failParticle=0;
int ugkwpGpuResidentStrictResetGasWallEnergy(void*){++gasCalls;return failGas;}
int ugkwpGpuResidentStrictResetParticleWallHeatLedgers(void*){++particleCalls;return failParticle;}
struct Resident {
 void* handle_=this;bool particleWallHeatTransferEnabled_=false;std::vector<int> gasWallEnergyFaceIds_;
 void requireInitialised(const char*){assert(handle_);}
 void check(int error,const char*){if(error)throw std::runtime_error("native error");}
'''+methods+r'''
};
int main(){Resident r;r.resetGasWallEnergy();r.resetParticleWallHeatLedgers();assert(gasCalls==0&&particleCalls==0);
 r.gasWallEnergyFaceIds_={1,4};r.resetGasWallEnergy();r.resetParticleWallHeatLedgers();assert(gasCalls==1&&particleCalls==0);
 r.particleWallHeatTransferEnabled_=true;r.resetParticleWallHeatLedgers();assert(particleCalls==1);
 failGas=1;bool failed=false;try{r.resetGasWallEnergy();}catch(const std::runtime_error&){failed=true;}assert(failed);
 failParticle=1;failed=false;try{r.resetParticleWallHeatLedgers();}catch(const std::runtime_error&){failed=true;}assert(failed);
 // Retained allocated inventory may exist when these selectors are disabled.
 r.gasWallEnergyFaceIds_.clear();r.particleWallHeatTransferEnabled_=false;int g=gasCalls,p=particleCalls;r.resetGasWallEnergy();r.resetParticleWallHeatLedgers();assert(g==gasCalls&&p==particleCalls);
 assert(copies==0&&memsets==0&&allocations==0);puts("actual frontend wrappers preserve selected-channel/no-op/error semantics");
}
''')


def test_first_slice_preserves_coupler_order_and_real_download_consumers():
    text = (ROOT/'applications/CHT/thermal/GpuSolidThermalCoupler.C').read_text()
    commit = text[text.index('        commitPhaseStarted = true;'):]
    assert commit.index('applyParticleRadiationEnergy') < commit.index('resetGasWallEnergy()')
    assert commit.index('resetGasWallEnergy()') < commit.index('resetParticleWallHeatLedgers()')
    assert commit.index('resetParticleWallHeatLedgers()') < commit.index('gasConvectiveWallHeatFlux->primitiveFieldRef()')
    assert commit.index('gasConvectiveWallHeatFlux->primitiveFieldRef()') < commit.index('publishSolidCandidate')
    assert 'consumedGasWallEnergy' not in commit and 'consumedParticleDepositedEnergy' not in commit
    header = (ROOT/'applications/CHT/gpu/GpuResidentStrict.H').read_text()
    assert 'void downloadAndResetGasWallEnergy(scalarField& configuredEnergy)' in header
    assert 'void downloadAndResetParticleWallHeatLedgers' in header
    assert 'savedVelocityBound' not in header
    assert 'Legacy transient restart lacks saved impact velocity' not in header
    assert 'scalarField puxOld(nParticles,0.0), puyOld(nParticles,0.0), puzOld(nParticles,0.0);' in header
    native = (ROOT/'common/thermal/GpuResidentStrict.cu').read_text()
    assert 'particleRadiationValidationError' not in native
    assert 'validateParticleRadiationAffineTemperatureKernel' not in native
