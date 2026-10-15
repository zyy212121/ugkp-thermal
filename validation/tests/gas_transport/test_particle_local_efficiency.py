"""Execute production local thermal functions and frozen same-input references.

All counters live in temporary test headers/function copies. They count actual
executed arithmetic or loop visits, with no production instrumentation. Host
execution and native CUDA compilation are separate checks.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import pytest
from local_heat_efficiency_reference import FINITE_REFERENCE, COLD_REFERENCE, PAIRED_REFERENCE

ROOT = SOURCE_ROOT


def function_slice(path, start, end):
    source = path.read_text()
    first = source.index(start)
    return source[first:source.index(end, first)]


def compile_run(tmp_path, code, flags=()):
    source = tmp_path / 'local_efficiency.cpp'
    source.write_text(code)
    exe = tmp_path / 'local_efficiency'
    build = subprocess.run(['g++', '-std=c++17', '-O2', *flags,
        '-I' + str(tmp_path), '-I' + str(ROOT / 'common'),
        str(source), '-o', str(exe)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    result = [json.loads(line) for line in run.stdout.splitlines()]
    print(json.dumps(result, sort_keys=True))
    return result


@pytest.mark.parametrize('contract', ['off', 'fast'])
def test_finite_contact_exact_fixed_point_reduces_evaluations(tmp_path, contract):
    actual = function_slice(ROOT / 'common/GpuParticleThermalConsumers.H',
        'template<class Storage, class Caloric>', '\ntemplate<class Storage, class Caloric>')
    actual = actual.replace('UGKWP_THERMAL_CONSUMER_HD', '').replace(
        'publishFiniteWallThermalHeat', 'stagnationFiniteHeat').replace(
        'const double middle =', '++stagnationIterations; const double middle =')
    reference = FINITE_REFERENCE.replace('const double middle =',
        '++referenceIterations; const double middle =')
    code = r'''
#include "GpuParticleThermalConsumers.H"
#include "GpuParticleCaloric.H"
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <cstdlib>
#include <iomanip>
namespace ugkwp { unsigned long long referenceIterations=0,stagnationIterations=0;
''' + reference + actual + '}\n' + FINITE_BODY
    result = compile_run(tmp_path, code, ['-ffp-contract=' + contract])[0]
    assert result['mismatches'] == result['atomic_failure_count'] == 0
    assert result['cases'] == 422750
    assert result['candidate_bisections'] < result['reference_bisections']
    assert result['cases_reduced'] > 0


def test_only_eight_node_candidate_prepares_equilibrium_increment(tmp_path):
    caloric = (ROOT / 'common/GpuParticleCaloric.H').read_text()
    caloric = caloric.replace('namespace ugkwp {',
        'static unsigned long long incrementCount=0;\nnamespace ugkwp {', 1)
    caloric = caloric.replace('double enthalpyIncrement(double t,double deltaT)const {',
        'double enthalpyIncrement(double t,double deltaT)const { ++incrementCount;')
    (tmp_path / 'GpuParticleCaloric.H').write_text(caloric)
    actual = function_slice(ROOT / 'common/GpuPairedParticleHeat.cuh',
        '__device__ __forceinline__ PairedParticleHeatCandidate pairedParticleHeatCandidate',
        '// Called from').replace('__device__', '').replace('__forceinline__', 'inline')
    actual = actual.replace('pairedParticleHeatCandidate', 'actualCandidate')
    bound = function_slice(ROOT / 'common/wall/GpuColdWallThermalCarry.cuh',
        'UGKWP_COLD_CARRY_HD inline double coldWallUniformCoolingBound',
        'UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication').replace('UGKWP_COLD_CARRY_HD', '')
    code = PAIRED_PREFIX + 'namespace ugkwp {\n' + bound + '}\n' + (
        'struct PairedParticleHeatCandidate {double capacity, weightedCapacity, specificHeat;};\n') + (
        PAIRED_REFERENCE + actual + PAIRED_BODY)
    result = compile_run(tmp_path, code, [])[0]
    assert result['storage_bits'] == 64
    assert result['cases'] == 75000 and result['mismatches'] == 0
    assert result['actual_enthalpyIncrement_calls'] < result['baseline_enthalpyIncrement_calls']


def test_cold_uniform_nonzero_combines_read_only_scans(tmp_path):
    actual = function_slice(ROOT / 'common/wall/GpuColdWallThermalCarry.cuh',
        'UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication publishColdWallUniformSpecificHeat',
        'template<class Parameters>').replace('UGKWP_COLD_CARRY_HD', '')
    def counted(source):

        source = re.sub(r'(for\(int j=0;j<count;\+\+j\)\s*\{)', r'\1 ++scanCount;', source)
        source = source.replace('for(int j=0;j<count;++j)nodes[j]=',
            'for(int j=0;j<count;++j){++scanCount;nodes[j]=')
        source = source.replace('residual=next;', '}\n residual=next;')
        return source
    floor = function_slice(ROOT / 'common/wall/GpuColdWallThermalCarry.cuh',
        'UGKWP_COLD_CARRY_HD inline float coldWallStoredEnthalpyFloor',
        '// Leave only').replace('UGKWP_COLD_CARRY_HD', '')
    code = r'''
#include <cmath>
#include <cfloat>
#include <cstring>
#include <limits>
#include <random>
#include <iostream>
#include <array>
struct ColdWallHeatPublication {bool valid;double acceptedSpecificHeat;};
static unsigned long long scanCount=0;
''' + floor + counted(COLD_REFERENCE) + counted(actual) + COLD_BODY
    comparison, counts = compile_run(tmp_path, code)
    assert comparison['cases'] == 300000 and comparison['mismatches'] == 0
    assert counts['zero_actual_node_visits'] == counts['zero_reference_node_visits'] > 0
    assert 3 * counts['nonzero_actual_node_visits'] == 2 * counts['nonzero_reference_node_visits'] > 0


def native_compile(tmp_path, code):
    nvcc = os.environ.get('NVCC') or shutil.which('nvcc')
    if not nvcc:
        pytest.skip('CUDA compiler unavailable; no CUDA runtime is claimed')
    source = tmp_path / 'local_efficiency.cu'
    source.write_text(code)
    command = [nvcc, '-std=c++17', '-arch=sm_89',
        *([]),
        '-I' + str(ROOT / 'common'), '-c', str(source),
        '-o', str(tmp_path / 'local_efficiency.o')]
    build = subprocess.run(command, capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    print(json.dumps({'native_compile_command': command, 'storage_bits': 64,
        'object_bytes': (tmp_path / 'local_efficiency.o').stat().st_size,
        'cuda_execution': 'NOT_RUN'}, sort_keys=True))


def test_shared_wall_local_functions_compile_as_cuda(tmp_path):
    code = r'''
#include <cuda_runtime.h>
#include "gasWall/WallModel.H"
#include "GpuParticleThermalConsumers.H"
#include "wall/GpuColdWallThermalCarry.cuh"
template<int N,int M> __global__ void wallRows(
 ugkwp::gaswall::WallInput<double,N>* in,ugkwp::gaswall::WallModelConfig<double>* c,
 ugkwp::gaswall::WallWorkspace<double,N,M>* w,double* out){
 ugkwp::gaswall::detail::LayerContext<double,N> ctx;
 ugkwp::gaswall::detail::makeContext(*in,*c,ctx);
 ugkwp::gaswall::detail::layerRow(*in,*c,ctx,w->state,w->y,1,out);
}
template<int N,int M> void instantiateWall(){wallRows<N,M><<<1,32>>>(nullptr,nullptr,nullptr,nullptr);}
void wallMatrix(){
 instantiateWall<2,24>();instantiateWall<2,48>();instantiateWall<2,128>();
 instantiateWall<10,24>();instantiateWall<10,48>();instantiateWall<10,128>();
}
__global__ void heat(ugkwp::ParticleThermalCarry<GpuReal>* state,float* nodes,double* residual){
 ugkwp::ParticleCaloric caloric;
 ugkwp::publishFiniteWallThermalHeat(*state,500.,1.,1.,.01,GpuReal(200),GpuReal(4000),caloric);
 ugkwp::publishColdWallUniformSpecificHeat(nodes,8,*residual,.01);
 ugkwp::publishColdWallUniformSpecificHeat(nodes,64,*residual,.01);
}
'''
    native_compile(tmp_path, code)


def test_actual_paired_candidate_compiles_as_cuda(tmp_path):
    prefix = PAIRED_PREFIX.replace('#define __device__\n', '').replace(
        '#define __forceinline__ inline\n', '')
    prefix = prefix.replace('template<class T>T ', 'template<class T>__device__ T ')
    prefix = prefix.replace('GpuReal molecularGasConductivity', '__device__ GpuReal molecularGasConductivity')
    bound = function_slice(ROOT / 'common/wall/GpuColdWallThermalCarry.cuh',
        'UGKWP_COLD_CARRY_HD inline double coldWallUniformCoolingBound',
        'UGKWP_COLD_CARRY_HD inline ColdWallHeatPublication').replace(
        'UGKWP_COLD_CARRY_HD', '__host__ __device__')
    actual = function_slice(ROOT / 'common/GpuPairedParticleHeat.cuh',
        'struct PairedParticleHeatCandidate', '// Called from')
    code = '#include <cuda_runtime.h>\n' + prefix + 'namespace ugkwp {\n' + bound + '}\n' + actual

    code += '\n__global__ void candidate(DeviceState* s,PairedParticleHeatCandidate* out){*out=pairedParticleHeatCandidate(*s,0,1.);}\n'
    native_compile(tmp_path, code)


FINITE_BODY = r'''
struct Linear { double cp; double enthalpy(double x) const {return cp*x;} double temperature(double x) const {return x/cp;} };
unsigned long long tests=0,failed=0,successful=0,ref=0,opt=0,stagnated=0,atomicFailures=0;
template<class T>bool bits(T a,T b){return std::memcmp(&a,&b,sizeof(T))==0;}
template<class T,class Law>void check(T t,double carry,double tw,double mass,double parcel,double g,T lo,T hi,const Law& law){
 ugkwp::ParticleThermalCarry<T>a{t,carry},b=a,initial=a;
 ugkwp::referenceIterations=ugkwp::stagnationIterations=0;
 auto ar=ugkwp::referenceFiniteHeat(a,tw,mass,parcel,g,lo,hi,law);
 auto br=ugkwp::stagnationFiniteHeat(b,tw,mass,parcel,g,lo,hi,law);
 bool equal=ar.valid==br.valid && bits(a.temperature,b.temperature)&&bits(a.specificEnthalpyResidual,b.specificEnthalpyResidual)&&bits(ar.wallEnergy,br.wallEnergy);
 if(!equal){std::cerr<<std::setprecision(17)<<"mismatch t="<<double(t)<<" carry="<<carry<<" tw="<<tw<<" mass="<<mass<<" parcel="<<parcel<<" g="<<g<<"\n";std::exit(2);}
 if(!ar.valid){++failed;if(!bits(initial.temperature,a.temperature)||!bits(initial.specificEnthalpyResidual,a.specificEnthalpyResidual)){++atomicFailures;std::exit(3);}}else ++successful;
 ref+=ugkwp::referenceIterations;opt+=ugkwp::stagnationIterations;stagnated+=ugkwp::stagnationIterations<ugkwp::referenceIterations;++tests;
}
double randomBits(std::mt19937_64&r){auto x=r();double d;std::memcpy(&d,&x,8);return d;}
double logRandom(std::mt19937_64&r){return std::ldexp(std::uniform_real_distribution<double>(.5,1.)(r),std::uniform_int_distribution<int>(-1073,1024)(r));}
template<class T>void run(){
 const double inf=std::numeric_limits<double>::infinity(),nan=std::numeric_limits<double>::quiet_NaN(),tiny=std::numeric_limits<double>::denorm_min(),max=std::numeric_limits<double>::max();
 ugkwp::ParticleCaloric law;law.solidCp=800;law.latentHeat=3e5;
 std::vector<double> temperatures={-inf,-1.,-0.,0.,tiny,std::numeric_limits<float>::denorm_min(),200.,300.,2317.,2327.,2337.,4000.,max,inf,nan};
 for(bool phase:{false,true}){law.phaseChange=phase;
  for(double t:temperatures)for(double wall:temperatures)for(double g:{-1.,-0.,0.,tiny,1e-20,1e-6,1.,max,inf,nan})for(double carry:{-1e-8,0.,1e-8,nan})check(T(t),carry,wall,1e-6,100.,g,T(200),T(4000),law);
  for(double t:{200.,300.,2317.,2327.,2337.,3999.,4000.})for(double w:{std::nextafter(t,0.),t,std::nextafter(t,inf)})for(double g:{tiny,1e-20,1e-6,1.,max})for(double mass:{tiny,1e-20,1e-6,1.,max})for(double parcel:{tiny,1.,max})check(T(t),0.,w,mass,parcel,g,T(200),T(4000),law);
 }
 std::mt19937_64 rng(0x59413210+sizeof(T));
 for(int i=0;i<150000;++i){law.phaseChange=(rng()&1);law.meltingTemperature=std::uniform_real_distribution<double>(100.,6000.)(rng);law.mushyRange=std::uniform_real_distribution<double>(1e-5,200.)(rng);law.solidCp=std::uniform_real_distribution<double>(1.,10000.)(rng);law.latentHeat=std::uniform_real_distribution<double>(0.,1e6)(rng);
  const double t=std::uniform_real_distribution<double>(200.,4000.)(rng),w=std::uniform_real_distribution<double>(1.,8000.)(rng),carry=std::uniform_real_distribution<double>(-1e-6,1e-6)(rng);
  check(T(t),carry,w,logRandom(rng),logRandom(rng),logRandom(rng),T(200),T(4000),law);
 }
 for(int i=0;i<40000;++i){Linear linear{logRandom(rng)};check(T(logRandom(rng)),randomBits(rng),logRandom(rng),logRandom(rng),logRandom(rng),logRandom(rng),T(0),std::numeric_limits<T>::max(),linear);}
 for(double cp:{tiny,1e-300,1.,1e300,max,-1.,0.,inf,nan})for(double t:{tiny,1e-300,1.,300.,1e300})for(double g:{tiny,1e-300,1.,1e300,max})check(T(t),0.,std::nextafter(t,inf),1.,1.,g,T(0),std::numeric_limits<T>::max(),Linear{cp});
}
int main(){run<double>();std::cout<<"{\"cases\":"<<tests<<",\"valid\":"<<successful<<",\"invalid_atomic_unchanged\":"<<failed<<",\"mismatches\":0,\"atomic_failure_count\":"<<atomicFailures<<",\"reference_bisections\":"<<ref<<",\"candidate_bisections\":"<<opt<<",\"cases_reduced\":"<<stagnated<<"}\n";}
'''

PAIRED_PREFIX = r'''

#include <cmath>
#include <cfloat>
#include <cstring>
#include <random>
#include <iostream>
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalCarry.H"
#define __device__
#define __forceinline__ inline
using GpuReal=double;
using GpuTime=double;
#define GPU_R(x) GpuReal(x)
namespace Foam {namespace gpuThermal {enum {particleWallMobile=0,particleWallSolidifyingDeposition=3,particleWallColdWall2D=4};}}
struct Parameters {GpuReal meltingTemperatureK=2327,mushyRangeK=20,latentHeatJkg=300000,solidSpecificHeatJkgK=800;};
struct DeviceState {
 int nFaces=1,nCells=1;
 int pStatus[1]{1},pCellId[1]{0},pStuck[1]{0},pStuckFaceId[1]{0},particleStuckCandidateMask[1]{0};
 int compactPStuck[1]{0},compactPStuckFaceId[1]{0};
 GpuReal pT[1]{300},compactPT[1]{300},pm[1]{1},pux[1]{0},puy[1]{0},puz[1]{0},pd[1]{.001};
 GpuReal couplingUxOld[1]{1},couplingUyOld[1]{0},couplingUzOld[1]{0},couplingRhoOld[1]{1},couplingTgasOld[1]{500};
 double pThermalEnthalpyResidual[1]{0},compactPThermalEnthalpyResidual[1]{0};
 float pColdNodeSpecificEnthalpy[8]{},compactPColdNodeSpecificEnthalpy[8]{},pCold2DNodeSpecificEnthalpy[64]{},compactPCold2DNodeSpecificEnthalpy[64]{};
 GpuReal particleDiameterFallback=.001,rhoMin=1e-10,gasMu=1.8e-5,gasPrOneThird=.9,rhoSolid=3900,TgasMin=200,TpMin=200,TpMax=4000,kappa=.02;
 Parameters coldWallSolidificationParameters;
};
template<class T>T finiteOr(T x,T fallback){return std::isfinite(x)?x:fallback;}
template<class T>T clampMin(T x,T y){return x>y?x:y;}
GpuReal molecularGasConductivity(const DeviceState&s){return s.kappa;}
#include "GpuParticleCaloricDevice.cuh"
'''

PAIRED_BODY = r'''
template<class T>bool bits(T a,T b){return std::memcmp(&a,&b,sizeof(T))==0;}
int main(){std::mt19937_64 rng(2034);unsigned long long cases=0,ri=0,gi=0;
 for(int kind:{0,3,4})for(int j=0;j<25000;++j){DeviceState s;s.pStuck[0]=kind?1:0;s.particleStuckCandidateMask[0]=kind;
  s.pT[0]=std::uniform_real_distribution<double>(200,4000)(rng);s.couplingTgasOld[0]=std::uniform_real_distribution<double>(200,5000)(rng);
  s.pm[0]=std::ldexp(1.,std::uniform_int_distribution<int>(-30,20)(rng));s.pThermalEnthalpyResidual[0]=std::uniform_real_distribution<double>(-1e-7,1e-7)(rng);
  const auto law=particleCaloricDevice(s,0);for(auto& x:s.pColdNodeSpecificEnthalpy)x=law.enthalpy(s.pT[0]);for(auto& x:s.pCold2DNodeSpecificEnthalpy)x=law.enthalpy(s.pT[0]);
  s.pd[0]=std::ldexp(1.,std::uniform_int_distribution<int>(-20,0)(rng));
  if(j%101==0)s.pStatus[0]=0;if(j%103==0)s.pCellId[0]=-1;if(j%107==0)s.pThermalEnthalpyResidual[0]=std::numeric_limits<double>::quiet_NaN();
  const double dt=j%109==0?0:std::ldexp(1.,std::uniform_int_distribution<int>(-80,10)(rng));
  incrementCount=0;auto a=referenceCandidate(s,0,dt);ri+=incrementCount;
  incrementCount=0;auto b=actualCandidate(s,0,dt);gi+=incrementCount;
  if(!bits(a.capacity,b.capacity)||!bits(a.weightedCapacity,b.weightedCapacity)||!bits(a.specificHeat,b.specificHeat))return 2;++cases;
 }
 std::cout<<"{\"storage_bits\":"<<sizeof(GpuReal)*8<<",\"cases\":"<<cases<<",\"mismatches\":0,\"baseline_enthalpyIncrement_calls\":"<<ri<<",\"actual_enthalpyIncrement_calls\":"<<gi<<"}"<<std::endl;
}
'''

COLD_BODY = r'''
template<class T>bool bits(T a,T b){return std::memcmp(&a,&b,sizeof(T))==0;}
int main(){std::mt19937_64 rng(1728);unsigned long long cases=0,invalid=0,zero=0,nonzeroRef=0,nonzeroActual=0,zeroRef=0,zeroActual=0;
 for(int n:{1,8,64})for(int i=0;i<100000;++i){std::array<float,64>a{},b{},init{};
  for(int j=0;j<n;++j){auto raw=std::uint32_t(rng());if(i%4==0)std::memcpy(&a[j],&raw,4);else a[j]=float(std::uniform_real_distribution<double>(1e-30,1e8)(rng));}
  double carry=std::uniform_real_distribution<double>(-1e7,1e7)(rng),q=std::uniform_real_distribution<double>(-1e7,1e7)(rng);
  if(i%5==0)q=0.;if(i%7==0)carry=1e-12;if(i%11==0)q=1e-15;
  if(i%17==0)q=std::numeric_limits<double>::infinity();if(i%19==0)carry=std::numeric_limits<double>::quiet_NaN();
  b=init=a;double r=carry,t=carry;scanCount=0;auto x=referenceColdHeat(a.data(),n,r,q);const auto ref=scanCount;scanCount=0;auto y=publishColdWallUniformSpecificHeat(b.data(),n,t,q);const auto actual=scanCount;
  if(x.valid){if(q==0.){zeroRef+=ref;zeroActual+=actual;}else{nonzeroRef+=ref;nonzeroActual+=actual;}}
  if(x.valid!=y.valid||!bits(x.acceptedSpecificHeat,y.acceptedSpecificHeat)||!bits(r,t)||std::memcmp(a.data(),b.data(),sizeof(a)))return 2;
  if(!x.valid){++invalid;if(!bits(r,carry)||std::memcmp(a.data(),init.data(),sizeof(a)))return 3;}if(q==0.)++zero;++cases;
 }
 std::cout<<"{\"cases\":"<<cases<<",\"invalid_atomic_unchanged\":"<<invalid<<",\"zero_heat_cases\":"<<zero<<",\"mismatches\":0}"<<"\n{\"nonzero_reference_node_visits\":"<<nonzeroRef<<",\"nonzero_actual_node_visits\":"<<nonzeroActual<<",\"zero_reference_node_visits\":"<<zeroRef<<",\"zero_actual_node_visits\":"<<zeroActual<<"}"<<std::endl;
}
'''
