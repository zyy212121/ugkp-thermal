"""B1 real producers, serial host execution only; no CUDA parity claim.

The immutable pre-B1 common tree is used for successful-path bit comparisons.
Only the actual recovery function is extracted, avoiding unrelated large TUs.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import os
import shutil
import subprocess
import tarfile
import pytest

ROOT = SOURCE_ROOT
BASE = "82f9c016704c9822b16b87cdeeb11f5bbd44d06f"

PREAMBLE = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>
#include <cfenv>
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasGeometryValidation.H"
using Real=double;
#define __device__
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
#define asm(...) std::abort()
constexpr double OfSmall=2.22044604925031308085e-16;
#include "operators/clampMin.cuh"
using namespace ugkwp;
struct State:GasStateView<double,double,int>{GasSpeciesState<double,2> gasSpecies;};
struct Model {
 SpeciesThermoData<double> s[2];double a[6]={1000,.2,-1e6,1100,.1,2e6};SpeciesThermoView<double,2> v;
 Model(){for(int k=0;k<2;++k){s[k].coefficientOffset=3*k;s[k].molarMass=k?.032:.028;s[k].minTemperature=200;s[k].maxTemperature=4000;s[k].referenceTemperature=300;s[k].hasEntropyReference=true;s[k].referenceEntropy=1500;}v.species=s;v.coefficients=a;v.coefficientCount=6;}
};
bool same(double a,double b){return (std::isnan(a)&&std::isnan(b))||std::memcmp(&a,&b,sizeof(a))==0;}
void dump(double x){std::printf("%a ",x);}
struct Caloric {
 int kind;mutable std::vector<double> seen;
 ThermoStatus operator()(double t,double&e,double&c)const{
  seen.push_back(t);if(kind>=3)return kind==3?ThermoStatus::InvalidModel:ThermoStatus::NonFiniteInput;
  c=kind==1?std::numeric_limits<double>::infinity():2.;e=kind==1?c:2*t+5;
  if(kind==2&&t>200&&t<4000)e=std::numeric_limits<double>::infinity();
  return ThermoStatus::Success;
 }
};
struct Cell {
 State s;Model m;double q[5]={2,6,-2,1,0},out[8]={-11,-12,-13,-14,-15,-16,-17,-18},mass[2]={.5,1.5};int status=0;
 Cell(){s.nCells=1;s.rho=q;s.rhoUx=q+1;s.rhoUy=q+2;s.rhoUz=q+3;s.rhoE=q+4;s.Ux=out;s.Uy=out+1;s.Uz=out+2;s.p=out+3;s.Tgas=out+4;s.gammaGas=1.4;s.Rgas=287;s.rhoMin=1e-12;s.TgasMin=1e-9;auto&g=s.gasSpecies;g.mode=GasMode::MixtureFrozen;g.thermo=m.v;g.rho=mass;g.cellStatus=&status;g.soundSpeed=out+5;g.heatCapacity=out+6;g.gasConstant=out+7;q[4]=mixtureEnergy(mass,700.,m.v)+.5*q[0]*(9+1+.25);}
};
'''

MAIN = r'''
int main(int argc,char**argv){assert(argc==2);const char*test=argv[1];Model m;double mass[2]={.5,1.5};
 if(!std::strcmp(test,"species")){
  gasThermoDetail::SpeciesValues<double> v{-11,-12,-13,-14,-15};double t=std::numeric_limits<double>::infinity();
  assert(gasThermoDetail::evaluateSpecies(0,t,m.v,v)==ThermoStatus::NonFiniteInput);
  const double R=universalGasConstant<double>()/m.s[0].molarMass;
  assert(same(v.cp,m.a[0]+m.a[1]*t));assert(same(v.h,m.a[2]+t*(m.a[0]+.5*m.a[1]*t)));
  assert(same(v.cv,v.cp-R));assert(same(v.e,v.h-R*t));assert(std::isinf(v.entropy));
 }else if(!std::strcmp(test,"energy")){
  double e=-11,c=-12;mass[0]=std::numeric_limits<double>::infinity();
  assert(evaluateMixtureEnergy(mass,700.,m.v,e,c)==ThermoStatus::InvalidComposition);
  assert(same(e,mass[0]*speciesE(0,700.,m.v)+mass[1]*speciesE(1,700.,m.v)));
  assert(same(c,mass[0]*speciesCv(0,700.,m.v)+mass[1]*speciesCv(1,700.,m.v)));
 }else if(!std::strcmp(test,"gas_constant")){
  mass[0]=mass[1]=1e307;double weighted=0,total=0;for(int k=0;k<2;++k){total+=mass[k];weighted+=mass[k]*universalGasConstant<double>()/m.s[k].molarMass;}
  assert(std::isinf(weighted/total));assert(same(mixtureGasConstant(mass,m.v),weighted/total));
  mass[0]=mass[1]=0;std::feclearexcept(FE_ALL_EXCEPT);volatile double r=mixtureGasConstant(mass,m.v);assert(std::isnan(r));assert(std::fetestexcept(FE_INVALID));
 }else if(!std::strcmp(test,"density")){
  double rho=-11;assert(mixtureDensityFromPressureTemperature(101325.,0.,mass,m.v,rho)==ThermoStatus::NonFiniteInput);assert(std::isinf(rho));
 }else if(!std::strcmp(test,"density_missing")){
  double rho=-11;m.v.species=nullptr;assert(mixtureDensityFromPressureTemperature(101325.,700.,mass,m.v,rho)==ThermoStatus::InvalidModel);assert(rho==-11);
 }else if(!std::strcmp(test,"caloric")){
  Caloric e{1,{}};double value=-11,c=-12;assert(evaluateCaloricEnergy(e,700.,value,c)==ThermoStatus::NonFiniteInput);assert(std::isinf(value)&&std::isinf(c));
  Caloric missing{3,{}};value=-11;c=-12;assert(evaluateCaloricEnergy(missing,700.,value,c)==ThermoStatus::InvalidModel);assert(value==-11&&c==-12);
 }else if(!std::strcmp(test,"inversion")){
  Caloric e{1,{}};ThermoInversionControls<double> c;double t=std::numeric_limits<double>::quiet_NaN();assert(invertCaloricEnergy(e,2005.,200.,4000.,c,t)==ThermoStatus::NonFiniteInput);assert(t==2100);assert(std::find(e.seen.begin(),e.seen.end(),t)!=e.seen.end());
  Caloric nanTarget{0,{}};t=std::numeric_limits<double>::quiet_NaN();assert(invertCaloricEnergy(nanTarget,std::numeric_limits<double>::quiet_NaN(),200.,4000.,c,t)==ThermoStatus::NonFiniteInput);assert(std::isnan(t)&&nanTarget.seen.empty()); // Generic outer fallbacks retain their original seed/evaluation sequence.
  Caloric missing{3,{}};t=-11;assert(invertCaloricEnergy(missing,2005.,200.,4000.,c,t)==ThermoStatus::InvalidModel);assert(t==-11&&missing.seen.size()==3);
 }else if(!std::strcmp(test,"mixture_inversion")){
  m.a[0]=m.a[3]=std::numeric_limits<double>::max();double t=std::numeric_limits<double>::quiet_NaN();ThermoInversionControls<double> c;assert(invertMixtureEnergy(mass,1.,m.v,c,t)==ThermoStatus::NonFiniteInput);assert(t==2100);
 }else if(!std::strcmp(test,"primitive")){
  Cell x;x.q[0]=0;x.out[4]=std::numeric_limits<double>::quiet_NaN();double before[5];std::memcpy(before,x.q,sizeof before);recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidComposition));assert(std::memcmp(before,x.q,sizeof before)==0);
  assert(std::isinf(x.out[0])&&std::isinf(x.out[1])&&std::isinf(x.out[2]));assert(x.out[4]==2100);double rD=0,cpD=0;for(int k=0;k<2;++k){rD+=x.mass[k]*universalGasConstant<double>()/x.m.s[k].molarMass;cpD+=x.mass[k]*speciesCp(k,x.out[4],x.m.v);}assert(same(x.out[3],rD*x.out[4]));assert(same(x.out[6],cpD/x.q[0]));assert(same(x.out[7],rD/x.q[0]));assert(std::isnan(x.out[5]));
 }else if(!std::strcmp(test,"status_carry")){
  Cell x;x.status=int(GasTransportCode::InvalidThermodynamics);recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidThermodynamics));assert(x.out[0]==3&&x.out[1]==-1&&x.out[2]==.5);assert(std::abs(x.out[4]-700)<1e-6);assert(x.out[3]>0&&x.out[5]>0&&x.out[6]>x.out[7]&&x.out[7]>0);
 }else if(!std::strcmp(test,"primitive_mass_nan")){
  Cell x;x.mass[0]=std::numeric_limits<double>::quiet_NaN();x.out[4]=std::numeric_limits<double>::quiet_NaN();double before[5];std::memcpy(before,x.q,sizeof before);recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidComposition));assert(std::memcmp(before,x.q,sizeof before)==0);assert(x.out[0]==3&&x.out[1]==-1&&x.out[2]==.5);assert(x.out[4]==2100);for(int k:{3,5,6,7})assert(std::isnan(x.out[k]));
 }else if(!std::strcmp(test,"primitive_metadata_count")){
  Cell x;x.s.gasSpecies.thermo.coefficientCount=1;x.out[4]=std::numeric_limits<double>::quiet_NaN();double before[5];std::memcpy(before,x.q,sizeof before);recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidThermodynamics));assert(std::memcmp(before,x.q,sizeof before)==0);assert(x.out[0]==3&&x.out[1]==-1&&x.out[2]==.5);assert(x.out[4]==2100);double rD=0,cpD=0;for(int k=0;k<2;++k){rD+=x.mass[k]*universalGasConstant<double>()/x.m.s[k].molarMass;cpD+=x.mass[k]*(x.m.a[3*k]+x.m.a[3*k+1]*x.out[4]);}const double r=rD/x.q[0],cp=cpD/x.q[0];assert(same(x.out[3],rD*x.out[4]));assert(same(x.out[6],cp)&&same(x.out[7],r));assert(same(x.out[5],std::sqrt(cp/(cp-r)*r*x.out[4])));
 }else if(!std::strcmp(test,"primitive_overflow")){
  Cell x;x.m.a[1]=x.m.a[4]=std::numeric_limits<double>::max();x.out[4]=std::numeric_limits<double>::quiet_NaN();recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidThermodynamics));assert(x.out[0]==3&&x.out[1]==-1&&x.out[2]==.5);assert(x.out[4]==2100&&x.out[3]>0&&x.out[7]>0);assert(std::isinf(x.out[6])&&std::isnan(x.out[5]));
 }else if(!std::strcmp(test,"moving")){
  Cell x;x.s.gasSpecies.mode=GasMode::SingleLegacy;x.s.gasGeometry.enabled=true;x.q[0]=0;x.status=int(GasTransportCode::InvalidComposition);recoverGasPrimitiveCell(x.s,0);assert(x.status==int(GasTransportCode::InvalidComposition));for(int k=0;k<3;++k)assert(std::isinf(x.out[k]));assert(std::isnan(x.out[3])&&std::isnan(x.out[4]));
 }else if(!std::strcmp(test,"missing")){
  Cell x;x.s.gasSpecies.thermo.coefficients=nullptr;double before[8];std::memcpy(before,x.out,sizeof before);recoverGasPrimitiveCell(x.s,0);assert(x.status!=0&&std::memcmp(before,x.out,sizeof before)==0);
  Cell y;y.s.gasSpecies.soundSpeed=nullptr;std::memcpy(before,y.out,sizeof before);recoverGasPrimitiveCell(y.s,0);assert(y.status==int(GasTransportCode::InvalidStorage)&&std::memcmp(before,y.out,sizeof before)==0);
 }else if(!std::strcmp(test,"success")){
  for(double t:{200.,700.,4000.}){gasThermoDetail::SpeciesValues<double> v;assert(gasThermoDetail::evaluateSpecies(0,t,m.v,v)==ThermoStatus::Success);for(double x:{v.cp,v.cv,v.h,v.e,v.entropy})dump(x);double e,c;assert(evaluateMixtureEnergy(mass,t,m.v,e,c)==ThermoStatus::Success);dump(e);dump(c);dump(mixtureGasConstant(mass,m.v));double rho;assert(mixtureDensityFromPressureTemperature(101325.,t,mass,m.v,rho)==ThermoStatus::Success);dump(rho);ThermoInversionControls<double> ctl;double out=333;assert(invertMixtureEnergy(mass,e,m.v,ctl,out)==ThermoStatus::Success);dump(out);}
  for(int kind:{0,2})for(int budget:{0,1,100}){Caloric e{kind,{}};ThermoInversionControls<double> c;c.maximumIterations=budget;double t=700;assert(invertCaloricEnergy(e,2005.,200.,4000.,c,t)==ThermoStatus::Success);dump(t);for(double x:e.seen)dump(x);std::puts("");}
  Caloric e{0,{}};ThermoInversionControls<double> c;double t=333;c.maximumIterations=1;double seed=t;assert(invertCaloricEnergy<true>(e,2005.,200.,4000.,c,t)==ThermoStatus::IterationLimit);assert(t==seed);c.maximumIterations=100;assert(invertCaloricEnergy<true>(e,2005.,200.,4000.,c,t)==ThermoStatus::Success);dump(t);for(double x:e.seen)dump(x);
  double value=-11,capacity=-12;mass[0]=-1;assert(evaluateMixtureEnergy<true>(mass,700.,m.v,value,capacity)==ThermoStatus::InvalidComposition);assert(value==-11&&capacity==-12);mass[0]=.5;
  Caloric callbackError{4,{}};t=333;assert(invertCaloricEnergy(callbackError,2005.,200.,4000.,c,t)==ThermoStatus::NonFiniteInput);assert(t==333&&callbackError.seen.size()==3);dump(t);for(double z:callbackError.seen)dump(z);
  Cell x;double outputs[8];std::memcpy(outputs,x.out,sizeof outputs);x.mass[0]=-.5;recoverGasPrimitiveCell(x.s,0,true);assert(x.status!=0&&std::memcmp(outputs,x.out,sizeof outputs)==0);x.mass[0]=.5;x.status=0;recoverGasPrimitiveCell(x.s,0,true);assert(x.status==0);for(double z:x.out)dump(z);for(double z:x.q)dump(z);
  Cell normal;recoverGasPrimitiveCell(normal.s,0);assert(normal.status==0);for(double z:normal.out)dump(z);for(double z:normal.q)dump(z);
  for(int moving:{0,1}){Cell legacy;legacy.s.gasSpecies.mode=GasMode::SingleLegacy;legacy.s.gasGeometry.enabled=bool(moving);recoverGasPrimitiveCell(legacy.s,0);assert(legacy.status==0);for(double z:legacy.out)dump(z);for(double z:legacy.q)dump(z);}std::puts("");
 }else return 2;
}
'''

def recovery(root):
    source = (root / "common/operators/computeSstFaceFluxKernel.cuh").read_text()
    start = source.index("template<class GasState>\n__device__ void recoverGasPrimitiveCell")
    end = source.index("template<class GasState>\n__global__ void recoverGasPrimitivesKernel", start)
    return source[start:end]


def build(folder, root):
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "probe.cpp"
    source.write_text(PREAMBLE + recovery(root) + MAIN)
    exe = folder / "probe"
    result = subprocess.run(["g++", "-std=c++17", "-O0", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(root / "common"), str(source), "-o", str(exe)], capture_output=True, text=True)
    (folder / "build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return exe


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    return build(tmp_path_factory.mktemp("b1-current"), ROOT)


@pytest.mark.parametrize("scenario", ["species", "energy", "gas_constant", "density", "density_missing", "caloric", "inversion", "mixture_inversion", "primitive", "status_carry", "primitive_mass_nan", "primitive_metadata_count", "primitive_overflow", "moving", "missing"])
def test_actual_failed_producer_completes_fields(probe, scenario):
    result = subprocess.run([str(probe), scenario], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_original_success_bits_and_strict_later_success(tmp_path, probe):
    original = tmp_path / "base"
    original.mkdir()
    archive = subprocess.check_output(["git", "archive", BASE, "common"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(original, filter="data")
    old = subprocess.run([str(build(tmp_path / "base-probe", original)), "success"], capture_output=True, text=True)
    new = subprocess.run([str(probe), "success"], capture_output=True, text=True)
    (tmp_path / "base-success.log").write_text(old.stdout + old.stderr)
    (tmp_path / "current-success.log").write_text(new.stdout + new.stderr)
    assert old.returncode == new.returncode == 0, old.stderr + new.stderr
    assert old.stdout == new.stdout


def test_cxx14_real_thermo_template_completion(tmp_path):
    source = tmp_path / "thermo.cpp"
    source.write_text(r'''
#include "gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
int main(){
 ugkwp::SpeciesThermoData<double> data;data.molarMass=.028;data.minTemperature=200;data.maxTemperature=4000;
 double coefficients[3]={1000,0,0};ugkwp::SpeciesThermoView<double,1> view;view.species=&data;view.coefficients=coefficients;view.coefficientCount=3;
 double mass[1]={1};ugkwp::ThermoInversionControls<double> controls;double T=std::numeric_limits<double>::quiet_NaN();
 ugkwp::MixtureEnergyEvaluator<double,1,double,false> evaluator{mass,view};
 assert(ugkwp::invertCaloricEnergy(evaluator,std::numeric_limits<double>::quiet_NaN(),200.,4000.,controls,T)==ugkwp::ThermoStatus::NonFiniteInput);assert(T==2100);
 double energy=-11,capacity=-12;T=2100;assert(ugkwp::evaluateMixtureEnergy<true>(mass,T,view,energy,capacity)==ugkwp::ThermoStatus::Success);
 double recovered=700;assert(ugkwp::invertMixtureEnergy<true>(mass,energy,view,controls,recovered)==ugkwp::ThermoStatus::Success);assert(recovered==T);
}
''')
    exe = tmp_path / "thermo"
    result = subprocess.run(["g++", "-std=c++14", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I" + str(ROOT / "common"), str(source), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cuda_real_primitive_completion_compilation_when_available(tmp_path):
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc is unavailable; serial host probes do not prove CUDA compilation")
    source = tmp_path / "completion.cu"
    source.write_text(r'''
#include <cmath>
#include "gasTransport/MixtureThermo.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasGeometryValidation.H"
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
constexpr double OfSmall=2.22044604925031308085e-16;
#include "operators/clampMin.cuh"
template<int N>struct State:ugkwp::GasStateView<double,double,int>{ugkwp::GasSpeciesState<double,N> gasSpecies;};
''' + recovery(ROOT) + r'''
__global__ void complete2(State<2>*s){recoverGasPrimitiveCell(*s,0,false);}
__global__ void retry2(State<2>*s){recoverGasPrimitiveCell(*s,0,true);}
__global__ void complete10(State<10>*s){recoverGasPrimitiveCell(*s,0,false);}
__global__ void legacy(ugkwp::GasStateView<double,double,int>*s){recoverGasPrimitiveCell(*s,0,false);}
''')
    result = subprocess.run([nvcc, "-std=c++17", "-arch=sm_80", "-I" + str(ROOT / "common"), "-c", str(source), "-o", str(tmp_path / "completion.o")], capture_output=True, text=True)
    (tmp_path / "cuda-build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
