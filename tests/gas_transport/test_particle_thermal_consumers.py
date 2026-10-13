"""HOST probes for actual carry-aware finite-wall and radiation publication."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_wall_and_radiation_publish_actual_small_heat(tmp_path):
    header = ROOT / "common/GpuParticleThermalConsumers.H"
    assert header.exists(), "wall and radiation need carry-aware publication, not a pT overwrite"
    code = r'''
#include "GpuParticleThermalConsumers.H"
#include "GpuParticleCaloric.H"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
using ugkwp::ParticleThermalCarry;
double change(const ParticleThermalCarry<double>& a,const ParticleThermalCarry<double>& b,const ugkwp::ParticleCaloric& law) {
 return (law.enthalpy(double(b.temperature))-law.enthalpy(double(a.temperature)))+(b.specificEnthalpyResidual-a.specificEnthalpyResidual);
}
int main() {
 ugkwp::ParticleCaloric law;
 const double lower=200,upper=4000;
 ParticleThermalCarry<double> old{300.0,0.0002}, state=old;
 auto wall=ugkwp::publishFiniteWallThermalHeat(state,500.0,1e-6,100.0,1.5e-12,lower,upper,law);
 assert(wall.valid&&state.temperature>300.0);
 double q=100*change(old,state,law);
 assert(q>.0299&&q<.0301&&std::abs(q+wall.wallEnergy)<1e-12);
 const double oldEffective=law.temperature(law.enthalpy(300.0)+old.specificEnthalpyResidual);
 const double expected=100.0*(500-oldEffective)*1.5e-12/(1e-6+1.5e-12/law.specificHeat(300.0));
 assert(std::abs(q-expected)<1e-7);
 state=old;
 auto zero=ugkwp::publishFiniteWallThermalHeat(state,500,1e-6,100,0,lower,upper,law);
 assert(zero.valid&&zero.wallEnergy==0&&state.temperature==old.temperature&&state.specificEnthalpyResidual==old.specificEnthalpyResidual);
 state=old;
 auto radiation=ugkwp::publishAffineRadiationThermalHeat(state,1.0,0.0003/law.specificHeat(300),lower,upper,law);
 q=100*change(old,state,law);
 assert(radiation.valid&&state.temperature>300.0&&q>.02999&&q<.03001);
 assert(std::abs(q-100*radiation.acceptedSpecificHeat)<1e-12);
 state={300.0,1e-20}; const auto microscopic=state;
 auto identity=ugkwp::publishAffineRadiationThermalHeat(state,1.0,0.0,lower,upper,law);
 assert(identity.valid&&identity.acceptedSpecificHeat==0&&state.temperature==microscopic.temperature&&state.specificEnthalpyResidual==microscopic.specificEnthalpyResidual);
 state=old;
 auto extended=ugkwp::publishAffineRadiationThermalHeat(state,1.0,5000.0,lower,upper,law);
 assert(extended.valid&&state.temperature==upper);
 assert(std::abs((ugkwp::effectiveParticleEnthalpy(state,law)
   -ugkwp::effectiveParticleEnthalpy(old,law))
   -law.enthalpyIncrement(ugkwp::effectiveParticleTemperature(old,law),5000.0))<1e-8);
 state=old;
 auto limited=ugkwp::publishFiniteWallThermalHeat(state,5000.0,1e-6,100.0,1.0,lower,upper,law);
 assert(limited.valid&&state.temperature<=upper&&std::abs(100*change(old,state,law)+limited.wallEnergy)<1e-7);
 for(int j=0;j<10000;++j) {
  auto status=ugkwp::publishAffineRadiationThermalHeat(old,1.0,0.0003/law.specificHeat(300),lower,upper,law);
  assert(status.valid);
 }
 const double total=100*((law.enthalpy(old.temperature)-law.enthalpy(300.0))+(old.specificEnthalpyResidual-.0002));
 assert(std::abs(total-300.0)<.0001);
 std::cout << "wall_and_radiation_actual_carry_publication_passed\n";
}
'''
    source, binary = tmp_path / "consumers.cpp", tmp_path / "consumers"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("heat_joules", [.02, .03, .05])
def test_actual_double_radiation_kernel_retains_small_heat(tmp_path, heat_joules):
    text = (ROOT / "applications/CHT/gpu/GpuResidentStrict.cu").read_text()
    start = text.index("__global__ void applyParticleRadiationAffineTemperatureKernel")
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    kernel = text[start:end].replace('asm("trap;");', 'throw std::runtime_error("required device error");')
    code = r'''
#include "GpuPrecisionTypes.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalCarry.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
#if __has_include("GpuParticleThermalConsumers.H")
#include "GpuParticleThermalConsumers.H"
#endif
#include <cassert>
#include <stdexcept>
#include <cmath>
#define __global__
struct Dim {int x=1;};Dim gridDim,blockDim;struct Index {int x=0;};Index blockIdx,threadIdx;
namespace Foam {namespace gpuThermal {constexpr int particleWallMobile=0;}}
struct DeviceState {
 int count=1,nCells=1;int* particleCountDevice=&count;
 int pStatus[1]={1},pStuck[1]={0},pCellId[1]={0};
 double pT[1]={300};double pThermalEnthalpyResidual[1]={.0002};
 double TpMin=200,TpMax=4000;
};
ugkwp::ParticleCaloric particleCaloricDevice(const DeviceState&,int) {return {};}
KERNEL
int main() {
 DeviceState state;
 const ugkwp::ParticleCaloric caloric;
 const double requested=REQUESTED_HEAT;
 Foam::gpuThermal::RadiationAffineTemperatureUpdate update{1.0,(requested/100)/caloric.specificHeat(300)};
 applyParticleRadiationAffineTemperatureKernel(&state,&update);
 const double gain=100*((caloric.enthalpy(double(state.pT[0]))-caloric.enthalpy(300.0))+(state.pThermalEnthalpyResidual[0]-.0002));
 assert(std::abs(gain-requested)<1e-5&&state.pT[0]>300.0);
}
'''.replace("KERNEL", kernel).replace("REQUESTED_HEAT", repr(heat_joules))
    source, binary = tmp_path / "radiation.cpp", tmp_path / "radiation"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "common"), "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, "actual radiation pT publication discarded its sub-ULP heat"


def test_radiation_wire_affine_coefficients_keep_double_information(tmp_path):
    code = r'''
#include "GpuPrecisionTypes.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
#include <cassert>
#include <stdexcept>
int main() {
 const double scale=1.0-1e-10, offset=300e-10+2e-7;
 Foam::gpuThermal::RadiationAffineTemperatureUpdate value{scale,offset};
 assert(value.scale==scale&&value.offset==offset);
}
'''
    source, binary = tmp_path / "wire.cpp", tmp_path / "wire"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT / "common"), "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode == 0, "radiation affine wire coefficients were narrowed before the carry update"


def test_legacy_material_cache_contribution_includes_live_and_compact_carry(tmp_path):
    header = (ROOT / "common/GpuMaterialEnthalpyMoment.cuh").read_text()
    assert "materialEnthalpyMomentFromParticle" in header, "derived material moments omit persistent carry"
    code = r'''
#include <cassert>
#include <stdexcept>
#include <cmath>
#define __device__
#define __forceinline__ inline
template<class T> T particleSpecificEnthalpyDevice(T temperature){return T(1000)*temperature;}
#include "GpuMaterialEnthalpyMoment.cuh"
struct State { double pT[1]={300},compactPT[1]={300}; double pThermalEnthalpyResidual[1]={.0003}; double compactPThermalEnthalpyResidual[1]={-.0002}; };
struct Legacy {};
int main(){
 State s;Legacy old;
 const auto live=materialEnthalpyMomentFromParticle<false>(s,0,100.0,300.0,0);
 const auto compact=materialEnthalpyMomentFromParticle<true>(s,0,100.0,300.0,0);
 assert(std::abs(live-30000000.03)<1e-8&&std::abs(compact-29999999.98)<1e-8);
 assert(materialEnthalpyMomentFromParticle<false>(old,0,100.0,300.0,0)==30000000.0);
 // These are explicitly derived caches. The persistent particle carry remains
 // authoritative alongside the double derived cell publication.
 const double cache=live;assert(cache==live&&cache>30000000.0&&s.pThermalEnthalpyResidual[0]==.0003);
}
'''
    source, binary = tmp_path / "moments.cpp", tmp_path / "moments"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O0", "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_actual_finite_contact_adapter_commits_carry_and_keeps_legacy_fallback(tmp_path):
    code = r'''
#include "GpuParticleCaloric.H"
#include "wall/GpuParticleWallContactHeat.H"
#include <cassert>
#include <stdexcept>
#include <cmath>
#define __device__
#define __forceinline__ inline
struct State {
 double pT[1]={300},TpMin=200,TpMax=4000;
 double pThermalEnthalpyResidual[1]={.0002};
};
struct Legacy { double pT[1]={300}; };
ugkwp::ParticleCaloric particleCaloricDevice(const State&,int) {return {};}
#include "GpuParticleThermalConsumerDevice.cuh"
int main() {
 State state; const ugkwp::ParticleCaloric law;
 const auto candidate=particleFiniteContactThermalCandidateDevice(state,0,500.0,1e-6,100.0,1.5e-12,0);
 assert(candidate.valid&&state.pT[0]==300&&state.pThermalEnthalpyResidual[0]==.0002);
 commitParticleFiniteContactThermalCandidateDevice(state,0,candidate,0);
 const double actual=100*((law.enthalpy(double(state.pT[0]))-law.enthalpy(300.0))+(state.pThermalEnthalpyResidual[0]-.0002));
 assert(actual>.0299&&actual<.0301&&std::abs(actual+candidate.wallEnergyJ)<1e-12);
 Legacy legacy;
 const auto expected=Foam::gpuThermal::lumpedParticleWallInterfaceContact(300.0,500.0,1e-6,100.0,1e-9);
 const auto fallback=particleFiniteContactThermalCandidateDevice(legacy,0,500.0,1e-6,100.0,1e-9,0);
 assert(fallback.valid==expected.valid&&fallback.particleTemperatureK==expected.particleTemperatureK&&fallback.wallEnergyJ==expected.wallEnergyJ);
 commitParticleFiniteContactThermalCandidateDevice(legacy,0,fallback,0);
 assert(legacy.pT[0]==expected.particleTemperatureK);
}
'''
    source, binary = tmp_path / "wall_adapter.cpp", tmp_path / "wall_adapter"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "common"), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_actual_radiation_apply_reports_required_errors_and_continues(tmp_path):
    text = (ROOT / "applications/CHT/gpu/GpuResidentStrict.cu").read_text()
    start = text.index("__global__ void applyParticleRadiationAffineTemperatureKernel")
    brace = text.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    kernel = text[start:end].replace('asm("trap;");', 'throw std::runtime_error("required device error");')
    code = r'''
#include "GpuPrecisionTypes.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <stdexcept>
using std::isfinite;
#define __global__
struct Dim {int x=1;};Dim gridDim,blockDim;struct Index {int x=0;};Index blockIdx,threadIdx;
int atomicCAS(int* p,int expected,int replacement){int old=*p;if(old==expected)*p=replacement;return old;}
namespace Foam {namespace gpuThermal {constexpr int particleWallMobile=0;}}
struct DeviceState {
 int count=1,nCells=1;int* particleCountDevice=&count;
 int pStatus[1]={1},pStuck[1]={0},pCellId[1]={0};unsigned long long pOrigId[1]={19};
 double pT[1]={300};double pThermalEnthalpyResidual[1]={.0002};double TpMin=200,TpMax=4000;
};
ugkwp::ParticleCaloric particleCaloricDevice(const DeviceState&,int){return {};}
double particleEffectiveTemperatureDevice(const DeviceState&s,int i){const auto c=particleCaloricDevice(s,i);return c.temperature(c.enthalpy(s.pT[i])+s.pThermalEnthalpyResidual[i]);}
KERNEL
int main(){
 DeviceState state;Foam::gpuThermal::RadiationAffineTemperatureUpdate update{1.0,2e-7};
 auto expected=ugkwp::ParticleThermalCarry<double>{state.pT[0],state.pThermalEnthalpyResidual[0]};
 assert(ugkwp::publishAffineRadiationThermalHeat(expected,update.scale,update.offset,state.TpMin,state.TpMax,particleCaloricDevice(state,0)).valid);
 applyParticleRadiationAffineTemperatureKernel(&state,&update);
 assert(state.pT[0]==expected.temperature&&state.pThermalEnthalpyResidual[0]==expected.specificEnthalpyResidual);
 state.pThermalEnthalpyResidual[0]=std::numeric_limits<double>::quiet_NaN();
 bool failed=false;try{applyParticleRadiationAffineTemperatureKernel(&state,&update);}catch(const std::runtime_error&){failed=true;}assert(failed);
 state.pThermalEnthalpyResidual[0]=.0002;state.pCellId[0]=-1;failed=false;
 try{applyParticleRadiationAffineTemperatureKernel(&state,&update);}catch(const std::runtime_error&){failed=true;}assert(failed);
 // Unconsumed inactive/stuck entries do not become new errors.
 state.pStatus[0]=0;applyParticleRadiationAffineTemperatureKernel(&state,&update);
 state.pStatus[0]=1;state.pStuck[0]=1;applyParticleRadiationAffineTemperatureKernel(&state,&update);
}
'''.replace("KERNEL", kernel.replace('asm("trap;");', 'throw std::runtime_error("required device error");'))
    source, binary = tmp_path / "radiation_validate.cpp", tmp_path / "radiation_validate"
    source.write_text(code)
    subprocess.run(["g++", "-std=c++17", "-O2", "-I" + str(ROOT / "common"), "-I" + str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
