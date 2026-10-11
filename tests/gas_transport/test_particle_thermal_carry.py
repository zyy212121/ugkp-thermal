"""Compensated material heat: actual stored temperature plus bounded J/kg carry."""
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE = r'''
#include <algorithm>
#include <initializer_list>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalCarry.H"
using Caloric=ugkwp::ParticleCaloric;
void ck(bool ok,const char*msg){if(!ok){std::fprintf(stderr,"%s\n",msg);std::abort();}}
template<class Storage>void run(){
 Caloric law;ugkwp::ParticleThermalCarry<Storage> p{Storage(3000),0};
 const double mass=100,base=law.enthalpy(double(p.temperature));
 const Storage original=p.temperature;
 const auto zero=ugkwp::publishParticleThermalHeat(p,0.,Storage(1),Storage(6000),law);
 ck(zero.valid&&zero.acceptedSpecificHeat==0&&p.temperature==original&&p.specificEnthalpyResidual==0,"zero heat reconstructed state");
 for(double q:{.02,.03,.05,-.02,-.03,-.05}){
  p={Storage(3000),0};const auto result=ugkwp::publishParticleThermalHeat(p,q/mass,Storage(1),Storage(6000),law);
  ck(result.valid,"microheat rejected");
  const long double actual=mass*((long double)law.enthalpy(double(p.temperature))-base+(long double)p.specificEnthalpyResidual);
  ck(std::abs(actual-q)<2e-8L,"actual temperature+carry lost microheat");
  ck(std::abs(mass*result.acceptedSpecificHeat-q)<2e-8,"accepted actual publication is not requested heat");
  ck((ugkwp::effectiveParticleTemperature(p,law)>3000)==(q>0),"carry did not feed effective temperature");
 }
 for(double sign:{-1.0,1.0}){
 p={Storage(3000),0};double gas=1e9;const double gas0=gas;
 for(int step=0;step<100000;++step){
  // The shared publication uses actual represented gas decrement. The particle
  // carry then retains that accepted amount rather than a prospective transfer.
  const double nextGas=gas-sign*.03,accepted=gas-nextGas;
  const auto result=ugkwp::publishParticleThermalHeat(p,accepted/mass,Storage(1),Storage(6000),law);
  ck(result.valid,"cumulative heat rejected");gas=nextGas;
  const Storage up=std::nextafter(p.temperature,std::numeric_limits<Storage>::infinity());
  const Storage down=std::nextafter(p.temperature,-std::numeric_limits<Storage>::infinity());
  const double spacing=std::max(law.enthalpy(double(up))-law.enthalpy(double(p.temperature)),law.enthalpy(double(p.temperature))-law.enthalpy(double(down)));
  ck(std::abs(p.specificEnthalpyResidual)<=spacing+1e-8,"carry grew beyond material storage resolution");
 }
 const long double gain=mass*((long double)law.enthalpy(double(p.temperature))-base+(long double)p.specificEnthalpyResidual);
 ck(p.temperature!=original,"long microheat trajectory never changed published temperature");
 ck(std::abs(gain-(gas0-gas))<2e-6L,"long trajectory gas/particle actual budget leaked");
 ck(std::abs(gain-sign*3000.L)<.012L,"cumulative physical exchange exceeded registered double-roundoff bound");
 std::printf("storage=%zu gain=%.17Lg reference_error=%.17Lg paired_error=%.17Lg carry=%.17g\n",sizeof(Storage),gain,gain-sign*3000.L,gain-(gas0-gas),p.specificEnthalpyResidual);
 }
 // Exercise both piecewise liquid breakpoints and phase boundaries with
 // nonzero legacy temperatures and zero carry, as old-checkpoint migration.
 for(bool phase:{false,true}){
  law.phaseChange=phase;law.solidCp=1200.;law.latentHeat=1e6;
  for(double t:{1.,2317.,2327.,2337.,4000.,5999.}){
   for(double sign:{-1.,1.}){
    p={Storage(t),0};const long double h0=law.enthalpy(double(p.temperature));
    const auto r=ugkwp::publishParticleThermalHeat(p,sign*.03/mass,Storage(.5),Storage(6000),law);
    ck(r.valid,"piecewise publication rejected");
    const long double delta=mass*((long double)law.enthalpy(double(p.temperature))-h0+p.specificEnthalpyResidual);
    ck(std::abs(delta-sign*.03)<2e-8L,"piecewise caloric microheat lost");
    const double effective=ugkwp::effectiveParticleTemperature(p,law);
    ck((effective>t)==(sign>0),"phase/piecewise effective temperature missed carry");
   }
  }
 }
 law=Caloric{};
 // A real shared bound clips the accepted heat, not only a primitive cache.
 p={Storage(3000),.001};const double oldH=ugkwp::effectiveParticleEnthalpy(p,law);
 const auto bounded=ugkwp::publishParticleThermalHeat(p,1e12,Storage(1),Storage(3100),law);
 ck(bounded.valid&&bounded.limited&&p.temperature==Storage(3100),"shared bound not honored");
 ck(std::abs((law.enthalpy(3100)-oldH)-bounded.acceptedSpecificHeat)<2e-8,"bound reports intended rather than actual accepted heat");
 const auto before=p;const auto invalid=ugkwp::publishParticleThermalHeat(p,std::numeric_limits<double>::infinity(),Storage(1),Storage(3100),law);
 ck(!invalid.valid&&p.temperature==before.temperature&&p.specificEnthalpyResidual==before.specificEnthalpyResidual,"invalid candidate partially published");
}
int main(){run<STORAGE>();}
'''

@pytest.mark.parametrize('storage', ['float','double'])
@pytest.mark.parametrize('bits', [32,64])
def test_actual_compensated_material_heat_and_cumulative_budget(tmp_path,storage,bits):
    code=tmp_path/'carry.cpp';code.write_text(SOURCE.replace('STORAGE',storage))
    executable=tmp_path/'carry'
    built=subprocess.run(['g++','-std=c++17','-O2',f'-DUGKWP_GPU_REAL_BITS={bits}','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),str(code),'-o',str(executable)],capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    result=subprocess.run([str(executable)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout,end='')
