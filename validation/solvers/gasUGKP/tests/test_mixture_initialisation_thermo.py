"""Common EOS initialization is shared by native application frontends."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT=SOURCE_ROOT

def test_initialisation_uses_common_eos_and_completes_diagnostic_output(tmp_path):
    source=tmp_path/"probe.cpp"
    source.write_text(r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
#include <limits>
template<class R> void check(){
 ugkwp::SpeciesThermoData<R> species[2];
 R coefficients[]={R(1040),R(0),R(-10000),R(920),R(0),R(20000)};
 for(int i=0;i<2;++i){species[i].coefficientOffset=3*i;species[i].molarMass=i?R(.032):R(.028);species[i].minTemperature=100;species[i].maxTemperature=4000;species[i].referencePressure=101325;}
 ugkwp::SpeciesThermoView<R,2> view;view.species=species;view.coefficients=coefficients;view.coefficientCount=6;
 R y[]={R(.3),R(.7)}, masses[]={R(3),R(7)};
 R expected=ugkwp::universalGasConstant<R>()*(R(.3)/R(.028)+R(.7)/R(.032));
 R gasConstant=ugkwp::mixtureGasConstant(y,view);
 assert(std::abs(gasConstant-expected)<R(1e-5)*expected);
 assert(std::abs(ugkwp::mixtureGasConstant(masses,view)-gasConstant)<R(1e-5)*expected);
 R rho=R(-23);
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),y,view,rho)==ugkwp::ThermoStatus::Success);
 assert(std::abs(rho-R(101325)/(expected*R(300)))<R(1e-5));
 // The public initialization EOS has no range/composition admission policy.
 // Keep each original input case and check its finite ideal-mixture result.
 auto acceptedEos=[&](R pressure,R temperature,const R* composition){
  const long double a=composition[0],b=composition[1];
  const long double referenceR=8.31446261815324L*(a/species[0].molarMass+b/species[1].molarMass)/(a+b);
  const long double referenceRho=static_cast<long double>(pressure)/(referenceR*temperature);
  assert(ugkwp::mixtureDensityFromPressureTemperature(pressure,temperature,composition,view,rho)==ugkwp::ThermoStatus::Success);
  assert(std::abs(static_cast<long double>(rho)-referenceRho)<R(1e-5));
 };
 acceptedEos(R(-1),R(300),y);
 acceptedEos(R(101325),R(5000),y);
 acceptedEos(R(101325),R(300),masses);
 y[0]=R(-.1);y[1]=R(1.1);
 acceptedEos(R(101325),R(300),y);
 y[0]=y[1]=0;assert(!std::isfinite(ugkwp::mixtureGasConstant(y,view)));
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),y,view,rho)==ugkwp::ThermoStatus::NonFiniteInput && std::isnan(rho));
 rho=R(-23);
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),static_cast<const R*>(nullptr),view,rho)==ugkwp::ThermoStatus::InvalidComposition && rho==R(-23));
 y[0]=R(.3);y[1]=R(.7);species[1].minTemperature=500;
 acceptedEos(R(101325),R(300),y);
 assert(ugkwp::mixtureDensityFromPressureTemperature(std::numeric_limits<R>::infinity(),R(300),y,view,rho)==ugkwp::ThermoStatus::NonFiniteInput && std::isinf(rho) && rho>0);
}
int main(){check<double>();}
''')
    binary=tmp_path/"probe"
    build=subprocess.run(["g++","-std=c++17","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
