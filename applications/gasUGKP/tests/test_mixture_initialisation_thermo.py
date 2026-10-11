"""Common EOS initialization is shared by native application frontends."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]

def test_initialisation_uses_common_eos_and_preserves_failed_output(tmp_path):
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
 const R saved=rho;
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(-1),R(300),y,view,rho)!=ugkwp::ThermoStatus::Success && rho==saved);
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(5000),y,view,rho)!=ugkwp::ThermoStatus::Success && rho==saved);
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),masses,view,rho)!=ugkwp::ThermoStatus::Success && rho==saved);
 y[0]=R(-.1);y[1]=R(1.1);
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),y,view,rho)!=ugkwp::ThermoStatus::Success && rho==saved);
 y[0]=y[1]=0;assert(!std::isfinite(ugkwp::mixtureGasConstant(y,view)));
 y[0]=R(.3);y[1]=R(.7);species[1].minTemperature=500;
 assert(ugkwp::mixtureDensityFromPressureTemperature(R(101325),R(300),y,view,rho)!=ugkwp::ThermoStatus::Success && rho==saved);
}
int main(){check<float>();check<double>();}
''')
    binary=tmp_path/"probe"
    build=subprocess.run(["g++","-std=c++14","-Wall","-Wextra","-Werror","-I",str(ROOT),str(source),"-o",str(binary)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(binary)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
