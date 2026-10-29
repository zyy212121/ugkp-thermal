"""Actual gas caloric capacity and valid-domain energy floor, independent datum."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "gasTransport/ParticleHeatReservoir.H"
#include <cassert>
#include <cmath>
#include <initializer_list>
struct Legacy {};
struct Mixture {int nCells=1;double rho[1]={1};ugkwp::GasSpeciesState<double,2> gasSpecies;};
int main(){
 Legacy old;assert(ugkwp::particleGasHeatCapacity(old,0,.2,123.)==123.);
 assert(ugkwp::particleGasMinimumInternalEnergy(old,0,-456.)==-456.);
 Mixture s;double partial[2]={.9,.1},cp[1],R[1],coefficients[6];
 ugkwp::SpeciesThermoData<double> meta[2];
 for(int k=0;k<2;++k){meta[k].molarMass=k?.044:.028;meta[k].minTemperature=k?200:100;meta[k].maxTemperature=3000;meta[k].coefficientOffset=3*k;}
 s.gasSpecies.rho=partial;s.gasSpecies.heatCapacity=cp;s.gasSpecies.gasConstant=R;
 s.gasSpecies.thermo.species=meta;s.gasSpecies.thermo.coefficients=coefficients;s.gasSpecies.thermo.coefficientCount=6;
 s.gasSpecies.thermoControls.minTemperature=125;
 assert(ugkwp::particleGasHeatCapacity(s,0,.2,123.)==123.);
 s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;
 const double ra=8.31446261815324/.028,rb=8.31446261815324/.044;
 for(double slope:{0.,.2})for(double datum:{0.,-1e7,1e7})for(double lower:{200.,250.}) {
  coefficients[0]=1000;coefficients[1]=slope;coefficients[2]=datum;
  coefficients[3]=800;coefficients[4]=.5*slope;coefficients[5]=datum+120000;
  cp[0]=.9*(1000+slope*1000)+.1*(800+.5*slope*1000);R[0]=.9*ra+.1*rb;
  s.gasSpecies.thermoControls.minTemperature=lower==200?125:lower;
  const double expected=.9*(datum+(1000-ra)*lower+.5*slope*lower*lower)
      +.1*(datum+120000+(800-rb)*lower+.25*slope*lower*lower);
  assert(std::abs(ugkwp::particleGasMinimumInternalEnergy(s,0,987.)-expected)<4e-9);
  assert(ugkwp::particleGasHeatCapacity(s,0,.7,987.)==.7*(cp[0]-R[0]));
 }
}
'''

def test_shared_gas_contact_reservoir(tmp_path):
    source=tmp_path/'reservoir.cpp';source.write_text(SOURCE);exe=tmp_path/'reservoir'
    run=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    subprocess.run([exe],check=True)
