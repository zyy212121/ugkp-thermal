"""Double caloric inversion and bounded spacing cases with independent long-double evaluation."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT
SOURCE=r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <initializer_list>
using namespace ugkwp;
int main(){
 SpeciesThermoData<double> species[3];
 double coefficients[9]={1000.,0.019999999552965164,-1e6,1400.,0.029999999329447746,2e6,1100.,0.009999999776482582,-5e5};
 double molar[3]={0.02800000086426735,0.03200000151991844,0.04399999976158142};
 double masses[3]={.850943922996521,.12271755933761597,.21287618577480316};
 for(int s=0;s<3;++s){species[s].coefficientOffset=3*s;species[s].molarMass=molar[s];species[s].minTemperature=100;species[s].maxTemperature=3000;species[s].referencePressure=101325;}
 SpeciesThermoView<double,3> thermo;thermo.species=species;thermo.coefficients=coefficients;thermo.coefficientCount=9;
 auto reference=[&](long double T){long double e=0;for(int s=0;s<3;++s){
   const long double a=coefficients[3*s],b=coefficients[3*s+1],formation=coefficients[3*s+2];
   e+=static_cast<long double>(masses[s])*(formation+T*(a+.5L*b*T)-static_cast<long double>(universalGasConstant<double>())/molar[s]*T);
 }return e;};
 double savedMasses[3],savedCoefficients[9];std::memcpy(savedMasses,masses,sizeof masses);std::memcpy(savedCoefficients,coefficients,sizeof coefficients);
 ThermoInversionControls<double> tight;tight.relativeEnergyTolerance=1e-12;tight.absoluteTemperatureTolerance=1e-11;tight.relativeTemperatureTolerance=1e-12;
 // These double literals preserve the exact promoted historical binary32 inputs.
 // Original cancellation target plus range endpoints, each with five neighbors
 // around the independently solved root. No representable-lattice sweep.
 for(double target:{-17978.841796875,double(reference(100.1)),double(reference(1000.)),double(reference(2999.))}){
   long double lowerRoot=100,upperRoot=3000;
   for(int iteration=0;iteration<96;++iteration){const long double mid=(lowerRoot+upperRoot)/2;
     if(reference(mid)<target)lowerRoot=mid;else upperRoot=mid;
   }
   const long double independentRoot=(lowerRoot+upperRoot)/2;
   const double nearest=double(independentRoot);double root=737.88739013671875;
   assert(invertMixtureEnergy(masses,target,thermo,tight,root)==ThermoStatus::Success);
   const double tolerance=tight.absoluteEnergyTolerance+tight.relativeEnergyTolerance*std::abs(target);
   assert(std::abs(reference(root)-target)<=tolerance+1e-9L);
   assert(std::abs(static_cast<long double>(root)-independentRoot)<1e-8L);
   if(target==-17978.841796875)assert(std::abs(root-737.8405347741046)<1e-9);
   double neighbors[5];neighbors[2]=nearest;
   neighbors[1]=std::nextafter(nearest,-std::numeric_limits<double>::infinity());
   neighbors[0]=std::nextafter(neighbors[1],-std::numeric_limits<double>::infinity());
   neighbors[3]=std::nextafter(nearest,std::numeric_limits<double>::infinity());
   neighbors[4]=std::nextafter(neighbors[3],std::numeric_limits<double>::infinity());
   for(int i=0;i<5;++i){const double T=neighbors[i];
     if(i)assert(neighbors[i-1]<T&&reference(neighbors[i-1])<reference(T));
     assert(std::abs(static_cast<long double>(mixtureEnergy(masses,T,thermo))-reference(T))<2e-9L);
   }
 }
 assert(tight.relativeEnergyTolerance==1e-12&&tight.relativeTemperatureTolerance==1e-12);
 assert(std::memcmp(savedMasses,masses,sizeof masses)==0);assert(std::memcmp(savedCoefficients,coefficients,sizeof coefficients)==0);
}
'''
@pytest.mark.parametrize('optimization',['-O0','-O2'])
def test_double_caloric_roots_and_bounded_neighbors_keep_tight_controls(tmp_path,optimization):
    source=tmp_path/'precision.cpp';source.write_text(SOURCE);exe=tmp_path/'precision'
    build=subprocess.run(['g++','-std=c++14',optimization,'-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
