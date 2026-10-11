"""Retain strict controls when FP32 formation-energy cancellation is limiting.

Minimized from a three-species periodic SSPRK3 review case. This characterizes
non-contracting host FP32 arithmetic; it does not impose a tolerance floor or
claim that CUDA fused arithmetic produces the same residual lattice.
"""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=r'''
#include "common/gasTransport/MixtureThermo.H"
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
using namespace ugkwp;
int main(){
 SpeciesThermoData<float> sf[3];
 float coefficients[9]={1000.f,.02f,-1e6f,1400.f,.03f,2e6f,1100.f,.01f,-5e5f};
 float molar[3]={.028f,.032f,.044f};
 float masses[3]={.850943922996521f,.12271755933761597f,.21287618577480316f};
 for(int s=0;s<3;++s){sf[s].coefficientOffset=3*s;sf[s].molarMass=molar[s];sf[s].minTemperature=100;sf[s].maxTemperature=3000;sf[s].referencePressure=101325;}
 SpeciesThermoView<float,3> tf;tf.species=sf;tf.coefficients=coefficients;tf.coefficientCount=9;
 const float target=-17978.841796875f;
 ThermoInversionControls<float> controls;controls.relativeEnergyTolerance=2e-6f;controls.relativeTemperatureTolerance=2e-6f;
 float output=737.88739013671875f;const float initial=output;
 float savedMasses[3],savedCoefficients[9];std::memcpy(savedMasses,masses,sizeof masses);std::memcpy(savedCoefficients,coefficients,sizeof coefficients);
 assert(invertMixtureEnergy(masses,target,tf,controls,output)==ThermoStatus::IterationLimit);
 assert(std::memcmp(&initial,&output,sizeof output)==0);
 assert(std::memcmp(savedMasses,masses,sizeof masses)==0);
 assert(std::memcmp(savedCoefficients,coefficients,sizeof coefficients)==0);
 // Exhaustive local representable-temperature search, not a sparse sampling.
 const float tolerance=controls.absoluteEnergyTolerance+controls.relativeEnergyTolerance*std::abs(target);
 float best=std::numeric_limits<float>::max(),bestT=0;
 for(float T=737.f;T<739.f;T=std::nextafter(T,1000.f)){
   const float residual=std::abs(mixtureEnergy(masses,T,tf)-target);
   if(residual<best){best=residual;bestT=T;}
 }
 assert(best==.041015625f);assert(best>tolerance);assert(bestT==737.84051513671875f);
 // Promote the ORIGINAL FP32 metadata, rather than substituting idealized
 // decimal coefficients. The physical caloric root is temperature-representable
 // within the requested tolerances; cancellation in FP32 evaluation is limiting.
 SpeciesThermoData<double> sd[3];double md[3],cd[9];
 for(int s=0;s<3;++s){sd[s].coefficientOffset=3*s;sd[s].molarMass=double(sf[s].molarMass);sd[s].minTemperature=100;sd[s].maxTemperature=3000;sd[s].referencePressure=101325;md[s]=double(masses[s]);}
 for(int i=0;i<9;++i)cd[i]=double(coefficients[i]);
 SpeciesThermoView<double,3> td;td.species=sd;td.coefficients=cd;td.coefficientCount=9;
 ThermoInversionControls<double> tight;tight.relativeEnergyTolerance=1e-12;tight.absoluteTemperatureTolerance=1e-11;tight.relativeTemperatureTolerance=1e-12;
 double root=double(initial);assert(invertMixtureEnergy(md,double(target),td,tight,root)==ThermoStatus::Success);
 assert(std::abs(root-737.8405347741046)<1e-9);
 const double lower=double(std::nextafter(bestT,0.f)),nearest=double(bestT),upper=double(std::nextafter(bestT,1000.f));
 const double belowResidual=mixtureEnergy(md,lower,td)-double(target);
 const double nearResidual=mixtureEnergy(md,nearest,td)-double(target);
 const double aboveResidual=mixtureEnergy(md,upper,td)-double(target);
 assert(std::abs(nearResidual+.01863512289)<1e-8);
 assert(std::abs(belowResidual)>double(tolerance));assert(std::abs(nearResidual)<double(tolerance));assert(std::abs(aboveResidual)>double(tolerance));
 assert(std::abs(nearest-root)<double(controls.absoluteTemperatureTolerance)+double(controls.relativeTemperatureTolerance)*std::abs(root));
}
'''
@pytest.mark.parametrize('optimization',['-O0','-O2'])
def test_fp32_evaluator_limit_rejects_without_relaxing_controls(tmp_path,optimization):
    source=tmp_path/'precision.cpp';source.write_text(SOURCE);exe=tmp_path/'precision'
    build=subprocess.run(['g++','-std=c++14',optimization,'-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-I',str(ROOT),str(source),'-o',str(exe)],capture_output=True,text=True)
    assert build.returncode==0,build.stderr
    run=subprocess.run([str(exe)],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
