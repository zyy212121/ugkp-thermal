"""Nonlinear parcel calorics against an independent long-double bath balance."""
import pytest
from test_mixture_state import compile_probe
from test_steady_mixture_particle_coupling import paired_fixture


def fixture():
    return paired_fixture()+r'''
#include "ParticleCaloricHeatResponse.H"
#include "GpuParticleThermalCarry.H"
struct ThermalState:PairedState {
 unsigned char pStuck[4]={};double pThermalEnthalpyResidual[4]={};
 double thetaDragAlpha[2]={1,1};ugkwp::ParticleCaloric law;
};
using DeviceState=ThermalState;
ugkwp::ParticleCaloric particleCaloricDevice(const ThermalState&s,int){return s.law;}
double particleEffectiveTemperatureDevice(const ThermalState&s,int i){
 return s.law.temperature(s.law.enthalpy(s.pT[i])+s.pThermalEnthalpyResidual[i]);
}
#include "GpuParticleGasHeatRelaxation.cuh"
#include "GpuMixtureThermalParticleHeat.cuh"
// Independent evaluation of the historical caloric law, including endpoint
// trapezoid and phase interval. No production increment/derivative is used.
long double cp(long double t){t=std::max(2327.L,std::min(4000.L,t));return (153.5L+.0031L*(t-2327.L))/.101961276L;}
long double h(const ugkwp::ParticleCaloric&law,long double t){
 if(!law.phaseChange){const long double d=std::max(0.L,std::min(1673.L,t-2327.L));
  return t<=2327?cp(2327)*t:cp(2327)*2327+cp(2327)*d+.0031L/.101961276L*d*d/2+(t>4000?cp(4000)*(t-4000):0);}
 const long double ts=law.meltingTemperature-.5L*law.mushyRange,tl=law.meltingTemperature+.5L*law.mushyRange;
 const long double cm=.5L*(law.solidCp+cp(law.meltingTemperature))+law.latentHeat/law.mushyRange;
 if(t<=ts)return law.solidCp*t;
 if(t<=tl)return law.solidCp*ts+cm*(t-ts);
 return law.solidCp*ts+cm*law.mushyRange+.5L*(cp(tl)+cp(t))*(t-tl);
}
'''


@pytest.mark.parametrize("phase",[False,True])
@pytest.mark.parametrize("directory",["scan","csr","split"])
@pytest.mark.parametrize("dt",[1e-8,.02,2.])
def test_actual_kernel_nonlinear_root_conservation_and_contact_exclusion(tmp_path,phase,directory,dt):
    compile_probe(tmp_path,fixture()+r'''
int main(){ThermalState s;setup(s);s.law.phaseChange=PHASE;s.law.solidCp=900;s.law.latentHeat=8e5;
s.TpMin=100;s.TpMax=10000;s.pT[0]=10000;s.pT[1]=2400;s.pT[2]=700;s.pT[3]=500;
s.pStuck[1]=2;s.pThermalEnthalpyResidual[0]=1e-8;s.pThermalEnthalpyResidual[1]=.25;
s.gasSpecies.thermoControls.relativeEnergyTolerance=1e-14;s.gasSpecies.thermoControls.relativeTemperatureTolerance=1e-13;
auto* coeff=const_cast<Real*>(s.gasSpecies.thermo.coefficients);
for(int k=0;k<2;++k){coeff[3*k]=1000;coeff[3*k+1]=-.22;coeff[3*k+2]=-1e7;}
s.pm[0]=.002;
if(SPLIT){s.preBaseCellOffset[0]=0;s.preBaseCellOffset[1]=1;s.preBaseCellOffset[2]=2;
 s.cellParticleOffset[0]=0;s.cellParticleOffset[1]=1;s.cellParticleOffset[2]=1;
 s.sortedParticleIndex[0]=1;}
for(int c=0;c<2;++c){Real partial[2]={s.gasSpecies.rho[c],s.gasSpecies.rho[2+c]};
s.rhoUx[c]=s.rhoUy[c]=s.rhoUz[c]=0;s.Tgas[c]=1000;
s.rhoE[c]=ugkwp::mixtureEnergy(partial,1000.,s.gasSpecies.thermo);threadIdx.x=c;recoverGasPrimitivesKernel(&s);}
s.pux[0]=s.puy[0]=s.puz[0]=0;
const long double oldT=particleEffectiveTemperatureDevice(s,0),oldE=s.rhoE[0];
const long double volume=s.V[0]*s.epsGPrev[0],mass=s.pm[0];
const long double oldH=h(s.law,s.pT[0])+s.pThermalEnthalpyResidual[0];
const long double rate=12.L*.06L/(2500.L*cp(oldT)*.001L*.001L);
const long double a=-expm1l(-rate*DT),R=8.31446261815324L/.028L;
long double lo=100,hi=3000;
auto equation=[&](long double t){const long double d=t-1000;
 const long double pt=std::max(100.L,std::min(10000.L,oldT+a*(t-oldT)));
 return volume*((1000-.22L*1000-R)*d-.11L*d*d)+mass*(h(s.law,pt)-h(s.law,oldT));};
for(int it=0;it<180;++it){long double mid=(lo+hi)/2;if(equation(mid)>0)hi=mid;else lo=mid;}
const long double expected=(lo+hi)/2,expectedP=oldT+a*(expected-oldT);
threadIdx.x=0;mixtureThermalParticleHeatKernel(&s,DT,CSR,SPLIT);
ck(s.gasSpecies.cellStatus[0]==0,"nonlinear heat status");
closeValue(s.Tgas[0],expected,3e-8,"gas finite bath root");
closeValue(particleEffectiveTemperatureDevice(s,0),expectedP,3e-8,"particle exponential constitutive equation");
const long double newH=h(s.law,s.pT[0])+s.pThermalEnthalpyResidual[0];
closeValue(volume*((long double)s.rhoE[0]-oldE)+mass*(newH-oldH),0,2e-8,"actual two phase inventory");
ck(s.pT[1]==2400 && s.pThermalEnthalpyResidual[1]==.25,"contact heat applied through free source");
ck(s.pT[2]==700 && s.pT[3]==500,"foreign or inactive parcel changed");
ck(s.pT[0]<10000 && s.pT[0]>3000 && s.Tgas[0]>1000,"hot particle clipped to gas domain or zero exchange accepted");
}
'''.replace("PHASE",str(phase).lower()).replace("DT",repr(dt))
       .replace("CSR",str(directory!="scan").lower()).replace("SPLIT",str(directory=="split").lower()))


def test_caloric_derivative_is_derivative_of_existing_enthalpy(tmp_path):
    compile_probe(tmp_path,fixture()+r'''
int main(){ugkwp::ParticleCaloric law;law.phaseChange=true;law.solidCp=900;law.latentHeat=8e5;
for(double t:{500.,2320.,2340.,2800.,3990.,4500.,10000.}){
 const long double epsilon=.001L;
 const auto independent=(h(law,(long double)t+epsilon)-h(law,(long double)t-epsilon))/(2*epsilon);
 closeValue(law.enthalpyDerivative(t),independent,1e-7,"nonlinear derivative mismatch");
}
}
''')
