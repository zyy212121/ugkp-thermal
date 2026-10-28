"""Production cell-average/mobile and contact heat paths in one cell (HOST adapters)."""
import subprocess
from test_paired_heat_production import SOURCE, ROOT, VALIDATION_ROOT

EXTRA = r'''
 int particleStuckModelConfigured=1,solveParticleTemperature=1,particleGasHeatTransferModelId=1;
 GpuReal epsSMin=1e-12,pTheta[4]{},thetaDragAlpha[1]{1.};
 double mobileMoments[6]{},*particleMobileHeatMoments=mobileMoments;
 GpuReal momRhoP[1]{},momRhoUPx[1]{},momRhoUPy[1]{},momRhoUPz[1]{},momRhoPD[1]{},momRhoHpP[1]{};
'''
HARNESS = r'''
#include "gpu/GpuCouplingMath.H"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include "operators/particleSpecificHeatDevice.cuh"
#include "operators/particleTemperatureFromSpecificEnthalpyDevice.cuh"
#define GPU_OPERATOR_THERMAL 1
GpuReal sqr3(GpuReal x,GpuReal y,GpuReal z){return x*x+y*y+z*z;}
GpuReal solidEpsFromMomentDevice(const DeviceState&s,int c){return s.momRhoP[c]/s.rhoSolid;}
#include "GpuMobileParticleHeat.cuh"
#include "operators/applyEulerianParticleMaterialHeatKernel.cuh"
void moments(DeviceState&s){
 s.momRhoP[0]=s.momRhoUPx[0]=s.momRhoUPy[0]=s.momRhoUPz[0]=s.momRhoPD[0]=s.momRhoHpP[0]=0.;
 for(int i=0;i<s.count;++i){s.momRhoP[0]+=s.pm[i]/s.V[0];s.momRhoUPx[0]+=s.pm[i]*s.pux[i]/s.V[0];
 s.momRhoPD[0]+=s.pm[i]*s.pd[i]/s.V[0];s.momRhoHpP[0]+=materialEnthalpyMomentFromParticle(s,i,s.pm[i],s.pT[i],0)/s.V[0];}
 s.epsGPrev[0]=1.-solidEpsFromMomentDevice(s,0);
}
void init(DeviceState&s){
 s.count=2;s.particleCountDevice=&s.count;s.particleMobileHeatMoments=s.mobileMoments;
 s.pStuck[0]=s.pStuck[1]=0;s.pux[0]=s.pux[1]=0;s.rhoUx[0]=s.Ux[0]=0;
 s.couplingTgasOld[0]=3200;s.rhoE[0]=717.5*3200;moments(s);
}
long double predictedGasHeat(const DeviceState&s,double dt){
 const long double mass=s.mobileMoments[0],h=s.mobileMoments[5];if(mass<=1e-9)return 0.;
 const long double t=particleTemperatureFromSpecificEnthalpyDevice(double(h/(mass+OfSmall)));
 const long double cp=particleSpecificHeatDevice(double(t)),d=s.mobileMoments[4]/mass;
 const long double cg=(1.-s.momRhoP[0]/s.rhoSolid)*s.couplingRhoOld[0]*s.Rgas/(s.gammaGas-1.);
 const long double cs=mass*cp,rate=12.L*molecularGasConductivity(s)/(s.rhoSolid*cp*d*d);
 return cg*cs/(cg+cs)*(s.couplingTgasOld[0]-t)*(-expm1l(-rate*(1.+cs/cg)*dt));
}
int main(){
 blockDim.x=gridDim.x=1;
 for(int contacts=0;contacts<3;++contacts)for(double dt:{0.,1e-10,.001}){
  DeviceState s;init(s);for(int i=0;i<contacts;++i)s.pStuck[i]=1;
  prepareMobileParticleHeatMomentsKernel(&s);subtractContactParticleHeatMomentsKernel(&s);
  ck(std::abs(s.mobileMoments[0]-(2-contacts)*100.)<1e-12,"contact mass entered cell-average heat");
  const double e0=s.rhoE[0],eps=s.epsGPrev[0];const auto expected=predictedGasHeat(s,dt);
  applyEulerianParticleMaterialHeatKernel(&s,dt);
  ck(fabsl((long double)s.rhoE[0]-e0+expected/eps)<3e-9L,"cell-average gas heat differs from independent finite-capacity solution");
  clearPairedParticleHeatKernel(&s);reducePairedParticleHeatKernel(&s,dt);
  for(int i=0;i<2;++i){ck(s.particleContactHeatEpoch[i]==(i<contacts),"contact epoch incorrect");
    ck((i<contacts)||s.particleContactHeatCandidate[i]==0.,"mobile particle entered contact heat budget");}
  acceptPairedParticleHeatKernel(&s);
  const long double pairedBefore=inventory(s);
  for(int i=0;i<contacts;++i)publishPairedParticleGasHeat(s,i,dt);
  settlePairedParticleHeatKernel(&s);
  ck(fabsl(inventory(s)-pairedBefore)<2e-6L,"contact heat does not independently close");
  for(int i=contacts;i<2;++i){
   const double before=particleEffectiveTemperatureDevice(s,i);
   const double cp=particleCaloricDevice(s,i).specificHeat(before);
   const double expectedT=particleTemperatureAfterGasRelaxation(s,0,before,s.couplingTgasOld[0],0.,s.pd[i],s.rhoSolid*cp,dt);
   relaxMobileParticleThermalState(s,i,0,0.,s.pd[i],dt);
   ck(std::abs(particleEffectiveTemperatureDevice(s,i)-expectedT)<3e-12,"mobile temperature differs from gas reference exponential relaxation");
  }
 }
 // Free-cell macro targets are not replaced by sampled micro moments.
 DeviceState macro;init(macro);macro.momRhoUPx[0]=17.;macro.momRhoHpP[0]+=123.;
 prepareMobileParticleHeatMomentsKernel(&macro);subtractContactParticleHeatMomentsKernel(&macro);
 ck(macro.mobileMoments[1]==17.&&macro.mobileMoments[5]==macro.momRhoHpP[0],"analytical macro target discarded");
 // At sub-temperature-ulp increments, retained caloric carry remains nonzero.
 DeviceState tiny;init(tiny);const double oldT=tiny.pT[0];
 relaxMobileParticleThermalState(tiny,0,0,0.,tiny.pd[0],1e-18);
 ck(tiny.pT[0]==oldT&&tiny.pThermalEnthalpyResidual[0]>0.,"sub-ulp mobile heat lost");
 // Quantify the accepted macro/micro approximation; it must refine with dt.
 double previous=1e300;
 for(int steps:{4,8,16,32}){
  DeviceState s;init(s);s.count=1;s.particleStuckModelConfigured=0;moments(s);
  const long double initial=inventory(s);const double duration=.002;
  for(int n=0;n<steps;++n){
   s.couplingTgasOld[0]=s.rhoE[0]/717.5;moments(s);
   applyEulerianParticleMaterialHeatKernel(&s,duration/steps);
   relaxMobileParticleThermalState(s,0,0,0.,s.pd[0],duration/steps);
  }
  const double drift=double(fabsl(inventory(s)-initial));
  ck(drift>0.&&drift<previous*.65,"accepted cell/micro heat error does not refine");previous=drift;
  printf("steps=%d macro_micro_energy_drift=%.17g\n",steps,drift);
 }
 puts("mobile/contact heat production regression passed");
}
'''

def test_mobile_contact_heat_production(tmp_path):
 prefix=(VALIDATION_ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]
 source=SOURCE.split('int main(){',1)[0].replace('struct DeviceState{','struct DeviceState{'+EXTRA)
 source=source.replace('GpuReal molecularGasConductivity(const DeviceState&){return GpuReal(.04);}','')
 cpp=tmp_path/'mobile.cpp';cpp.write_text(prefix+source+HARNESS);exe=tmp_path/'mobile'
 command=['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(VALIDATION_ROOT/'tests/gas_transport'),str(cpp),'-o',str(exe)]
 result=subprocess.run(command,capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 result=subprocess.run([str(exe)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 print(result.stdout,end='')


def test_released_contact_is_not_heated_twice(tmp_path):
 # Dispatch-only probe: the numeric laws are exercised separately above.
 # A 1D contact can become mobile before the mobile relaxation pass.
 from test_actual_particle_energy_budget import particle_fixture
 source=particle_fixture().split('void printState',1)[0]
 source=source.replace('struct DeviceState:State{','struct DeviceState:State{\n int particleStuckModelConfigured=1; unsigned char particleContactHeatEpoch[1]{1};')
 source=source.replace('#include "GpuThermalParticleRelaxation.cuh"','''
#define GPU_CELL_MEAN_MOBILE_HEAT 1
int mobileHeatCalls=0;
void relaxMobileParticleThermalState(DeviceState&,int,int,Real,Real,Time){++mobileHeatCalls;}
#include "GpuThermalParticleRelaxation.cuh"
''')
 source+=r'''
int main(){DeviceState s;setup(s);s.dragModelId=0;s.pStuck[0]=0;
 relaxOneParticleToResidentGas(s,0,.001,0);ck(mobileHeatCalls==0,"released contact received a second gas heat update");
 s.particleContactHeatEpoch[0]=0;relaxOneParticleToResidentGas(s,0,.001,0);ck(mobileHeatCalls==1,"free particle missed heat update");
 s.solveParticleTemperature=0;relaxOneParticleToResidentGas(s,0,.001,0);ck(mobileHeatCalls==1,"disabled heat updated mobile particle");
}
'''
 prefix=(VALIDATION_ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes',1)[0]
 cpp=tmp_path/'epoch.cpp';cpp.write_text(prefix+source);exe=tmp_path/'epoch'
 result=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str(VALIDATION_ROOT/'tests/gas_transport'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
 result=subprocess.run([str(exe)],capture_output=True,text=True)
 assert result.returncode==0,result.stdout+result.stderr
