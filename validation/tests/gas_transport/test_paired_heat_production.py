"""Execute production contact heat transaction kernels with HOST launch adapters."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest
ROOT=SOURCE_ROOT
SOURCE=r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#define __device__
#define __global__
#define __forceinline__ inline
#include "wall/GpuColdWallSolidification.H"
namespace Foam{namespace gpuThermal{constexpr unsigned char particleWallMobile=0;}}
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalCarry.H"
#include "GpuPairedParticleHeat.H"
#define GPU_PAIRED_PARTICLE_HEAT 1
#include "wall/GpuColdWallThermalCarry.cuh"
using GpuTime=double;
ThreadIndex gridDim{1};
void atomicAdd(double*p,double x){*p+=x;}
struct DeviceState{
 int particleStuckModelConfigured=1;
 ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
 GpuReal gasPrClamped=.9*.9*.9,gasCp=1004.5;
 int nCells=1,nFaces=1,particleCapacity=4,count=2;int*particleCountDevice=&count;
 int pStatus[4]={1,1},pCellId[4]={},pStuckFaceId[4]={-1,-1},compactPStuckFaceId[4]={};
 unsigned char pStuck[4]={1,1},compactPStuck[4]={},particleStuckCandidateMask[1]={};
 GpuReal pT[4]={2800,3400},compactPT[4]={},pm[4]={100,100},pd[4]={.001,.002};
 GpuReal pux[4]={40000,40000},puy[4]={},puz[4]={};
 float pColdNodeSpecificEnthalpy[32]={},pCold2DNodeSpecificEnthalpy[256]={},compactPColdNodeSpecificEnthalpy[32]={},compactPCold2DNodeSpecificEnthalpy[256]={};
 double pThermalEnthalpyResidual[4]={},compactPThermalEnthalpyResidual[4]={};
 GpuReal gasBoundaryT[1]={300};
 GpuReal couplingRhoOld[1]={1},couplingUxOld[1]={0},couplingUyOld[1]={},couplingUzOld[1]={},couplingTgasOld[1]={3200};
 GpuReal Tgas[1]={3200},p[1]={},Ux[1]={40000},Uy[1]={},Uz[1]={};
 GpuReal rho[1]={1},rhoUx[1]={40000},rhoUy[1]={},rhoUz[1]={},V[1]={1},epsGPrev[1]={.9};
 double rhoE[1]={800000000.+717.5*3200},rhoENext[1]={};double A[1]={},Q[1]={};
 double*particleHeatCoefficient=A,*particleHeatBudget=Q;
 double particleContactHeatCandidate[4]={};unsigned char particleContactHeatEpoch[4]={};
 GpuReal rhoMin=1e-6,TgasMin=1,TpMin=1,TpMax=6000,particleDiameterFallback=.001;
 GpuReal rhoSolid=3000,gasMu=1e-5,gasPrOneThird=.9,Rgas=287,gammaGas=1.4;
 Foam::gpuThermal::ColdWallSolidificationParameters coldWallSolidificationParameters{};
};
GpuReal molecularGasConductivity(const DeviceState&){return GpuReal(.04);}
#include "GpuParticleCaloricDevice.cuh"
#include "GpuPairedParticleHeat.cuh"
#include "paired_heat_reference.cuh"
void ck(bool b,const char*m){if(!b){puts(m);abort();}}
long double inventory(const DeviceState&s){
 long double value=(long double)pairedGasPhysicalVolume(s,0)*s.rhoE[0];
 for(int i=0;i<s.count;++i)value+=(long double)s.pm[i]*((long double)particleMaterialEnthalpyBaseDevice(s,i)+s.pThermalEnthalpyResidual[i]);
 return value;
}
void advance(DeviceState&s,double dt){
 clearPairedParticleHeatKernel(&s);reducePairedParticleHeatKernel(&s,dt);acceptPairedParticleHeatKernel(&s);
 for(int i=0;i<s.count;++i)publishPairedParticleGasHeat(s,i,dt);
 settlePairedParticleHeatKernel(&s);
}
void reference(double duration,long double (&t)[3]){
 const long double dt=duration/20000.;
 DeviceState initial;
 const long double Cg=pairedGasPhysicalVolume(initial,0)*double(initial.Rgas)/(double(initial.gammaGas)-1.);
 const long double conductivity=double(molecularGasConductivity(initial));
 const long double g0=6.L*2.L*conductivity*100.L/(double(initial.rhoSolid)*double(initial.pd[0])*double(initial.pd[0]));
 const long double g1=6.L*2.L*conductivity*100.L/(double(initial.rhoSolid)*double(initial.pd[1])*double(initial.pd[1]));
 auto rhs=[&](const long double*x,long double*d){
  auto cp=[](long double z){z=std::max(2327.L,std::min(4000.L,z));return (153.5L+.0031L*(z-2327.L))/.101961276L;};
  const long double q0=g0*(x[0]-x[1]),q1=g1*(x[0]-x[2]);
  d[0]=-(q0+q1)/Cg;d[1]=q0/(100.L*cp(x[1]));d[2]=q1/(100.L*cp(x[2]));
 };
 for(int n=0;n<20000;++n){long double k1[3],k2[3],k3[3],k4[3],x[3];rhs(t,k1);
  for(int j=0;j<3;++j)x[j]=t[j]+dt*k1[j]/2;rhs(x,k2);
  for(int j=0;j<3;++j)x[j]=t[j]+dt*k2[j]/2;rhs(x,k3);
  for(int j=0;j<3;++j)x[j]=t[j]+dt*k3[j];rhs(x,k4);
  for(int j=0;j<3;++j)t[j]+=dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
 }
}
int main(){
 blockDim.x=1;
 DeviceState zero;zero.pThermalEnthalpyResidual[0]=1e-4;const auto old=zero;
 advance(zero,0.);ck(zero.rhoE[0]==old.rhoE[0]&&zero.pT[0]==old.pT[0]&&zero.pThermalEnthalpyResidual[0]==old.pThermalEnthalpyResidual[0],"dt0 mutated physical authority");
 for(double dt:{1e-8,.001,1.,1e6}){
  DeviceState s;const long double before=inventory(s);const double gas=s.rhoE[0];
  advance(s,dt);const long double error=inventory(s)-before;
  DeviceState separate;separate.rhoE[0]=gas-s.Q[0]/pairedGasPhysicalVolume(s,0);
  recoverGasPrimitiveCell(separate,0);
  ck(s.rhoE[0]==separate.rhoE[0]&&s.Tgas[0]==separate.Tgas[0]&&s.p[0]==separate.p[0]
    &&s.Ux[0]==separate.Ux[0]&&s.Uy[0]==separate.Uy[0]&&s.Uz[0]==separate.Uz[0],"fused settlement gas recovery differs from separate actual EOS");
  ck(std::abs(error)<2e-6L,"actual paired thermal inventory leaked");
  ck(particleEffectiveTemperatureDevice(s,0)>2800,"microheat missing from future temperature");
  ck(particleEffectiveTemperatureDevice(s,1)<3400,"cooling particle lost accepted heat");
  ck(particleEffectiveTemperatureDevice(s,0)<=3200&&particleEffectiveTemperatureDevice(s,1)>=3200,"particle overshot frozen bath");
  ck(s.rhoE[0]>=pairedGasEnergyFloor(s,0),"gas floor violated");
  printf("dt=%.9g gas_delta=%.17g budget_error=%.17Lg p0=%.17g carry0=%.17g\n",dt,s.rhoE[0]-gas,error,double(s.pT[0]),s.pThermalEnthalpyResidual[0]);
 }
 // Independent coupled caloric ODE reference, with refreshed gas cache at
 // each whole physical step; mechanical fields and geometry are constant.
 long double ref[3]={3200,2800,3400};reference(.02,ref);
 double previous=1e300;
 for(int steps:{2,4,8,16,32}){
  DeviceState s;const double cv=double(s.Rgas)/(double(s.gammaGas)-1.);
  s.rhoE[0]=800000000.+cv*3200.;
  for(int n=0;n<steps;++n){s.couplingTgasOld[0]=GpuReal((s.rhoE[0]-800000000.)/cv);advance(s,.02/steps);}
  const double error=std::abs((s.rhoE[0]-800000000.)/cv-double(ref[0]))+
   std::abs(particleEffectiveTemperatureDevice(s,0)-double(ref[1]))+
   std::abs(particleEffectiveTemperatureDevice(s,1)-double(ref[2]));
  ck(error<previous*.75,"paired heat failed first-order step refinement");previous=error;
  printf("refinement_steps=%d temperature_L1_error=%.17g\n",steps,error);
 }
 // Actual production order, not the earlier gas-first mathematical prototype.
 DeviceState cumulative;const long double initial=inventory(cumulative);
 long double roundingBound=0.,absoluteExchange=0.;
 for(int n=0;n<100000;++n){
  const double gas0=cumulative.rhoE[0],volume=pairedGasPhysicalVolume(cumulative,0);
  double base[2],carry[2];for(int i=0;i<2;++i){base[i]=particleMaterialEnthalpyBaseDevice(cumulative,i);carry[i]=cumulative.pThermalEnthalpyResidual[i];}
  advance(cumulative,1e-8);
  // Derived publication bound: final double gas subtraction, divide and
  // reduction rounding, plus compensated particle-component arithmetic.
  const double eps=std::numeric_limits<double>::epsilon();
  roundingBound+=.5L*(std::nextafter(gas0,INFINITY)-gas0)*volume;
  roundingBound+=4.L*eps*std::abs(cumulative.Q[0]);
  for(int i=0;i<2;++i){
   const double db=particleMaterialEnthalpyBaseDevice(cumulative,i)-base[i];
   const double r=cumulative.pThermalEnthalpyResidual[i];
   const long double actual=(long double)cumulative.pm[i]*((long double)db+r-carry[i]);
   absoluteExchange+=std::abs(actual);
   roundingBound+=16.L*eps*cumulative.pm[i]*(std::abs(db)+std::abs(r)+std::abs(carry[i])+double(std::abs(actual)/cumulative.pm[i]));
  }
 }
 const long double error=inventory(cumulative)-initial;
 ck(std::abs(error)<roundingBound,"production-order error exceeds summed local publication roundoff");
 ck(cumulative.pT[0]!=GpuReal(2800),"cumulative stored temperature never changed");
 printf("production_order_100000_budget_error=%.17Lg bound=%.17Lg absolute_exchange=%.17Lg relative_exchange_error=%.17Lg\n",error,roundingBound,absoluteExchange,error/absoluteExchange);
 // Stiff unequal masses/diameters, with the cache refreshed in the real step
 // order. Compare final equilibrium from an independent conserved-H solve.
 DeviceState stiff;stiff.pm[0]=1;stiff.pm[1]=1000;stiff.pd[0]=GpuReal(.00001);
 stiff.pT[0]=1000;stiff.pT[1]=5000;stiff.couplingTgasOld[0]=3000;
 const double cv=double(stiff.Rgas)/(double(stiff.gammaGas)-1.),vol=pairedGasPhysicalVolume(stiff,0);
 stiff.rhoE[0]=800000000.+cv*3000.;const auto law=ugkwp::ParticleCaloric{};
 const long double thermal=vol*cv*3000.L+law.enthalpy(1000)+1000.L*law.enthalpy(5000);
 double lo=1000,hi=5000;for(int it=0;it<80;++it){const double t=.5*(lo+hi);
  if(vol*cv*t+1001.L*law.enthalpy(t)<thermal)lo=t;else hi=t;}
 const double equilibrium=.5*(lo+hi);
 for(int n=0;n<100000;++n){
  stiff.couplingTgasOld[0]=GpuReal((stiff.rhoE[0]-800000000.)/cv);advance(stiff,1000.);
  ck(stiff.rhoE[0]>=pairedGasEnergyFloor(stiff,0),"stiff gas left positive caloric domain");
  ck(std::isfinite(particleEffectiveTemperatureDevice(stiff,0))&&std::isfinite(particleEffectiveTemperatureDevice(stiff,1)),"stiff particle caloric failure");
 }
 const double equilibriumError=std::max(std::abs((stiff.rhoE[0]-800000000.)/cv-equilibrium),
  std::max(std::abs(particleEffectiveTemperatureDevice(stiff,0)-equilibrium),std::abs(particleEffectiveTemperatureDevice(stiff,1)-equilibrium)));
 const double storageUlp=std::nextafter(GpuReal(equilibrium),GpuReal(INFINITY))-GpuReal(equilibrium);
 ck(equilibriumError<4.*storageUlp+1e-6,"stiff repeated steps fail represented equilibrium");
 printf("stiff_unequal_equilibrium_error=%.17g storage_ulp=%.17g\n",equilibriumError,storageUlp);
}
'''
def test_actual_production_heat_transaction(tmp_path):
 p=tmp_path/'production.cpp';prefix=((VALIDATION_ROOT / 'tests/gas_transport/gas_state_probe.cpp')).read_text().split('void hashBytes',1)[0];p.write_text(prefix+SOURCE);exe=tmp_path/'production'
 r=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),'-I'+str(ROOT/'common/gasNumerics'),'-I'+str((VALIDATION_ROOT / 'tests/gas_transport')),str(p),'-o',str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
 r=subprocess.run([str(exe)],capture_output=True,text=True)
 assert r.returncode==0,r.stdout+r.stderr
 print(r.stdout,end='')

EOS_FLOOR_MAIN=r'''
int main(){
 std::setvbuf(stdout,nullptr,_IONBF,0);
 blockDim.x=1;
 for(int mode=0;mode<5;++mode){
  DeviceState s;s.count=1;s.particleCountDevice=&s.count;s.rhoMin=GPU_R(1);
  s.Rgas=GPU_R(.0001);s.rho[0]=GPU_R(1);s.rhoUx[0]=0;s.pux[0]=0;s.Ux[0]=0;
  s.couplingUxOld[0]=0;s.pT[0]=GPU_R(1000);s.pm[0]=GPU_R(1);s.couplingTgasOld[0]=GPU_R(4000);
  if(mode==1){s.Rgas=GPU_R(287);s.rhoMin=GPU_R(1.e-6);}
  if(mode==2){s.rho[0]=GPU_R(1.3);s.rhoUx[0]=GPU_R(1.17);s.rhoUy[0]=GPU_R(-.31);}
  if(mode==3)s.gammaGas=std::nextafter(GpuReal(1),GpuReal(2));
  if(mode==4)s.gammaGas=GPU_R(1); // Scalar clamp contract only: zero gamma-1 has no positive-pressure EOS state.
  const double density=double(s.rho[0]);
  const double x=double(s.rhoUx[0])/density,y=double(s.rhoUy[0])/density,z=double(s.rhoUz[0])/density;
  const double kinetic=.5*density*(x*x+y*y+z*z);
  const double denominator=std::max(double(s.gammaGas)-1.,double(OfSmall));
  const double minInternal=std::max(double(s.rhoMin),density*double(s.Rgas)*double(s.TgasMin)/denominator);
  const double eosFloor=kinetic+minInternal;
  const double floor=pairedGasEnergyFloor(s,0);
  ck(std::isfinite(floor)&&floor==eosFloor,"paired floor differs from actual legacy recovery floor");
  s.rhoE[0]=eosFloor+(mode==3||mode==4?1.:.1);
  const double e0=s.rhoE[0];const long double total0=inventory(s);
  clearPairedParticleHeatKernel(&s);reducePairedParticleHeatKernel(&s,1.e6);acceptPairedParticleHeatKernel(&s);
  ck(std::isfinite(s.A[0])&&s.A[0]>=0.&&s.A[0]<=1.,"clamped gamma denominator produced invalid acceptance");
  if(mode==4)continue;
  publishPairedParticleGasHeat(s,0,1.e6);
  ck(s.Q[0]>0.,"low-budget heat acceptance disappeared");
  const double expected=e0-s.Q[0]/pairedGasPhysicalVolume(s,0);
  ck(expected>=eosFloor,"paired accepted heat crossed actual EOS floor");
  settlePairedParticleHeatKernel(&s);
  ck(s.rhoE[0]==expected,"actual gas recovery added energy after paired acceptance");
  const double acceptedEnergy=s.rhoE[0];recoverGasPrimitiveCell(s,0);
  ck(s.rhoE[0]==acceptedEnergy,"second actual recovery rewrote accepted authority");
  const long double tolerance=2.L*(std::nextafter(e0,INFINITY)-e0)*pairedGasPhysicalVolume(s,0)+1.e-9L;
  ck(std::abs(inventory(s)-total0)<=tolerance,"EOS floor correction created paired energy");
  printf("EOS_floor_mode=%d floor=%.17g E0=%.17g accepted=%.17g recovery_added=%.17g\n",mode,floor,e0,s.Q[0],s.rhoE[0]-expected);
 }
}
'''

def test_actual_paired_floor_matches_gas_recovery(tmp_path):
    prefix=((VALIDATION_ROOT / 'tests/gas_transport/gas_state_probe.cpp')).read_text().split('void hashBytes',1)[0]
    source=SOURCE.split('int main(){',1)[0]+EOS_FLOOR_MAIN
    path=tmp_path/'floor.cpp';path.write_text(prefix+source);exe=tmp_path/'floor'
    result=subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gpu'),'-I'+str(ROOT/'common/gasNumerics'),
        '-I'+str((VALIDATION_ROOT / 'tests/gas_transport')),str(path),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout,end='')
