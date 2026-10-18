"""HOST execution of real sampling/correction kernels and fused paired heat.

The serial launch adapter checks the actual selected-stuck partition and compares
against an unfused reduction after the same production collision correction.
This establishes arithmetic/selection behavior, not CUDA scheduling or speed.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT


def function(source, marker):
    """Copy one complete production definition, without rewriting its body."""
    start = source.index(marker)
    body = source.index('{', start)
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end] + '\n'


EXTRA_STATE = r'''
 double pThermalEnthalpyResidual[16]{}, heatBudget[16]{}, heatCoefficient[16]{};
 double *particleHeatBudget=heatBudget, *particleHeatCoefficient=heatCoefficient;
 int coldWall2DEnabled=1;
 float cold2DH[1024]{}; GpuTime cold2DAge[128]{};
 float* pCold2DNodeSpecificEnthalpy=cold2DH;
 GpuTime* pCold2DRingContactAge=cold2DAge;
 GpuReal epsGPrev[16]{},V[16]{},rho[16]{},rhoUx[16]{},rhoUy[16]{},rhoUz[16]{};
 GpuReal Tgas[16]{},p[16]{},Ux[16]{},Uy[16]{},Uz[16]{};
 GpuReal Rgas=GPU_R(287),gammaGas=GPU_R(1.4);
 double rhoE[16]{},rhoENext[16]{};
 unsigned char *compactPStuck=pStuck; int *compactPStuckFaceId=pStuckFaceId;
 GpuReal* compactPT=pT;
 float *compactPColdNodeSpecificEnthalpy=coldH,*compactPCold2DNodeSpecificEnthalpy=cold2DH;
 double* compactPThermalEnthalpyResidual=pThermalEnthalpyResidual;
 int compactCount=0,*compactCountDevice=&compactCount,compactPStatus[16]{};
 int poolThermalCount[16]{},poissonPoolSampleTargetCount[16]{};
 GpuReal poolThermalSumUx[16]{},poolThermalSumUy[16]{},poolThermalSumUz[16]{},poolThermalSumU2[16]{};
 GpuReal poissonPoolMass[16]{},poissonPoolMomX[16]{},poissonPoolMomY[16]{},poissonPoolMomZ[16]{};
 GpuReal poissonPoolEnergy[16]{},poissonPoolDiameter[16]{},poissonPoolDiameter2[16]{};
 GpuReal particleDiameterMin=GPU_R(1e-8),particleDiameterMax=GPU_R(.01),particleDiameterSigma=0;
 unsigned long long pRng[16]{};
'''

INCLUDES = r'''
#define GPU_OPERATOR_THERMAL 1
#define GPU_PAIRED_PARTICLE_HEAT 1
#define GPU_POOL_INITIALIZATION_WITH_PROBABILITY 0
constexpr GpuReal OfSmall=GPU_R(2.22044604925031308085e-16);
constexpr GpuReal OfGreat=GPU_R(1.0)/OfSmall;
#include "GpuColdWall2DSolidification.H"
#include "GpuParticleThermalCarry.H"
#include "GpuPairedParticleHeat.H"
#include "GpuColdWallThermalCarry.cuh"
#include "GpuParticleCaloricDevice.cuh"
#include "actual_gas_recovery.cuh"
#define molecularGasConductivity actualMolecularGasConductivity
#include "route_conductivity.cuh"
#undef molecularGasConductivity
int thermalEvaluations=0;
template<class GasState> GpuReal molecularGasConductivity(const GasState& s){++thermalEvaluations;return actualMolecularGasConductivity(s);}
#include "GpuPairedParticleHeat.cuh"
#include "paired_heat_reference.cuh"
#include "operators/clearColdWallParticleState.cuh"
#include "actual_cold_2d_clear.cuh"
#include "actual_rng.cuh"
#include "operators/normalDevice.cuh"
#include "operators/appendSelectedStuckParticleIndex.cuh"
#include "GpuCollisionPoolSampling.cuh"
#include "actual_mobile_finalizer.cuh"
#include "GpuThermalParticleFinalization.cuh"
#include "operators/finalizeOneThermalizedParticlePath.cuh"
#include "operators/samplePoissonPoolParticlesKernel.cuh"
#include "operators/clearPoissonThermalPoolKernel.cuh"
#include "operators/correctOnePoissonThermalizedParticlePath.cuh"
'''

HARNESS = r'''
void clear_case() {
 DeviceState s;s.nCells=3;s.particleCapacity=16;
 for(int c=0;c<3;++c){s.heatCoefficient[c]=17.;s.heatBudget[c]=-19.;s.rhoENext[c]=23.;}
 for(int c=0;c<3;++c){threadIdx.x=c;clearPoissonThermalPoolKernel(&s);}
 for(int c=0;c<3;++c){
  near(s.heatCoefficient[c],0.,0.,"pool clear must initialize paired weighted capacity");
  near(s.heatBudget[c],0.,0.,"pool clear must initialize paired signed candidate heat");
  near(s.rhoENext[c],0.,0.,"pool clear must initialize paired capacity scratch");
 }
}
// Slots: selected mobile, selected 1D detach, unselected cold stuck,
// hot uncollided mobile, cold uncollided mobile, newly injected selected mobile,
// empty, dead, invalid-cell selected stuck, bad-pool selected stuck/mobile,
// retained selected stuck, selected 2D detach, and zero-mass live slot.
// Injection publication itself is outside this collision/thermal fixture;
// its resulting live slot is included in the actual particle-count scan.
void setup(DeviceState& s,bool nearFloor,bool noCollision=false,bool empty=false) {
 s.nCells=3;s.nFaces=16;s.particleCapacity=16;s.particleCount=empty?0:14;
 for(int c=0;c<s.nCells;++c){
  s.heatCoefficient[c]=17.;s.heatBudget[c]=-19.;s.rhoENext[c]=23.;
  s.V[c]=GPU_R(1.e-8);s.epsGPrev[c]=GPU_R(.9);s.rho[c]=GPU_R(1);
  s.rhoUx[c]=GPU_R(40000);s.Ux[c]=GPU_R(40000);s.Tgas[c]=GPU_R(4200);
  s.couplingTgasOld[c]=GPU_R(4200);s.couplingRhoOld[c]=GPU_R(1);
  s.couplingUxOld[c]=GPU_R(40000);s.thetaDragAlpha[c]=GPU_R(.6);
  s.rhoE[c]=pairedGasEnergyFloor(s,c)+(nearFloor?1.e-3:3.e6);
  threadIdx.x=c;clearPoissonThermalPoolKernel(&s);
 }
 for(int i=0;i<16;++i){
  s.pStatus[i]=1;s.pStuck[i]=particleWallMobile;s.pStuckFaceId[i]=-1;
  s.pCellId[i]=0;s.pm[i]=GPU_R(1.e-4);s.pd[i]=GPU_R(1.e-4);
  s.pT[i]=GPU_R(2800+i*20);s.pux[i]=GPU_R(39000+i*30);
  s.puy[i]=GPU_R(40-i);s.puz[i]=GPU_R(i);s.pRng[i]=1299709ULL+7919ULL*i;
  s.pThermalEnthalpyResidual[i]=.03125;
  s.particleStuckCandidateMask[i]=particleWallReboundContact;
  s.particleWallContactAreaScale[i]=GPU_R(1);s.magSf[i]=GPU_R(1);s.Sfx[i]=GPU_R(1);
  s.pDepositionArea[i]=1.e-8f;s.pContactDuration[i]=.01f;s.pContactMaximumArea[i]=2.2e-8f;
  s.pContactPeakFraction[i]=.4f;s.pContactAge[i]=.008;
 }
 for(int i:{0,1,5,8,9,10,11,12})s.pStatus[i]=noCollision?1:2;
 for(int i:{1,2,8,9,11,12}){s.pStuck[i]=particleWallDeposited;s.pStuckFaceId[i]=i;}
 for(int i:{1,2}){
  s.particleStuckCandidateMask[i]=particleWallSolidifyingDeposition;
  initialiseColdWallParticleState(s,i,s.pT[i]);
 }
 s.particleStuckCandidateMask[12]=particleWallColdWall2D;
 initialiseColdWall2DParticleState(s,12,s.pT[12]);
 // Two opposite-sign candidates make the sum sensitive to reduction order.
 s.pT[2]=GPU_R(4800);s.pm[2]=GPU_R(1.e-7);initialiseColdWallParticleState(s,2,s.pT[2]);
 s.pT[3]=GPU_R(5000);s.pm[3]=GPU_R(1.e-7);
 s.pStatus[6]=0;s.pStatus[7]=0;s.pm[7]=0;s.pm[13]=0;
 s.pCellId[8]=-1;s.pCellId[9]=1;s.pCellId[10]=1;
 // A valid, strongly bound selected parcel must remain in the deferred route.
 s.pDepositionArea[11]=1.f;
 // If collisions are absent, keep 2D out of this gas-only publication fixture.
 if(noCollision){s.pStuck[12]=particleWallMobile;s.pStuckFaceId[12]=-1;}
 if(empty)return;
 const int count=noCollision?0:5;
 s.poissonPoolSampleTargetCount[0]=count;
 GpuReal mass=0;for(int i:{0,1,5,11,12})mass+=s.pm[i];
 s.poissonPoolMass[0]=count?mass:GPU_R(0);
 s.poissonPoolMomX[0]=s.poissonPoolMass[0]*GPU_R(40000);
 s.poissonPoolEnergy[0]=s.poissonPoolMass[0]*(GPU_R(.5)*GPU_R(40000)*GPU_R(40000)+GPU_R(1.5e6));
}
void sample(DeviceState& s,bool noCollision) {
 blockDim.x=16;blockIdx.x=0;gridDim.x=1;
 for(int lane=0;lane<16;++lane){threadIdx.x=lane;samplePoissonPoolParticlesKernel(&s,1);}
 if(noCollision||!s.particleCount){require(s.compactCount==0,"no collisions append no selected-stuck indices");return;}
 require(s.pStatus[8]==0&&s.pStatus[9]==0&&s.pStatus[10]==0,"actual sampler invalid-cell and bad-pool guards kill before append");
 require(s.compactCount==3,"actual selected-stuck list contains three valid parcels only");
 for(int pos=0;pos<3;++pos)require(s.compactPStatus[pos]==(pos==0?1:pos==1?11:12),"actual selected-stuck list exact indices once");
}
void monitor(DeviceState& s){traceA=s.heatCoefficient;traceQ=s.heatBudget;traceC=s.rhoENext;traceCalls=0;thermalEvaluations=0;}
void stop_monitor(){traceA=traceQ=traceC=nullptr;}
void correct(DeviceState& s,double dt,bool fused,bool reversed) {
 for(int lane=0;lane<16;++lane){
  const int i=reversed?15-lane:lane;threadIdx.x=i;
  const bool deferred=i<s.particleCount&&s.pStuck[i]!=0&&s.pStatus[i]==2;
  monitor(s);
  if(fused)correctPoissonThermalizedMobileParticlesKernel<true>(&s,1,dt);
  else correctPoissonThermalizedMobileParticlesKernel<false>(&s,1);
  const int evaluations=thermalEvaluations;
  if(fused){
   const bool active=i<s.particleCount&&!deferred&&pairedParticleHeatCandidate(s,i,dt).capacity>0.;
   require(traceCalls==(active?3:0),"full scan accumulates exactly once, excluding selected stuck before correction",traceCalls,active?3:0);
   require(evaluations==(active?1:0),"full scan performs heat evaluation only in its unique partition");
  }else require(traceCalls==0&&evaluations==0,"reference correction with heat disabled performs zero thermal work");
  stop_monitor();
 }
 for(int lane=0;lane<16;++lane){
  const int pos=reversed?15-lane:lane;threadIdx.x=pos;
  monitor(s);
  if(fused)correctPoissonThermalizedStuckParticlesKernel<true>(&s,1,dt);
  else correctPoissonThermalizedStuckParticlesKernel<false>(&s,1);
  const int evaluations=thermalEvaluations;
  if(fused){
   const bool active=pos<s.compactCount&&pairedParticleHeatCandidate(s,s.compactPStatus[pos],dt).capacity>0.;
   require(traceCalls==(active?3:0),"selected-stuck correction accumulates exactly once after finalization",traceCalls,active?3:0);
   require(evaluations==(active?1:0),"selected-stuck heat is evaluated exactly once after finalization");
  }else require(traceCalls==0&&evaluations==0,"reference selected correction performs zero thermal work");
  stop_monitor();
 }
}
void compare_corrections(const DeviceState&a,const DeviceState&b){
 for(int i=0;i<a.particleCount;++i){
  require(a.pStatus[i]==b.pStatus[i]&&a.pStuck[i]==b.pStuck[i],"fused schedule preserves actual collision status");
  require(a.pux[i]==b.pux[i]&&a.puy[i]==b.puy[i]&&a.puz[i]==b.puz[i]&&a.pTheta[i]==b.pTheta[i],"fused schedule preserves corrected velocity and theta");
  require(a.pStuckFaceId[i]==b.pStuckFaceId[i]&&a.pContactAge[i]==b.pContactAge[i]&&a.pDepositionArea[i]==b.pDepositionArea[i],"fused schedule preserves mechanical contact metadata");
  require(a.pT[i]==b.pT[i]&&a.pThermalEnthalpyResidual[i]==b.pThermalEnthalpyResidual[i],"candidate accumulation does not publish particle heat");
 }
 require(std::equal(a.coldH,a.coldH+128,b.coldH)&&std::equal(a.cold2DH,a.cold2DH+1024,b.cold2DH),"fused collision preserves exact cold profile finalization");
}
long double particle_inventory(const DeviceState&s){
 long double result=0;for(int i=0;i<s.particleCount;++i)if(s.pStatus[i])
  result+=(long double)s.pm[i]*((long double)particleMaterialEnthalpyBaseDevice(s,i)+s.pThermalEnthalpyResidual[i]);
 return result;
}
void transaction(bool nearFloor,bool reversed,bool noCollision,bool empty,double dt){
 DeviceState fused,reference;setup(fused,nearFloor,noCollision,empty);setup(reference,nearFloor,noCollision,empty);
 sample(fused,noCollision);sample(reference,noCollision);
 PairedParticleHeatCandidate coldBefore{};
 if(!noCollision&&!empty)coldBefore=pairedParticleHeatCandidate(fused,1,dt);
 correct(fused,dt,true,reversed);correct(reference,dt,false,false);
 compare_corrections(fused,reference);
 if(!noCollision&&!empty){
  for(int i:{1,12}){
   require(fused.pStuck[i]==particleWallMobile&&fused.pStuckFaceId[i]==-1,"selected cold parcel really detached in actual finalizer");
   require(particleColdProfileNodeCountDevice(fused,i)==0&&!particleCaloricDevice(fused,i).phaseChange,"detachment switches to mobile caloric law before heat candidate");
   for(int j=0;j<8;++j)require(fused.coldH[i*8+j]==0.f,"detachment clears all 1D nodes");
   for(int j=0;j<64;++j)require(fused.cold2DH[i*64+j]==0.f,"detachment clears all 2D nodes");
  }
  require(fused.pStuck[11]!=particleWallMobile,"strongly bound selected parcel remains stuck");
  if(dt>0.)require(std::abs(pairedParticleHeatCandidate(fused,1,dt).weightedCapacity-coldBefore.weightedCapacity)>1.e-9,"selected cold pre/post correction candidate must be numerically distinguishable");
 }
 for(int c=0;c<reference.nCells;++c){threadIdx.x=c;clearPairedParticleHeatKernel(&reference);}
 for(int lane=0;lane<16;++lane){threadIdx.x=lane;reducePairedParticleHeatKernel(&reference,dt);}
 const double eps=std::numeric_limits<double>::epsilon();
 double totalAbsQ=0.,qTolerance=0.;
 for(int c=0;c<fused.nCells;++c){
  double absQ=0.,sumA=0.,sumC=0.;long double expectedA=0.,expectedQ=0.,expectedC=0.;int positive=0,negative=0;
  for(int i=0;i<fused.particleCount;++i){const auto v=pairedParticleHeatCandidate(fused,i,dt);if(v.capacity>0.&&fused.pCellId[i]==c){
   const double q=double(fused.pm[i])*v.specificHeat;absQ+=std::abs(q);sumA+=std::abs(v.weightedCapacity);sumC+=std::abs(v.capacity);positive+=q>0.;negative+=q<0.;
   expectedA+=v.weightedCapacity;expectedQ+=q;expectedC+=v.capacity;}}
  const double qTol=32.*eps*absQ;
  near(fused.heatCoefficient[c],double(expectedA),32.*eps*sumA,"actual fused coefficient equals independent sum of post-correction candidates");
  near(fused.heatBudget[c],double(expectedQ),qTol,"actual fused heat equals independent signed candidate sum");
  near(fused.rhoENext[c],double(expectedC),32.*eps*sumC,"actual fused scratch equals independent capacity sum");
  near(fused.heatCoefficient[c],reference.heatCoefficient[c],32.*eps*sumA,"fused weighted capacity equals unfused within double reduction bound");
  near(fused.heatBudget[c],reference.heatBudget[c],qTol,"fused signed heat equals unfused within absolute double reduction bound");
  near(fused.rhoENext[c],reference.rhoENext[c],32.*eps*sumC,"fused capacity equals unfused within double reduction bound");
  if(c==0&&!empty&&dt>0.)require(positive>0&&negative>0&&fused.heatBudget[c]>0.,"mixed signed proposals retain net positive gas withdrawal");
  if(c==2)require(fused.heatCoefficient[c]==0.&&fused.heatBudget[c]==0.&&fused.rhoENext[c]==0.,"empty cell reduction remains exactly zero");
  totalAbsQ+=absQ;qTolerance+=qTol;
 }
 const long double p0=particle_inventory(fused);double oldGas[3];for(int c=0;c<3;++c)oldGas[c]=fused.rhoE[c];
 for(int c=0;c<3;++c){threadIdx.x=c;acceptPairedParticleHeatKernel(&fused);acceptPairedParticleHeatKernel(&reference);
  near(fused.heatCoefficient[c],reference.heatCoefficient[c],128.*eps,"shared acceptance differs by at most double reduction propagation");}
 if(nearFloor&&!empty&&dt>0.)require(fused.heatCoefficient[0]>0.&&fused.heatCoefficient[0]<1.e-3,"actual gas floor limits accepted scale");
 double publicationBound=0.;
 for(int i=0;i<fused.particleCount;++i){
  const double base=particleMaterialEnthalpyBaseDevice(fused,i),carry=fused.pThermalEnthalpyResidual[i];
  publishPairedParticleGasHeat(fused,i,dt);publishPairedParticleGasHeat(reference,i,dt);
  const double db=particleMaterialEnthalpyBaseDevice(fused,i)-base;
  publicationBound+=32.*eps*double(fused.pm[i])*(std::abs(db)+std::abs(carry)+std::abs(fused.pThermalEnthalpyResidual[i]));
  // Compare the represented increment, not a relative tolerance against total H.
  const long double actual=(long double)particleMaterialEnthalpyBaseDevice(fused,i)+fused.pThermalEnthalpyResidual[i];
  const long double expected=(long double)particleMaterialEnthalpyBaseDevice(reference,i)+reference.pThermalEnthalpyResidual[i];
  const double heatTol=512.*eps*std::abs(pairedParticleHeatCandidate(reference,i,dt).specificHeat);
  require(std::abs(actual-expected)<=heatTol+4.*std::numeric_limits<long double>::epsilon()*std::abs(expected),"published fused enthalpy agrees within heat-sized arithmetic bound");
 }
 double accepted=0.,gasErrorBound=0.;
 for(int c=0;c<3;++c){
  near(fused.heatBudget[c],reference.heatBudget[c],256.*eps*totalAbsQ+qTolerance,"actual signed published heat matches unfused reference");
  accepted+=fused.heatBudget[c];threadIdx.x=c;settlePairedParticleHeatKernel(&fused);settlePairedParticleHeatKernel(&reference);
  const double ulp=std::nextafter(oldGas[c],INFINITY)-oldGas[c];
  near(fused.rhoE[c]-oldGas[c],reference.rhoE[c]-oldGas[c],2.*ulp,"gas energy increment agrees within two explicit double subtraction ULPs");
  require(fused.rhoE[c]>=pairedGasEnergyFloor(fused,c),"fused accepted gas heat respects EOS floor");
  gasErrorBound+=2.*ulp*pairedGasPhysicalVolume(fused,c);
 }
 const long double particleDelta=particle_inventory(fused)-p0;
 const double inventoryEvaluation=8.*double(std::numeric_limits<long double>::epsilon()*std::abs(p0));
 near(double(particleDelta),accepted,publicationBound+inventoryEvaluation,"actual particle authority change equals published heat budget");
 long double gasDelta=0.;for(int c=0;c<3;++c)gasDelta+=(long double)pairedGasPhysicalVolume(fused,c)*(fused.rhoE[c]-oldGas[c]);
 near(double(particleDelta+gasDelta),0.,publicationBound+inventoryEvaluation+gasErrorBound+64.*eps*std::abs(accepted),"fused full transaction conserves within local arithmetic bounds");
 std::printf("fusion dt=%.9g floor=%d reverse=%d no_collision=%d empty=%d accepted=%.17g closure=%.17Lg\n",dt,nearFloor,reversed,noCollision,empty,accepted,particleDelta+gasDelta);
}
void disabled_case(int nullBuffers,bool falsePositiveDt=false){
 DeviceState s,reference;setup(s,false);setup(reference,false);
 if(nullBuffers&1)s.particleHeatCoefficient=nullptr;
 if(nullBuffers&2)s.particleHeatBudget=nullptr;
 for(int c=0;c<3;++c){s.heatCoefficient[c]=17.;s.heatBudget[c]=-19.;s.rhoENext[c]=23.;}
 // No active heat buffers: the existing pool clear must not touch thermal scratch.
 if(nullBuffers)for(int c=0;c<3;++c){threadIdx.x=c;clearPoissonThermalPoolKernel(&s);
  require(s.heatCoefficient[c]==17.&&s.heatBudget[c]==-19.&&s.rhoENext[c]==23.,"heat-disabled pool clear must preserve thermal scratch");}
 // Rebuild real collision pools after testing their clear.
 if(nullBuffers){setup(s,false);if(nullBuffers&1)s.particleHeatCoefficient=nullptr;if(nullBuffers&2)s.particleHeatBudget=nullptr;}
 sample(s,false);sample(reference,false);
 for(int c=0;c<3;++c){s.heatCoefficient[c]=17.;s.heatBudget[c]=-19.;s.rhoENext[c]=23.;}
 // A positive dt with either missing buffer must also bypass candidate work.
 if(nullBuffers){
  for(int lane=0;lane<16;++lane){threadIdx.x=lane;monitor(s);correctPoissonThermalizedMobileParticlesKernel<true>(&s,1,.001);
   require(traceCalls==0&&thermalEvaluations==0,"missing heat buffer bypasses mobile thermal work even at positive dt");stop_monitor();}
  for(int lane=0;lane<16;++lane){threadIdx.x=lane;monitor(s);correctPoissonThermalizedStuckParticlesKernel<true>(&s,1,.001);
   require(traceCalls==0&&thermalEvaluations==0,"missing heat buffer bypasses selected-stuck thermal work even at positive dt");stop_monitor();}
 }else if(falsePositiveDt){
  for(int lane=0;lane<16;++lane){threadIdx.x=lane;monitor(s);correctPoissonThermalizedMobileParticlesKernel<false>(&s,1,.001);
   require(traceCalls==0&&thermalEvaluations==0,"false specialization bypasses mobile thermal math with positive dt and live buffers");stop_monitor();}
  for(int lane=0;lane<16;++lane){threadIdx.x=lane;monitor(s);correctPoissonThermalizedStuckParticlesKernel<false>(&s,1,.001);
   require(traceCalls==0&&thermalEvaluations==0,"false specialization bypasses selected-stuck thermal math with positive dt and live buffers");stop_monitor();}
 }else correct(s,0.,false,false);
 correct(reference,0.,false,false);compare_corrections(s,reference);
 for(int c=0;c<3;++c)require(s.heatCoefficient[c]==17.&&s.heatBudget[c]==-19.&&s.rhoENext[c]==23.,"heat-disabled correction does not clear or accumulate thermal scratch");
}
int main(int argc,char**argv){
 const char* mode=argc>1?argv[1]:"clear";
 if(!std::strcmp(mode,"clear"))clear_case();
 else if(!std::strcmp(mode,"mixed"))transaction(false,false,false,false,1.e-8);
 else if(!std::strcmp(mode,"reversed"))transaction(false,true,false,false,.001);
 else if(!std::strcmp(mode,"floor"))transaction(true,true,false,false,.001);
 else if(!std::strcmp(mode,"zero"))transaction(false,false,false,false,0.);
 else if(!std::strcmp(mode,"no_collision"))transaction(false,true,true,false,.001);
 else if(!std::strcmp(mode,"empty"))transaction(false,false,false,true,.001);
 else if(!std::strcmp(mode,"disabled"))disabled_case(0);
 else if(!std::strcmp(mode,"disabled_null"))disabled_case(3);
 else if(!std::strcmp(mode,"disabled_no_coefficient"))disabled_case(1);
 else if(!std::strcmp(mode,"disabled_no_budget"))disabled_case(2);
 else if(!std::strcmp(mode,"disabled_false_positive_dt"))disabled_case(0,true);
 else return 2;
}
'''


@pytest.fixture(scope='module')
def fusion_binary( tmp_path_factory):
    build = tmp_path_factory.mktemp(f'paired_heat_fusion_64')
    fixture = ((VALIDATION_ROOT / 'tests/fixtures/cold_wall_contact_routes.cpp')).read_text()
    prefix = fixture[:fixture.index('#include "operators/particleSpecificHeatDevice.cuh"')]
    prefix = prefix.replace('[1]', '[16]').replace('coldH[8]', 'coldH[128]').replace('coldRing[8]', 'coldRing[128]')
    prefix = prefix.replace('struct DeviceState {', 'struct DeviceState {' + EXTRA_STATE)
    prefix = prefix.replace('template<class T> T atomicAdd(T* target,T value) {T old=*target;*target+=value;return old;}',
        'double *traceA=nullptr,*traceQ=nullptr,*traceC=nullptr; int traceCalls=0;\n'
        'template<class T> T atomicAdd(T* target,T value) {\n'
        'if constexpr(std::is_same_v<T,double>){for(int c=0;c<3;++c)if((traceA&&target==traceA+c)||(traceQ&&target==traceQ+c)||(traceC&&target==traceC+c))++traceCalls;}\n'
        'T old=*target;*target+=value;return old;}')
    helpers = fixture[fixture.index('void require('):fixture.index('template<class F> int launch8')]
    prefix = prefix.replace('struct DeviceState {', '#include "Diagnostics.H"\nstruct DeviceState {\n ugkp::diagnostics::DeviceContext* diagnostics=nullptr;\n')
    recovery = (ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    (build/'actual_gas_recovery.cuh').write_text(
        '#include "gasTransport/GasCapabilities.H"\n#include "gasTransport/MixtureThermo.H"\n#include "gasTransport/GasGeometryValidation.H"\n'
        + '#include "operators/gasStencilDiagnostics.cuh"\n'
        + recovery[recovery.index('template<class GasState,class Scalar>'):recovery.index('template<class GasState>\n__device__ void sstInternalRhoDiffusivities')]
        + function(recovery, 'template<class GasState>\n__device__ void recoverGasPrimitiveCell'))
    conductivity = (ROOT/'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh').read_text()
    (build/'route_conductivity.cuh').write_text('#include "gasTransport/GasStateView.H"\n'
        + function(conductivity, 'template<class GasState>\n__device__ GPU_OPERATOR_REAL molecularGasConductivity'))
    two = (ROOT/'common/wall/GpuColdWall2DDevice.cuh').read_text()
    (build/'actual_cold_2d_clear.cuh').write_text(function(two, '__device__ void clearColdWall2DParticleState')
        + function(two, '__device__ void initialiseColdWall2DParticleState'))
    backend = (ROOT/'applications/CHT/gpu/GpuResidentStrict.cu').read_text()
    (build/'actual_mobile_finalizer.cuh').write_text(function(backend, '__device__ void finalizeOneThermalizedMobileParticle'))
    seed = (ROOT/'common/operators/snapshotParticleGasCouplingStateKernel.cuh').read_text()
    (build/'actual_rng.cuh').write_text(function(seed, '__device__ unsigned long long mixSeed')
        + function(backend, '__device__ GpuReal uniform01Device'))
    source = build/'fusion.cpp'
    source.write_text(prefix + INCLUDES + helpers + HARNESS)
    exe = build/'fusion'
    run = subprocess.run(['g++','-std=c++20','-O2','-pthread','-DROUTE_FSH=0','-I'+str(build),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/wall'),
        '-I'+str((VALIDATION_ROOT / 'tests/gas_transport')),str(source),'-o',str(exe)],capture_output=True,text=True,timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    return exe


@pytest.mark.parametrize('case',['clear','mixed','reversed','floor','zero','no_collision','empty','disabled','disabled_null','disabled_no_coefficient','disabled_no_budget','disabled_false_positive_dt'])
def test_actual_sampling_correction_heat_fusion(fusion_binary,case):
    run = subprocess.run([str(fusion_binary), case],capture_output=True,text=True,timeout=30)
    assert run.returncode == 0,run.stdout+run.stderr
