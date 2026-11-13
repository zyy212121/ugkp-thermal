#pragma once
#include "gasTransport/ParticleHeatReservoir.H"
// Contact/profile heat transaction. Membership is captured with the macro
// source; coefficients use the corrected post-collision state, including a
// parcel that has detached since it was excluded from the mobile source.
__device__ __forceinline__ bool pairedParticleHeatMember(const DeviceState& s,int i)
{
 if(!s.particleStuckModelConfigured)return false;
 return s.pStuck[i]!=Foam::gpuThermal::particleWallMobile || s.particleContactHeatEpoch[i];
}
struct PairedParticleHeatCandidate {float capacity, weightedCapacity, specificHeat;};
__device__ __forceinline__ PairedParticleHeatCandidate pairedParticleHeatCandidate
(const DeviceState& s,int i,GpuTime dt)
{
 if(!(dt>0.f)||s.pStatus[i]==0||!(s.pm[i]>0)||!pairedParticleHeatMember(s,i))return {0.f,0.f,0.f};
 const int c=s.pCellId[i];if(c<0||c>=s.nCells)return {0.f,0.f,0.f};
 const auto law=particleCaloricDevice(s,i);
 const float base=particleMaterialEnthalpyBaseDevice(s,i);
 const float oldH=base,Ti=particleEffectiveTemperatureDevice(s,i);
 float cp=law.specificHeat(Ti);
 if(!(cp>0.f)||!ugkwp::finiteThermalCarry(oldH)||!ugkwp::finiteThermalCarry(Ti))return {0.f,0.f,0.f};
 const bool stuck=s.pStuck[i]!=Foam::gpuThermal::particleWallMobile;
 const float ux=stuck?0.f:float(s.pux[i]),uy=stuck?0.f:float(s.puy[i]),uz=stuck?0.f:float(s.puz[i]);
 const float dx=float(s.couplingUxOld[c])-ux,dy=float(s.couplingUyOld[c])-uy,dz=float(s.couplingUzOld[c])-uz;
 const float d=float(clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),GPU_R(1e-12f)));
 const float re=float(clampMin(s.couplingRhoOld[c],s.rhoMin))*d*std::sqrt(dx*dx+dy*dy+dz*dz)/float(clampMin(ugkwp::gasCellMolecularViscosityAt(s,c,s.couplingTgasOld[c]),GPU_R(1e-30f)));
 const float nu=2.f+.6f*std::sqrt(re>0.f?re:0.f)*float(ugkwp::gasCellPrandtlOneThird(s,c));
 const float conductancePerMass=6.f*nu*float(molecularGasConductivity(s,c,s.couplingTgasOld[c]))/(float(s.rhoSolid)*d*d);
 const float Tg=float(clampMin(s.couplingTgasOld[c],s.TgasMin));
 const int count=particleColdProfileNodeCountDevice(s,i);
 const float equilibriumIncrement=count==8 ? law.enthalpyIncrement(Ti,Tg-Ti) : 0.f;
 if(count==8 && Tg!=Ti){
  const float secant=equilibriumIncrement/(Tg-Ti);
  if(secant>0.f)cp=secant;
 }
 const float rate=conductancePerMass/cp;
 const float a=-std::expm1(-(rate>0.f?rate:0.f)*float(dt));
 if(count==64){
  // The2D model applies gas power to its top surface during explicit
  // conduction substeps. Reserve a positive upper budget; actual signed work
  // is recorded by that existing solver, not replaced by a uniform source.
  // Existing phase-changing substeps guarantee positive H/T but not a
  // local temperature maximum principle across a sharp Cp transition.
  const float upper=conductancePerMass*float(dt)*(Tg>0.f?Tg:0.f);
  return {float(s.pm[i])*cp,float(s.pm[i])*cp*a,upper};
 }
 float deltaT=a*(Tg-Ti);
 const float lowerT=float(s.TpMin)-Ti;
 const float upperT=float(s.TpMax)-Ti;
 const float lower=lowerT<0.f?lowerT:0.f;
 const float upper=upperT>0.f?upperT:0.f;
 if(deltaT<lower)deltaT=lower;
 if(deltaT>upper)deltaT=upper;
 // Evaluate the caloric heat increment directly in binary32.
 float dh=law.enthalpyIncrement(Ti,deltaT);
 // Secant weights make Q = A*(Tg-weighted Ti) exactly in the unclipped
 // interval, preventing nonlinear-Cp gas-temperature overshoot.
 float effectiveA=a;
 if(Tg!=Ti)effectiveA=deltaT/(Tg-Ti);
 if(deltaT!=0.f)cp=dh/deltaT;
 if(count==8){
  dh=equilibriumIncrement*a;
  effectiveA=a;
  if(Tg!=Ti)cp=equilibriumIncrement/(Tg-Ti);
  const float lower=ugkwp::coldWallUniformCoolingBound
      (s.pColdNodeSpecificEnthalpy+i*8,8);
  if(dh<lower){dh=lower;if(equilibriumIncrement!=0.f)effectiveA=dh/equilibriumIncrement;}
 }
 const float capacity=float(s.pm[i])*cp;
 return {capacity,capacity*effectiveA,dh};
}
// Called from the existing post-RK collision-pool clear, before any candidate.
__device__ __forceinline__ void clearPairedParticleHeatCell(DeviceState& s,int c)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
 s.particleHeatCoefficient[c]=0.f;s.particleHeatBudget[c]=0.f;s.rhoENext[c]=0.f;
}
// Each receiver reaches this helper exactly once after its final collision
// correction. The selected-stuck partition is owned by the correction kernels.
__device__ __forceinline__ void accumulatePairedParticleHeatCandidate
(DeviceState& s,int i,GpuTime dt)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
#if defined(GPU_MIXTURE_PARTICLE_COUPLING)
 // Mixture free-particle coupling runs after collisions and owns the current
 // mobile set. It does not use the earlier single-gas macro partition.
 if(ugkwp::mixtureGasActive(s))
  s.particleContactHeatEpoch[i]=s.pStuck[i]!=Foam::gpuThermal::particleWallMobile;
 else
#endif
  s.particleContactHeatEpoch[i]=pairedParticleHeatMember(s,i);
 const auto p=pairedParticleHeatCandidate(s,i,dt);
 s.particleContactHeatCandidate[i]=p.capacity>0.f?p.specificHeat:0.f;
 if(!(p.capacity>0.f))return;
 const int c=s.pCellId[i];atomicAdd(s.particleHeatCoefficient+c,p.weightedCapacity);
 atomicAdd(s.particleHeatBudget+c,float(s.pm[i])*p.specificHeat);
 atomicAdd(s.rhoENext+c,p.capacity);
}
__device__ __forceinline__ float pairedGasPhysicalVolume(const DeviceState&s,int c)
{return float(clampMin(s.epsGPrev[c],OfSmall))*float(s.V[c]);}
__device__ __forceinline__ float pairedGasEnergyFloor(const DeviceState&s,int c)
{
 const GpuReal rhoSafe=clampMin(finiteOr(s.rho[c],s.rhoMin),s.rhoMin);
 const bool repairDensity=!finiteDevice(s.rho[c]) || s.rho[c]<s.rhoMin;
 // Match the actual conserved candidates and arithmetic order used by EOS
 // recovery, rather than the algebraically equivalent momentum-square form.
 const GpuReal mx=repairDensity||!finiteDevice(s.rhoUx[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUx[c]/rhoSafe,GPU_R(0)):s.rhoUx[c];
 const GpuReal my=repairDensity||!finiteDevice(s.rhoUy[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUy[c]/rhoSafe,GPU_R(0)):s.rhoUy[c];
 const GpuReal mz=repairDensity||!finiteDevice(s.rhoUz[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUz[c]/rhoSafe,GPU_R(0)):s.rhoUz[c];
 const float rho=float(rhoSafe);
 const float kinetic=ugkwp::gasLegacyKineticEnergy(rho,
  ugkwp::gasEnergyFiniteOr(float(mx)/rho,0.f),ugkwp::gasEnergyFiniteOr(float(my)/rho,0.f),ugkwp::gasEnergyFiniteOr(float(mz)/rho,0.f));
 const float legacy=ugkwp::gasLegacyMinimumInternalEnergy(float(s.rhoMin),rho,
  float(s.Rgas),float(s.TgasMin),float(s.gammaGas),float(OfSmall));
 return kinetic+ugkwp::particleGasMinimumInternalEnergy(s,c,legacy);
}
__global__ void acceptPairedParticleHeatKernel(DeviceState*sp)
{
 DeviceState&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
 const float C=s.rhoENext[c],A=s.particleHeatCoefficient[c],Q=s.particleHeatBudget[c];
 const float volume=pairedGasPhysicalVolume(s,c);
 float Cg=volume*float(clampMin(finiteOr(s.rho[c],s.rhoMin),s.rhoMin))*float(s.Rgas)
  /ugkwp::gasLegacyCaloricDenominator(float(s.gammaGas),float(OfSmall));
 // Retain the existing finite-bath exponential and candidate/settlement
 // organization. Its gas capacity is the actual local caloric derivative.
 Cg=ugkwp::particleGasHeatCapacity(s,c,volume,Cg);
 float scale=C>0.f?ugkwp::pairedHeatBathFraction(A/C,C,Cg):0.f;
 const float available=(s.rhoE[c]-pairedGasEnergyFloor(s,c))*volume;
 scale*=ugkwp::pairedHeatAcceptanceScale(scale*Q,available);
 s.particleHeatCoefficient[c]=scale;s.particleHeatBudget[c]=0.f;
}
__device__ __forceinline__ void publishPairedParticleGasHeat(DeviceState&s,int i,GpuTime dt)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
 if(!pairedParticleHeatMember(s,i) || particleColdProfileNodeCountDevice(s,i)==64)return;
 const int c=s.pCellId[i];const float dh=s.particleHeatCoefficient[c]*s.particleContactHeatCandidate[i];
 if(dh==0.f)return;
 float accepted=0.f;const int count=particleColdProfileNodeCountDevice(s,i);
 if(count){
  float* nodes=(count==8?s.pColdNodeSpecificEnthalpy:s.pCold2DNodeSpecificEnthalpy)+i*count;
  const auto result=ugkwp::publishColdWallUniformSpecificHeat(nodes,count,dh);
  if(!result.valid)return;
  accepted=result.acceptedSpecificHeat;
  s.pT[i]=GpuReal(particleEffectiveTemperatureDevice(s,i));
 }else{
  ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i]};
  const auto result=ugkwp::publishParticleThermalHeat(state,dh,s.TpMin,s.TpMax,particleCaloricDevice(s,i));
  if(!result.valid)return;
  s.pT[i]=state.temperature;
  accepted=result.acceptedSpecificHeat;
 }
 atomicAdd(s.particleHeatBudget+c,float(s.pm[i])*accepted);
}
__global__ void settlePairedParticleHeatKernel(DeviceState*sp)
{
 DeviceState&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
 const float q=s.particleHeatBudget[c];if(q==0.f)return;
 // Settle the represented exchange directly; no rounding carry is retained.
 s.rhoE[c]-=q/pairedGasPhysicalVolume(s,c);
 recoverGasPrimitiveCell(s,c);
}
