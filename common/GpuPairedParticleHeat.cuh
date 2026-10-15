#pragma once
// CHT heat transaction. Coefficients and candidates share the post-collision
// epoch; no particle field changes between this reduction and its relaxation.
struct PairedParticleHeatCandidate {double capacity, weightedCapacity, specificHeat;};
__device__ __forceinline__ PairedParticleHeatCandidate pairedParticleHeatCandidate
(const DeviceState& s,int i,GpuTime dt)
{
 if(!(dt>0.)||s.pStatus[i]==0||!(s.pm[i]>0))return {0.,0.,0.};
 const int c=s.pCellId[i];if(c<0||c>=s.nCells)return {0.,0.,0.};
 const auto law=particleCaloricDevice(s,i);
 const double base=particleMaterialEnthalpyBaseDevice(s,i),carry=s.pThermalEnthalpyResidual[i];
 const double oldH=base+carry,Ti=law.temperature(oldH);
 double cp=law.specificHeat(Ti);
 if(!(cp>0.)||!ugkwp::finiteThermalCarry(oldH)||!ugkwp::finiteThermalCarry(Ti))return {0.,0.,0.};
 const bool stuck=s.pStuck[i]!=Foam::gpuThermal::particleWallMobile;
 const double ux=stuck?0.:double(s.pux[i]),uy=stuck?0.:double(s.puy[i]),uz=stuck?0.:double(s.puz[i]);
 const double dx=double(s.couplingUxOld[c])-ux,dy=double(s.couplingUyOld[c])-uy,dz=double(s.couplingUzOld[c])-uz;
 const double d=double(clampMin(finiteOr(s.pd[i],s.particleDiameterFallback),GPU_R(1e-12)));
 const double re=double(clampMin(s.couplingRhoOld[c],s.rhoMin))*d*::sqrt(dx*dx+dy*dy+dz*dz)/double(clampMin(s.gasMu,GPU_R(1e-30)));
 const double nu=2.+.6*::sqrt(re>0.?re:0.)*double(s.gasPrOneThird);
 const double conductancePerMass=6.*nu*double(molecularGasConductivity(s))/(double(s.rhoSolid)*d*d);
 const double Tg=double(clampMin(s.couplingTgasOld[c],s.TgasMin));
 const int count=particleColdProfileNodeCountDevice(s,i);
 const double equilibriumIncrement=count==8 ? law.enthalpyIncrement(Ti,Tg-Ti) : 0.;
 if(count==8 && Tg!=Ti){
  const double secant=equilibriumIncrement/(Tg-Ti);
  if(secant>0.)cp=secant;
 }
 const double rate=conductancePerMass/cp;
 const double a=-::expm1(-(rate>0.?rate:0.)*double(dt));
 if(count==64){
  // The2D model applies gas power to its top surface during explicit
  // conduction substeps. Reserve a positive upper budget; actual signed work
  // is recorded by that existing solver, not replaced by a uniform source.
  // Existing phase-changing substeps guarantee positive H/T but not a
  // local temperature maximum principle across a sharp Cp transition.
  const double storageEpsilon=0x1p-52;
  const double roundoffGuard=1.+64.*storageEpsilon;
  const double upper=conductancePerMass*double(dt)*(Tg>0.?Tg:0.)*roundoffGuard;
  return {double(s.pm[i])*cp,double(s.pm[i])*cp*a,upper};
 }
 double deltaT=a*(Tg-Ti);
 const double lowerT=double(s.TpMin)-Ti;
 const double upperT=double(s.TpMax)-Ti;
 const double lower=lowerT<0.?lowerT:0.;
 const double upper=upperT>0.?upperT:0.;
 if(deltaT<lower)deltaT=lower;
 if(deltaT>upper)deltaT=upper;
 // Evaluate the small component last, so residual heat is consumed rather than
 // being rounded away in a large H+carry subtraction.
 double dh=law.enthalpyIncrement(Ti,deltaT);
 // Secant weights make Q = A*(Tg-weighted Ti) exactly in the unclipped
 // interval, preventing nonlinear-Cp gas-temperature overshoot.
 double effectiveA=a;
 if(Tg!=Ti)effectiveA=deltaT/(Tg-Ti);
 if(deltaT!=0.)cp=dh/deltaT;
 if(count==8){
  dh=equilibriumIncrement*a;
  effectiveA=a;
  if(Tg!=Ti)cp=equilibriumIncrement/(Tg-Ti);
  const double lower=ugkwp::coldWallUniformCoolingBound
      (s.pColdNodeSpecificEnthalpy+i*8,8,carry);
  if(dh<lower){dh=lower;if(equilibriumIncrement!=0.)effectiveA=dh/equilibriumIncrement;}
 }
 const double capacity=double(s.pm[i])*cp;
 return {capacity,capacity*effectiveA,dh};
}
// Called from the existing post-RK collision-pool clear, before any candidate.
__device__ __forceinline__ void clearPairedParticleHeatCell(DeviceState& s,int c)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
 s.particleHeatCoefficient[c]=0.;s.particleHeatBudget[c]=0.;s.rhoENext[c]=0.;
}
// Each receiver reaches this helper exactly once after its final collision
// correction. The selected-stuck partition is owned by the correction kernels.
__device__ __forceinline__ void accumulatePairedParticleHeatCandidate
(DeviceState& s,int i,GpuTime dt)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
 const auto p=pairedParticleHeatCandidate(s,i,dt);if(!(p.capacity>0.))return;
 const int c=s.pCellId[i];atomicAdd(s.particleHeatCoefficient+c,p.weightedCapacity);
 atomicAdd(s.particleHeatBudget+c,double(s.pm[i])*p.specificHeat);
 atomicAdd(s.rhoENext+c,p.capacity);
}
__device__ __forceinline__ double pairedGasPhysicalVolume(const DeviceState&s,int c)
{return double(clampMin(s.epsGPrev[c],OfSmall))*double(s.V[c]);}
__device__ __forceinline__ double pairedGasEnergyFloor(const DeviceState&s,int c)
{
 const GpuReal rhoSafe=clampMin(finiteOr(s.rho[c],s.rhoMin),s.rhoMin);
 const bool repairDensity=!finiteDevice(s.rho[c]) || s.rho[c]<s.rhoMin;
 // Match the actual conserved candidates and arithmetic order used by EOS
 // recovery, rather than the algebraically equivalent momentum-square form.
 const GpuReal mx=repairDensity||!finiteDevice(s.rhoUx[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUx[c]/rhoSafe,GPU_R(0)):s.rhoUx[c];
 const GpuReal my=repairDensity||!finiteDevice(s.rhoUy[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUy[c]/rhoSafe,GPU_R(0)):s.rhoUy[c];
 const GpuReal mz=repairDensity||!finiteDevice(s.rhoUz[c]/rhoSafe)?rhoSafe*finiteOr(s.rhoUz[c]/rhoSafe,GPU_R(0)):s.rhoUz[c];
 const double rho=double(rhoSafe);
 const double kinetic=ugkwp::gasLegacyKineticEnergy(rho,
  ugkwp::gasEnergyFiniteOr(double(mx)/rho,0.),ugkwp::gasEnergyFiniteOr(double(my)/rho,0.),ugkwp::gasEnergyFiniteOr(double(mz)/rho,0.));
 return kinetic+ugkwp::gasLegacyMinimumInternalEnergy(double(s.rhoMin),rho,
  double(s.Rgas),double(s.TgasMin),double(s.gammaGas),double(OfSmall));
}
__global__ void acceptPairedParticleHeatKernel(DeviceState*sp)
{
 DeviceState&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
 const double C=s.rhoENext[c],A=s.particleHeatCoefficient[c],Q=s.particleHeatBudget[c];
 const double volume=pairedGasPhysicalVolume(s,c);
 const double Cg=volume*double(clampMin(finiteOr(s.rho[c],s.rhoMin),s.rhoMin))*double(s.Rgas)
  /ugkwp::gasLegacyCaloricDenominator(double(s.gammaGas),double(OfSmall));
 double scale=C>0.?ugkwp::pairedHeatBathFraction(A/C,C,Cg):0.;
 const double available=(s.rhoE[c]-pairedGasEnergyFloor(s,c))*volume;
 scale*=ugkwp::pairedHeatAcceptanceScale(scale*Q,available);
 s.particleHeatCoefficient[c]=scale;s.particleHeatBudget[c]=0.;
}
__device__ __forceinline__ void publishPairedParticleGasHeat(DeviceState&s,int i,GpuTime dt)
{
 if(s.particleHeatCoefficient==nullptr||s.particleHeatBudget==nullptr)return;
 if(particleColdProfileNodeCountDevice(s,i)==64)return;
 const auto p=pairedParticleHeatCandidate(s,i,dt);if(!(p.capacity>0.))return;
 const int c=s.pCellId[i];const double dh=s.particleHeatCoefficient[c]*p.specificHeat;
 if(dh==0.)return;
 double accepted=0.;const int count=particleColdProfileNodeCountDevice(s,i);
 if(count){
  float* nodes=(count==8?s.pColdNodeSpecificEnthalpy:s.pCold2DNodeSpecificEnthalpy)+i*count;
  const auto result=ugkwp::publishColdWallUniformSpecificHeat(nodes,count,s.pThermalEnthalpyResidual[i],dh);
  if(!result.valid)return;
  accepted=result.acceptedSpecificHeat;
  s.pT[i]=GpuReal(particleEffectiveTemperatureDevice(s,i));
 }else{
  ugkwp::ParticleThermalCarry<GpuReal> state{s.pT[i],s.pThermalEnthalpyResidual[i]};
  const auto result=ugkwp::publishParticleThermalHeat(state,dh,s.TpMin,s.TpMax,particleCaloricDevice(s,i));
  if(!result.valid)return;
  s.pT[i]=state.temperature;s.pThermalEnthalpyResidual[i]=state.specificEnthalpyResidual;
  accepted=result.acceptedSpecificHeat;
 }
 atomicAdd(s.particleHeatBudget+c,double(s.pm[i])*accepted);
}
__global__ void settlePairedParticleHeatKernel(DeviceState*sp)
{
 DeviceState&s=*sp;int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
 const double q=s.particleHeatBudget[c];if(q==0.)return;
 // Actual represented particle change, with the unavoidable final double
 // subtraction roundoff left visible in the physical budget tests.
 s.rhoE[c]-=q/pairedGasPhysicalVolume(s,c);
 recoverGasPrimitiveCell(s,c);
}
