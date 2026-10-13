"""Frozen test-only algorithms from d6110c02745a8ad726208a426c4bb3a28233c1fd.

References deliberately retain the duplicate arithmetic. They are numerical and
work-count oracles, never imported by production code.
"""
FINITE_REFERENCE = r'''
template<class Storage, class Caloric>
 inline ParticleWallThermalPublication referenceFiniteHeat
(
    ParticleThermalCarry<Storage>& state,
    const double wallTemperature,
    const double physicalMass,
    const double parcelMass,
    const double conductanceIntegral,
    const Storage minimumTemperature,
    const Storage maximumTemperature,
    const Caloric& caloric
)
{
    const double oldEnthalpy = effectiveParticleEnthalpy(state, caloric);
    const double oldTemperature = caloric.temperature(oldEnthalpy);
    if (!finiteThermalCarry(oldEnthalpy) || !finiteThermalCarry(oldTemperature)
        || !finiteThermalCarry(wallTemperature) || !finiteThermalCarry(physicalMass)
        || !finiteThermalCarry(parcelMass) || !finiteThermalCarry(conductanceIntegral)
        || !(oldTemperature > 0.0) || !(wallTemperature > 0.0)
        || !(physicalMass > 0.0) || !(parcelMass > 0.0) || conductanceIntegral < 0.0)
        return {false, 0.0};
    if (conductanceIntegral == 0.0 || wallTemperature == oldTemperature)
    {
        const auto publication = publishParticleThermalHeat
            (state, 0.0, minimumTemperature, maximumTemperature, caloric);
        return {publication.valid, 0.0};
    }
    double lower = oldTemperature < wallTemperature ? oldTemperature : wallTemperature;
    double upper = oldTemperature > wallTemperature ? oldTemperature : wallTemperature;
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        const double middle = lower + 0.5*(upper - lower);
        const double residual = physicalMass*(caloric.enthalpy(middle) - oldEnthalpy)
            + conductanceIntegral*(middle - wallTemperature);
        if (!finiteThermalCarry(residual)) return {false, 0.0};
        if (residual < 0.0) lower = middle;
        else upper = middle;
    }
    const double solvedTemperature = lower + 0.5*(upper - lower);
    // The small flux stays small; subtracting two large absolute h values here
    // would discard heat that the persistent carry is meant to retain.
    const double proposedSpecificHeat =
        conductanceIntegral*(wallTemperature - solvedTemperature)/physicalMass;
    auto candidate = state;
    const auto publication = publishParticleThermalHeat
        (candidate, proposedSpecificHeat, minimumTemperature, maximumTemperature, caloric);
    const double wallEnergy = -parcelMass*publication.acceptedSpecificHeat;
    if (!publication.valid || !finiteThermalCarry(wallEnergy)) return {false, 0.0};
    state = candidate;
    return {true, wallEnergy};
}
'''

COLD_REFERENCE = r'''
 inline ColdWallHeatPublication referenceColdHeat
(float* nodes, int count, double& residual, double specificHeat) {
 if (!nodes || count<=0 || !(::fabs(residual)<=DBL_MAX)
     || !(::fabs(specificHeat)<=DBL_MAX)) return {false,0.0};
 for(int j=0;j<count;++j) {
  const double effective=double(nodes[j])+residual;
  if(!(nodes[j]>0.0f) || !(effective>0.0) || !(effective<=FLT_MAX)) return {false,0.0};
 }
 if (specificHeat==0.0) return {true,0.0};
 const double correction=residual+specificHeat;
 double storageChange=0.0,minimumStored=DBL_MAX,maximumStored=0.;
 for(int j=0;j<count;++j) {
  const double target=double(nodes[j])+correction;
  const float stored=coldWallStoredEnthalpyFloor(target);
  if(!(target>0.0) || !(target<=FLT_MAX) || !(stored>0.0f)) return {false,0.0};
  storageChange+=(double(stored)-double(nodes[j]))/count;
  if(double(stored)<minimumStored)minimumStored=stored;
  if(double(stored)>maximumStored)maximumStored=stored;
 }
 const double next=(residual-storageChange)+specificHeat;
 const double accepted=storageChange+(next-residual);
 // Validate the actual shared-mean representation BEFORE any component write.
 if(!(::fabs(next)<=DBL_MAX) || !(minimumStored+next>0.)
    || !(maximumStored+next<=FLT_MAX))return {false,0.};
 for(int j=0;j<count;++j)nodes[j]=coldWallStoredEnthalpyFloor(double(nodes[j])+correction);
 residual=next;
 return {true,accepted};
}
'''

PAIRED_REFERENCE = r'''
 inline PairedParticleHeatCandidate referenceCandidate
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
 const double equilibriumIncrement=law.enthalpyIncrement(Ti,Tg-Ti);
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
  const double storageEpsilon=sizeof(GpuReal)==sizeof(float)?0x1p-23:0x1p-52;
  const double roundoffGuard=1.+64.*storageEpsilon;
  const double upper=conductancePerMass*double(dt)*(Tg>0.?Tg:0.)*roundoffGuard;
  return {double(s.pm[i])*cp,double(s.pm[i])*cp*a,upper};
 }
 double deltaT=a*(Tg-Ti);
 if(deltaT<double(s.TpMin)-Ti)deltaT=double(s.TpMin)-Ti;
 if(deltaT>double(s.TpMax)-Ti)deltaT=double(s.TpMax)-Ti;
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
'''
