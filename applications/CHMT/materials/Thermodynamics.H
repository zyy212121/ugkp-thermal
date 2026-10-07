#ifndef CHMT_MATERIALS_THERMODYNAMICS_H
#define CHMT_MATERIALS_THERMODYNAMICS_H
#include "core/Configuration.H"
#include <string>
namespace chmt {
CHMT_HD inline Real speciesE(const SpeciesThermo& s,Real T) { return s.e0+(s.cp0-s.R)*T+0.5*s.cp1*T*T; }
CHMT_HD inline Real speciesH(const SpeciesThermo& s,Real T) { return speciesE(s,T)+s.R*T; }
CHMT_HD inline Real condensedE(const CondensedThermo& s,Real T) { return s.e0+s.cp0*T+0.5*s.cp1*T*T; }
CHMT_HD inline Real liquidH(const PhysicsConfig& p,Real T,Real pressure) {
    return condensedE(p.liquid,T)+(pressure-p.liquidReferencePressure)/p.liquid.rho;
}
CHMT_HD inline Real filmInternalEnergy(const FilmQ& q,Real pressure,const PhysicsConfig& p) {
    return q.enthalpy-pressure*q.mass/p.liquid.rho;
}
CHMT_HD inline bool validSpecies(const SpeciesThermo& s) {
    if(!finite(s.R)||s.R<=0||!finite(s.cp0)||!finite(s.cp1)||!finite(s.e0)||!finite(s.Tmin)||!finite(s.Tmax)||s.Tmin<=0||s.Tmax<=s.Tmin)return false;
    const Real cvlo=s.cp0-s.R+s.cp1*s.Tmin,cvhi=s.cp0-s.R+s.cp1*s.Tmax;
    const Real sensibleLo=(s.cp0-s.R)*s.Tmin+0.5*s.cp1*s.Tmin*s.Tmin;
    const Real sensibleHi=(s.cp0-s.R)*s.Tmax+0.5*s.cp1*s.Tmax*s.Tmax;
    const Real elo=speciesE(s,s.Tmin),ehi=speciesE(s,s.Tmax),hlo=speciesH(s,s.Tmin),hhi=speciesH(s,s.Tmax);
    // The first-model gas contract requires positive sensible integral above
    // its declared e0 intercept throughout the operating interval. Positive cv
    // alone would admit intervals whose initialized thermal inventory is invalid.
    return finite(cvlo)&&finite(cvhi)&&cvlo>0&&cvhi>0&&finite(sensibleLo)&&finite(sensibleHi)&&sensibleLo>0&&sensibleHi>0
        &&finite(elo)&&finite(ehi)&&ehi>elo&&finite(hlo)&&finite(hhi)&&hhi>hlo;
}
CHMT_HD inline bool validCondensed(const CondensedThermo& s) {
    if(!finite(s.rho)||s.rho<=0||!finite(s.cp0)||!finite(s.cp1)||!finite(s.e0)||!finite(s.Tmin)||!finite(s.Tmax)||s.Tmin<=0||s.Tmax<=s.Tmin)return false;
    const Real cplo=s.cp0+s.cp1*s.Tmin,cphi=s.cp0+s.cp1*s.Tmax;
    const Real elo=condensedE(s,s.Tmin),ehi=condensedE(s,s.Tmax);
    return finite(cplo)&&finite(cphi)&&cplo>0&&cphi>0&&finite(elo)&&finite(ehi)&&ehi>elo;
}
CHMT_HD inline bool closeEnough(Real a,Real b,Real absolute,Real relative) {
    if(!finite(a)||!finite(b)||!finite(absolute)||!finite(relative)||absolute<0||relative<0)return false;
    const Real residual=a-b,tolerance=absolute+relative*maxValue(absValue(a),absValue(b));
    return finite(residual)&&finite(tolerance)&&absValue(residual)<=tolerance;
}
CHMT_HD inline Real caloricIntegral(Real a0,Real a1,Real a2,Real T) { return a0+T*(a1+0.5*a2*T); }
CHMT_HD inline bool invertCaloric(Real a0,Real a1,Real a2,Real target,Real lo,Real hi,const Tolerances& tol,Real& temperature) {
    if(!finite(a0)||!finite(a1)||!finite(a2)||!finite(target)||!finite(lo)||!finite(hi)||lo<=0||hi<=lo
        ||a1+a2*lo<=0||a1+a2*hi<=0)return false;
    const Real emin=caloricIntegral(a0,a1,a2,lo),emax=caloricIntegral(a0,a1,a2,hi);
    if(!finite(emin)||!finite(emax)||emax<=emin)return false;
    const Real etol=tol.absoluteEnergy+tol.relativeEnergy*maxValue(absValue(target),maxValue(absValue(emin),absValue(emax)));
    if(!finite(etol)||etol<0||target<emin-etol||target>emax+etol)return false;
    if(target<=emin||target>=emax){
        const Real endpoint=target<=emin?lo:hi;
        const Real residual=target<=emin?emin-target:target-emax;
        const Real ttol=tol.absoluteTemperature+tol.relativeTemperature*absValue(endpoint);
        const Real minimumDerivative=minValue(a1+a2*lo,a1+a2*hi);
        const Real temperatureError=residual/minimumDerivative;
        // An integral joule tolerance never authorizes appreciable clipping in
        // K, including tiny inventories or large formation-energy offsets.
        if(!finite(residual)||!finite(ttol)||ttol<0||!finite(minimumDerivative)||minimumDerivative<=0
            ||!finite(temperatureError)||temperatureError>ttol)return false;
        temperature=endpoint;return true;
    }
    Real T=lo+(hi-lo)*(target-emin)/(emax-emin);
    for(int i=0;i<100;++i){
        const Real value=caloricIntegral(a0,a1,a2,T),residual=value-target;
        if(!finite(value)||!finite(residual))return false;
        const Real derivative=a1+a2*T;
        const Real ttol=tol.absoluteTemperature+tol.relativeTemperature*absValue(T);
        if(!finite(ttol)||ttol<0)return false;
        // Require temperature accuracy, not only a relative total-energy test
        // whose scale can be dominated by formation offsets.
        if(absValue(residual)<=etol&&absValue(residual)/derivative<=ttol){temperature=T;return true;}
        if(residual>0)hi=T;else lo=T;
        if(hi-lo<=ttol){temperature=0.5*(lo+hi);return true;}
        const Real trial=T-residual/derivative;
        T=finite(trial)&&trial>lo&&trial<hi?trial:0.5*(lo+hi);
    }
    return false;
}
CHMT_HD inline bool recoverGas(const GasQ& q,Real volume,const PhysicsConfig& p,GasPrimitive& out,DeviceStatus* status,int index) {
    if(!finite(volume)||volume<=0||!finite(q.mass)||q.mass<=0||!finite(q.momentum)||!finite(q.energy))
        return failStatus(status,ErrorCode::InvalidInput,index,q.mass,ErrorLocation::Cell);
    Real sum=0,a0=0,a1=0,a2=0,R=0,lo=0,hi=std::numeric_limits<Real>::max();
    for(int s=0;s<Ns;++s){
        const Real m=q.species[s];if(!finite(m)||m<0)return failStatus(status,ErrorCode::Composition,index,m,ErrorLocation::Cell);
        sum+=m;if(m==0)continue;
        if(!validSpecies(p.species[s]))return failStatus(status,ErrorCode::PropertyRange,index,s,ErrorLocation::Cell);
        const auto& t=p.species[s];a0+=m*t.e0;a1+=m*(t.cp0-t.R);a2+=m*t.cp1;R+=m*t.R;
        lo=maxValue(lo,t.Tmin);hi=minValue(hi,t.Tmax);
    }
    if(!closeEnough(sum,q.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))
        return failStatus(status,ErrorCode::Composition,index,sum-q.mass,ErrorLocation::Cell);
    const Vec3 velocity=q.momentum/q.mass;const Real kinetic=0.5*q.mass*dot(velocity,velocity);
    Real T=0;if(!invertCaloric(a0,a1,a2,q.energy-kinetic,lo,hi,p.tolerances,T))
        return failStatus(status,ErrorCode::PropertyRange,index,q.energy-kinetic,ErrorLocation::Cell);
    if(!(q.energy-kinetic-a0>0))return failStatus(status,ErrorCode::Inventory,index,q.energy-kinetic-a0,ErrorLocation::Cell);
    GasPrimitive candidate{};candidate.rho=q.mass/volume;candidate.velocity=velocity;candidate.temperature=T;
    const Real Rmix=R/q.mass,cv=(a1+a2*T)/q.mass;
    candidate.pressure=candidate.rho*Rmix*T;candidate.soundSpeed=::sqrt((1+Rmix/cv)*Rmix*T);
    for(int s=0;s<Ns;++s)candidate.Y[s]=q.species[s]/q.mass;
    if(!finite(candidate.rho)||!finite(candidate.pressure)||!finite(candidate.soundSpeed))
        return failStatus(status,ErrorCode::Nonfinite,index,T,ErrorLocation::Cell);
    out=candidate;return true;
}
CHMT_HD inline bool recoverGas(const GasQ& q,Real volume,const PhysicsConfig& p,GasPrimitive& out) { return recoverGas(q,volume,p,out,nullptr,-1); }
CHMT_HD inline GasQ invalidGas() {
    GasQ q{};q.mass=q.energy=undefinedValue();q.momentum={undefinedValue(),undefinedValue(),undefinedValue()};
    for(int s=0;s<Ns;++s)q.species[s]=undefinedValue();
    return q;
}
CHMT_HD inline GasQ conservativeGas(const GasPrimitive& w,Real volume,const PhysicsConfig& p) {
    if(!finite(volume)||volume<=0||!finite(w.rho)||w.rho<=0||!finite(w.velocity)||!finite(w.temperature)||w.temperature<=0)return invalidGas();
    Real sum=0;GasQ q{};q.mass=w.rho*volume;q.momentum=q.mass*w.velocity;
    q.energy=0.5*q.mass*dot(w.velocity,w.velocity);
    for(int s=0;s<Ns;++s){
        if(!finite(w.Y[s])||w.Y[s]<0)return invalidGas();
        sum+=w.Y[s];
        q.species[s]=q.mass*w.Y[s];if(w.Y[s]==0)continue;
        if(!validSpecies(p.species[s])||w.temperature<p.species[s].Tmin||w.temperature>p.species[s].Tmax)return invalidGas();
        q.energy+=q.species[s]*speciesE(p.species[s],w.temperature);
    }
    if(!closeEnough(sum,1,p.tolerances.relativeMass,p.tolerances.relativeMass)||!finite(q.mass)||!finite(q.energy))return invalidGas();
    return q;
}
CHMT_HD inline bool recoverSolid(const SolidQ& q,const PhysicsConfig& p,Real& out,DeviceStatus* status,int index) {
    if(!finite(q.energy)||!finite(q.porosity)||q.porosity<0||q.porosity>=1)return failStatus(status,ErrorCode::InvalidInput,index,q.porosity,ErrorLocation::Cell);
    Real mass=0,a0=0,a1=0,a2=0,lo=0,hi=std::numeric_limits<Real>::max();
    for(int r=0;r<Nr;++r)if(!finite(q.progress[r])||q.progress[r]<0)return failStatus(status,ErrorCode::Inventory,index,q.progress[r],ErrorLocation::Cell);
    for(int c=0;c<Nc;++c){
        Real m=q.condensed[c];if(!finite(m)||m<0)return failStatus(status,ErrorCode::Inventory,index,m,ErrorLocation::Cell);
        if(m==0)continue;
        if(!validCondensed(p.condensed[c]))return failStatus(status,ErrorCode::PropertyRange,index,c,ErrorLocation::Cell);
        const auto& t=p.condensed[c];mass+=m;a0+=m*t.e0;a1+=m*t.cp0;a2+=m*t.cp1;lo=maxValue(lo,t.Tmin);hi=minValue(hi,t.Tmax);
    }
    for(int s=0;s<Ns;++s){
        Real m=q.pore[s];if(!finite(m)||m<0)return failStatus(status,ErrorCode::Inventory,index,m,ErrorLocation::Cell);
        if(m==0)continue;
        if(!validSpecies(p.species[s]))return failStatus(status,ErrorCode::PropertyRange,index,s,ErrorLocation::Cell);
        const auto& t=p.species[s];mass+=m;a0+=m*t.e0;a1+=m*(t.cp0-t.R);a2+=m*t.cp1;lo=maxValue(lo,t.Tmin);hi=minValue(hi,t.Tmax);
    }
    Real T=0;if(mass<=0||!invertCaloric(a0,a1,a2,q.energy,lo,hi,p.tolerances,T))return failStatus(status,ErrorCode::PropertyRange,index,q.energy,ErrorLocation::Cell);
    out=T;return true;
}
CHMT_HD inline bool recoverSolid(const SolidQ& q,const PhysicsConfig& p,Real& out) { return recoverSolid(q,p,out,nullptr,-1); }
CHMT_HD inline bool recoverFilm(const FilmQ& q,Real area,Real pressure,const PhysicsConfig& p,FilmAux& out,DeviceStatus* status,int index) {
    if(!finite(q.mass)||q.mass<0||!finite(q.enthalpy)||!finite(area)||area<=0||!finite(pressure)||!validCondensed(p.liquid)||!finite(p.liquidReferencePressure))
        return failStatus(status,ErrorCode::InvalidInput,index,q.mass,ErrorLocation::Face);
    Real sum=0;for(int s=0;s<Ns;++s){if(!finite(q.species[s])||q.species[s]<0)return failStatus(status,ErrorCode::Composition,index,q.species[s],ErrorLocation::Face);sum+=q.species[s];}
    if(!closeEnough(sum,q.mass,p.tolerances.absoluteMass,p.tolerances.relativeMass))return failStatus(status,ErrorCode::Composition,index,sum-q.mass,ErrorLocation::Face);
    FilmAux candidate=out;candidate.area=area;candidate.pressure=pressure;
    if(q.mass==0){
        if(q.enthalpy!=0||sum!=0)return failStatus(status,ErrorCode::Inventory,index,q.enthalpy,ErrorLocation::Face);
        candidate.thickness=0;candidate.temperature=undefinedValue();candidate.kineticEnergy=0;
        candidate.meanVelocity=candidate.baseVelocity;candidate.topVelocity=candidate.baseVelocity;out=candidate;return true;
    }
    const Real target=q.enthalpy-q.mass*(pressure-p.liquidReferencePressure)/p.liquid.rho;
    Real T=0;if(!invertCaloric(q.mass*p.liquid.e0,q.mass*p.liquid.cp0,q.mass*p.liquid.cp1,target,p.liquid.Tmin,p.liquid.Tmax,p.tolerances,T))
        return failStatus(status,ErrorCode::PropertyRange,index,target,ErrorLocation::Face);
    candidate.thickness=q.mass/(p.liquid.rho*area);candidate.temperature=T;
    if(!finite(candidate.thickness))return failStatus(status,ErrorCode::Nonfinite,index,candidate.thickness,ErrorLocation::Face);
    out=candidate;return true;
}
CHMT_HD inline bool recoverFilm(const FilmQ& q,Real area,Real pressure,const PhysicsConfig& p,FilmAux& out) { return recoverFilm(q,area,pressure,p,out,nullptr,-1); }

CHMT_HD inline Real reactionRate(const Reaction& r,const Real* condensedMass,Real volume,Real T) {
    if(!finite(volume)||volume<=0||!finite(T)||T<=0||!finite(r.A)||r.A<=0||!finite(r.temperaturePower)||!finite(r.activationEnergy)||r.activationEnergy<0)return undefinedValue();
    Real rate=r.A*volume*::pow(T,r.temperaturePower)*::exp(-r.activationEnergy/(universalGasConstant*T));
    for(int c=0;c<Nc;++c){
        if(!finite(condensedMass[c])||condensedMass[c]<0||!finite(r.order[c])||r.order[c]<0)return undefinedValue();
        if(r.order[c]>0)rate*=::pow(condensedMass[c]/volume,r.order[c]);
    }
    return finite(rate)&&rate>=0?rate:undefinedValue();
}
CHMT_HD inline Real reactionFormationEnergy(const Reaction& r,const PhysicsConfig& p) {
    Real value=0;for(int c=0;c<Nc;++c)value+=r.condensedNu[c]*p.condensed[c].e0;
    for(int s=0;s<Ns;++s)value+=r.gasNu[s]*p.species[s].e0;
    return value;
}
inline bool validateReaction(const Reaction& r,const PhysicsConfig& p,int index,std::string& error) {
    const std::string prefix="reaction "+std::to_string(index)+": ";
    if(!finite(r.A)||r.A<=0||!finite(r.temperaturePower)||!finite(r.activationEnergy)||r.activationEnergy<0){error=prefix+"invalid/missing Arrhenius parameters";return false;}
    if(p.nElements<=0||p.nElements>Ne){error=prefix+"missing/invalid active element count";return false;}
    if(!finite(p.tolerances.relativeMass)||p.tolerances.relativeMass<0){error=prefix+"invalid mass-balance tolerance";return false;}
    Real sum=0,scale=0,element[Ne]{},elementScale[Ne]{};bool consumes=false,produces=false;
    const auto addCoefficient=[&](Real nu,const Real* composition)->bool {
        if(!finite(nu))return false;
        sum+=nu;scale+=absValue(nu);
        if(!finite(sum)||!finite(scale))return false;
        consumes=consumes||nu<0;produces=produces||nu>0;
        Real defined=0;
        for(int e=0;e<Ne;++e){
            const Real n=composition[e];if(!finite(n)||n<0)return false;
            if(e<p.nElements){defined+=n;if(!finite(defined))return false;}
            const Real product=nu*n;if(!finite(product))return false;
            element[e]+=product;elementScale[e]+=absValue(product);
            if(!finite(element[e])||!finite(elementScale[e]))return false;
        }
        return nu==0||defined>0;
    };
    for(int c=0;c<Nc;++c){
        if(!finite(r.order[c])||r.order[c]<0||(r.order[c]>0&&r.condensedNu[c]>=0)){
            error=prefix+"invalid condensed reaction order";return false;
        }
        if(!addCoefficient(r.condensedNu[c],p.condensed[c].element)){
            error=prefix+"missing/invalid/nonfinite condensed mass or element arithmetic";return false;
        }
    }
    for(int s=0;s<Ns;++s)if(!addCoefficient(r.gasNu[s],p.species[s].element)){
        error=prefix+"missing/invalid/nonfinite gas mass or element arithmetic";return false;
    }
    const Real massTolerance=64*std::numeric_limits<Real>::epsilon()+p.tolerances.relativeMass;
    if(!finite(massTolerance)||!consumes||!produces||scale==0||absValue(sum)/scale>massTolerance){error=prefix+"stoichiometric mass imbalance";return false;}
    // Element residuals are mol/mol extent: their threshold is dimensionless
    // roundoff, not the absolute kg inventory tolerance used by time updates.
    for(int e=0;e<Ne;++e){
        const Real threshold=64*std::numeric_limits<Real>::epsilon()*maxValue(1,elementScale[e]);
        if(!finite(threshold)||absValue(element[e])>threshold){error=prefix+"element "+std::to_string(e)+" imbalance";return false;}
    }
    error.clear();return true;
}

// Restrict one molar extent before rebuilding ALL linked mass changes. No
// component clipping and no independent heat source accompanies formation e0.
CHMT_HD inline bool limitReactionExtent(const Reaction& r,const SolidQ& q,Real requested,Real& accepted,DeviceStatus* status=nullptr,int index=-1) {
    if(!finite(requested)||requested<0)return failStatus(status,ErrorCode::InvalidInput,index,requested,ErrorLocation::Cell);
    Real extent=requested;
    for(int c=0;c<Nc;++c){
        if(!finite(q.condensed[c])||q.condensed[c]<0||!finite(r.condensedNu[c]))return failStatus(status,ErrorCode::Inventory,index,q.condensed[c],ErrorLocation::Cell);
        if(r.condensedNu[c]<0)extent=minValue(extent,q.condensed[c]/(-r.condensedNu[c]));
    }
    for(int s=0;s<Ns;++s){
        if(!finite(q.pore[s])||q.pore[s]<0||!finite(r.gasNu[s]))return failStatus(status,ErrorCode::Inventory,index,q.pore[s],ErrorLocation::Cell);
        if(r.gasNu[s]<0)extent=minValue(extent,q.pore[s]/(-r.gasNu[s]));
    }
    if(!finite(extent))return failStatus(status,ErrorCode::Nonfinite,index,extent,ErrorLocation::Cell);
    accepted=extent;return true;
}
CHMT_HD inline bool applyReactionExtent(const Reaction& r,int reactionIndex,Real extent,SolidQ& q,DeviceStatus* status=nullptr,int index=-1) {
    if(reactionIndex<0||reactionIndex>=Nr||!finite(q.energy)||!finite(q.porosity)||q.porosity<0||q.porosity>=1)
        return failStatus(status,ErrorCode::InvalidInput,index,reactionIndex,ErrorLocation::Cell);
    Real bound=0;if(!limitReactionExtent(r,q,extent,bound,status,index))return false;
    if(extent>bound)return failStatus(status,ErrorCode::Inventory,index,extent,ErrorLocation::Cell);
    SolidQ candidate=q;
    for(int c=0;c<Nc;++c){candidate.condensed[c]+=extent*r.condensedNu[c];if(!finite(candidate.condensed[c])||candidate.condensed[c]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.condensed[c],ErrorLocation::Cell);}
    for(int s=0;s<Ns;++s){candidate.pore[s]+=extent*r.gasNu[s];if(!finite(candidate.pore[s])||candidate.pore[s]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.pore[s],ErrorLocation::Cell);}
    candidate.progress[reactionIndex]+=extent;
    if(!finite(candidate.progress[reactionIndex])||candidate.progress[reactionIndex]<0)return failStatus(status,ErrorCode::Inventory,index,candidate.progress[reactionIndex],ErrorLocation::Cell);
    q=candidate;return true;
}
struct ValidationContext {
    bool gasThermo=true;
    unsigned condensedMask=(1u<<Nc)-1;
    bool liquid=true;
};
inline bool validatePhysicsConfig(const PhysicsConfig& p,const ValidationContext& context,std::string& error) {
    for(int s=0;s<Ns;++s)if(context.gasThermo&&!validSpecies(p.species[s])){error="configuration gas species "+std::to_string(s)+": missing/invalid thermo table";return false;}
    for(int c=0;c<Nc;++c)if((context.condensedMask&(1u<<c))&&(!validCondensed(p.condensed[c])||!finite(p.condensed[c].conductivity)||p.condensed[c].conductivity<0)){error="configuration condensed constituent "+std::to_string(c)+": missing/invalid property table";return false;}
    if(p.nReactions<0||p.nReactions>Nr||p.nElements<0||p.nElements>Ne){error="configuration reaction/element count exceeds compiled first-model limits";return false;}
    if(p.enableReactions&&p.nReactions==0){error="configuration reactions enabled without any mechanism";return false;}
    for(int r=0;r<p.nReactions;++r)if(!validateReaction(p.reactions[r],p,r,error))return false;
    if(context.liquid&&p.enableFilm&&(!validCondensed(p.liquid)||!finite(p.liquid.conductivity)||p.liquid.conductivity<0
        ||!finite(p.liquidViscosity)||p.liquidViscosity<=0||!finite(p.liquidReferencePressure)
        ||((p.material.enableMelting||p.filmThermalMode==FilmThermalMode::ResolvedNormal)
            &&(!finite(p.meltTemperature)||p.meltTemperature<p.liquid.Tmin||p.meltTemperature>p.liquid.Tmax)))){
        error="configuration film: missing/invalid liquid thermo, viscosity, reference pressure or melt temperature";return false;
    }
    if(!finite(p.permeability)||p.permeability<0||!finite(p.poreViscosity)||p.poreViscosity<0||(p.permeability>0&&p.poreViscosity<=0)
        ||!finite(p.gasViscosity)||p.gasViscosity<0||!finite(p.gasConductivity)||p.gasConductivity<0||!finite(p.gravity)){
        error="configuration transport/gravity: invalid coefficient";return false;
    }
    for(int s=0;s<Ns;++s)if(!finite(p.gasDiffusivity[s])||p.gasDiffusivity[s]<0){error="configuration diffusivity species "+std::to_string(s)+": invalid coefficient";return false;}
    if(!finite(p.emissivity)||p.emissivity<0||p.emissivity>1||!finite(p.ambientTemperature)||p.ambientTemperature<0||(p.enableRadiation&&p.ambientTemperature<=0)){
        error="configuration radiation: invalid emissivity or missing ambient temperature";return false;
    }
    const Real absolute[]={p.tolerances.absoluteMass,p.tolerances.absoluteEnergy,p.tolerances.absoluteTemperature,p.tolerances.absoluteGeometry};
    const Real relative[]={p.tolerances.relativeMass,p.tolerances.relativeEnergy,p.tolerances.relativeTemperature,p.tolerances.relativeGeometry};
    for(int i=0;i<4;++i)if(!finite(absolute[i])||absolute[i]<0||!finite(relative[i])||relative[i]<0||absolute[i]+relative[i]==0){error="configuration tolerance "+std::to_string(i)+": must be finite, nonnegative and nonzero";return false;}
    if(p.tolerances.maxCouplingIterations<1||p.tolerances.maxRetries<0||!finite(p.minDt)||p.minDt<=0||!finite(p.maxDt)||p.maxDt<p.minDt||!finite(p.cfl)||p.cfl<=0||p.cfl>1
        ||(p.spatialOrder!=1&&p.spatialOrder!=2)){error="configuration timestep/order: required explicit positive minDt, maxDt, CFL and supported order";return false;}
    if(p.speciesFingerprint!=compiledSpeciesHash||p.schemaFingerprint!=checkpointSchema||p.modelFingerprint==0){error="configuration fingerprints: missing/incompatible model, species or schema hash";return false;}
    if(p.enableSst){
        const auto& s=p.sst;
        const Real values[]={s.a1,s.betaStar,s.beta1,s.beta2,s.gamma1,s.gamma2,s.sigmaK1,s.sigmaK2,s.sigmaOmega1,s.sigmaOmega2,s.turbulentPrandtl,s.turbulentSchmidt,s.productionLimit};
        for(Real value:values)if(!finite(value)||value<=0){error="configuration SST: invalid model constant";return false;}
        if(!finite(s.minimumK)||s.minimumK<0||!finite(s.minimumOmega)||s.minimumOmega<0){error="configuration SST: invalid admissibility threshold";return false;}
    }
    if(p.reconstruction!=ReconstructionMode::LimitedLinear&&p.reconstruction!=ReconstructionMode::SmoothVerification){error="configuration reconstruction: unsupported policy";return false;}
    if(p.filmThermalMode!=FilmThermalMode::ThicknessAveraged&&p.filmThermalMode!=FilmThermalMode::ResolvedNormal){error="configuration film: unsupported thermal mode";return false;}
    if(p.meshMotion.policy!=MeshMotionPolicy::Static&&p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession&&p.meshMotion.policy!=MeshMotionPolicy::PrescribedSinusoidal){error="configuration mesh motion: unsupported policy";return false;}
    if(!finite(p.meshMotion.amplitude)||!finite(p.meshMotion.spatialWaveNumber)||!finite(p.meshMotion.spatialOrigin)||!finite(p.meshMotion.angularFrequency)||!finite(p.meshMotion.timeOrigin)){error="configuration mesh motion: nonfinite input";return false;}
    if(!p.enableGas&&(p.enableSst||p.enableParticles)){error="SST/particles require external gas enabled";return false;}
    error.clear();return true;
}
inline bool validatePhysicsConfig(const PhysicsConfig& p,std::string& error) {
    return validatePhysicsConfig(p,ValidationContext{},error);
}
} // namespace chmt
#endif
