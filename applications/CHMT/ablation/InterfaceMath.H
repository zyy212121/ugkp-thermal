#ifndef CHMT_ABLATION_INTERFACEMATH_H
#define CHMT_ABLATION_INTERFACEMATH_H
#include "film/FilmMath.H"
#include "gpu/CommonGasTraction.H"
#include "../../../common/gasWall/WallModel.H"
namespace chmt {
CHMT_HD inline Real interfacePartialPressure(Real pressure, const Real* Y,
    const PhysicsConfig& p, int species) {
    Real R=0;
    for (int s=0; s<Ns; ++s) R+=Y[s]*p.species[s].R;
    return R>0 ? pressure*Y[species]*p.species[species].R/R : undefinedValue();
}
CHMT_HD inline bool compositionResidual(const Real* Y, const Real* bulk, const Real* Js,
    Real pressure, Real T, Real distance, const PhysicsConfig& p, Real* residual, Real* jacobian) {
    Real R=0, J=0, S=0, a[Ns]{};
    for (int s=0; s<Ns; ++s) { R+=Y[s]*p.species[s].R; J+=Js[s]; }
    if (!(R>0 && pressure>0 && T>0 && distance>0)) return false;
    const Real rho=pressure/(R*T);
    for (int s=0; s<Ns; ++s) {
        a[s]=rho*p.gasDiffusivity[s]/distance;
        S+=a[s]*(Y[s]-bulk[s]);
    }
    for (int s=0; s<Ns; ++s) {
        const Real diffusion=a[s]*(Y[s]-bulk[s])-Y[s]*S;
        residual[s]=J*Y[s]+diffusion-Js[s];
        if (!finite(residual[s])) return false;
        if (jacobian) {
            for (int t=0; t<Ns; ++t) {
                jacobian[s*Ns+t]=(s==t ? J+a[s]-S : 0)-Y[s]*a[t]
                    -diffusion*p.species[t].R/R;
            }
        }
    }
    return true;
}
CHMT_HD inline bool interfaceComposition(const Real* Js, const Real* bulk, Real pressure,
    Real T, Real distance, const PhysicsConfig& p, Real absoluteFluxTolerance, Real* out) {
    Real J=0, scale=0, sum=0, maxD=0;
    for (int s=0; s<Ns; ++s) {
        if (!finite(Js[s]) || !finite(bulk[s]) || bulk[s]<0
            || !finite(p.gasDiffusivity[s]) || p.gasDiffusivity[s]<0) return false;
        J+=Js[s]; scale+=absValue(Js[s]); sum+=bulk[s];
        maxD=maxValue(maxD,p.gasDiffusivity[s]);
    }
    if (!closeEnough(sum,1,1e-13,p.tolerances.relativeMass)
        || !finite(absoluteFluxTolerance) || absoluteFluxTolerance<0) return false;
    const Real tolerance=absoluteFluxTolerance+p.tolerances.relativeMass*scale;
    Real Y[Ns]{};
    if (maxD==0) {
        for (int s=0; s<Ns; ++s) {
            if (J>0) Y[s]=Js[s]/J;
            else {
                Y[s]=bulk[s];
                if (absValue(Js[s]-J*Y[s])>tolerance) return false;
            }
            if (!finite(Y[s]) || Y[s]<0 || Y[s]>1) return false;
        }
        for (int s=0; s<Ns; ++s) out[s]=Y[s];
        return true;
    }
    for (int s=0; s<Ns; ++s) Y[s]=bulk[s];
    for (int iteration=0; iteration<60; ++iteration) {
        Real F[Ns]{}, jac[Ns*Ns]{};
        if (!compositionResidual(Y,bulk,Js,pressure,T,distance,p,F,jac)) return false;
        Real norm=0;
        for (int s=0; s<Ns; ++s) norm=maxValue(norm,absValue(F[s]));
        if (norm<=tolerance) {
            for (int s=0; s<Ns; ++s) out[s]=Y[s];
            return true;
        }
        Real matrix[Ns][Ns+1]{};
        int normalizationRow=-1;
        Real normalization=0;
        for (int species=0; species<Ns; ++species) {
            normalization+=Y[species];
            if (bulk[species]>0 || Js[species]!=0 || p.gasDiffusivity[species]>0) {
                normalizationRow=species;
            }
        }
        if (normalizationRow<0) return false;
        for (int species=0; species<Ns; ++species) {
            const bool absent=bulk[species]==0 && Js[species]==0 && p.gasDiffusivity[species]==0;
            if (species==normalizationRow) {
                for (int t=0; t<Ns; ++t) matrix[species][t]=1;
                matrix[species][Ns]=1-normalization;
            } else if (absent) {
                matrix[species][species]=1;
                matrix[species][Ns]=-Y[species];
            } else {
                for (int t=0; t<Ns; ++t) matrix[species][t]=jac[species*Ns+t];
                matrix[species][Ns]=-F[species];
            }
        }
        for (int col=0; col<Ns; ++col) {
            int pivot=col;
            for (int row=col+1; row<Ns; ++row) {
                if (absValue(matrix[row][col])>absValue(matrix[pivot][col])) pivot=row;
            }
            if (!finite(matrix[pivot][col]) || absValue(matrix[pivot][col])<1e-30) return false;
            for (int t=col; t<=Ns; ++t) {
                const Real tmp=matrix[col][t]; matrix[col][t]=matrix[pivot][t]; matrix[pivot][t]=tmp;
            }
            const Real diagonal=matrix[col][col];
            for (int t=col; t<=Ns; ++t) matrix[col][t]/=diagonal;
            for (int row=0; row<Ns; ++row) if (row!=col) {
                const Real factor=matrix[row][col];
                for (int t=col; t<=Ns; ++t) matrix[row][t]-=factor*matrix[col][t];
            }
        }
        Real alpha=1;
        bool found=false;
        for (int line=0; line<50; ++line) {
            Real trial[Ns]{}, residual[Ns]{};
            bool admissible=true;
            for (int s=0; s<Ns; ++s) {
                trial[s]=Y[s]+alpha*matrix[s][Ns];
                admissible=admissible && finite(trial[s]) && trial[s]>=0 && trial[s]<=1;
            }
            if (admissible && compositionResidual(trial,bulk,Js,pressure,T,distance,p,residual,nullptr)) {
                Real nextNorm=0;
                for (int s=0; s<Ns; ++s) nextNorm=maxValue(nextNorm,absValue(residual[s]));
                if (nextNorm<norm || nextNorm<=tolerance) {
                    for (int s=0; s<Ns; ++s) Y[s]=trial[s];
                    found=true; break;
                }
            }
            alpha*=0.5;
        }
        if (!found) return false;
    }
    return false;
}
CHMT_HD inline bool makeInterfacePrimitive(const Real* Y, Real T, Real pressure,
    Vec3 tangentialVelocity, Vec3 normal, Real normalSpeed, Real J,
    const PhysicsConfig& p, GasPrimitive& out) {
    Real R=0, cv=0, sum=0;
    for (int s=0; s<Ns; ++s) {
        if (!validSpecies(p.species[s]) || !finite(Y[s]) || Y[s]<0
            || T<p.species[s].Tmin || T>p.species[s].Tmax) return false;
        R+=Y[s]*p.species[s].R;
        cv+=Y[s]*speciesCv(p.species[s],T);
        sum+=Y[s];
    }
    if (!closeEnough(sum,1,1e-13,p.tolerances.relativeMass) || pressure<=0 || R<=0 || cv<=0) return false;
    GasPrimitive state;
    state.temperature=T; state.pressure=pressure; state.rho=pressure/(R*T);
    state.velocity=tangentialVelocity-normal*dot(tangentialVelocity,normal)
        +normal*(normalSpeed+J/state.rho);
    state.soundSpeed=::sqrt((1+R/cv)*R*T);
    for (int s=0; s<Ns; ++s) state.Y[s]=Y[s];
    if (!finite(state.velocity) || !finite(state.rho) || state.rho<=0) return false;
    out=state; return true;
}
// Borrowed stage data. The matching state and geometric quadrature are rebuilt
// against the current stage; canonical mechanism storage remains with the caller.
struct GasWallClosureContext {
    ugkwp::GasMechanismView<Real,Ns> mechanism{};
    ugkwp::gaswall::WallMatchingState<Real,Ns> matching{};
    ugkwp::gaswall::WallOwnerQuadrature<Real> quadrature{};
    // Matching pressure closes EOS/BVP/film thermodynamics. Mechanical pressure
    // is the local gas-face pressure used by momentum and its paired sweep work.
    Real matchingPressure=0,mechanicalPressure=0,matchingDistance=0,ownerDistance=0;
    void* workspace=nullptr;int workspaceCapacity=48;
    ugkwp::gaswall::WallStatus* failureStatus=nullptr;
};
struct InterfaceGasClosure {
    GasPrimitive trace{};Vec3 traction{};Real conductiveHeatFlux=0;
    ugkwp::gaswall::WallOutput<Real,Ns> layer{};
};
CHMT_HD inline bool evaluateInterfaceGasClosure(const GasPrimitive& bulk,const GasGradient& inputGradient,
    Real temperature,Real distance,Vec3 velocity,Vec3 normal,Real normalSpeed,
    const Real* species,Real fluxTolerance,const PhysicsConfig& p,
    const GasWallClosureContext& context,InterfaceGasClosure& output) {
    InterfaceGasClosure result;
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        if(p.material.gasContactResistance!=0)return false;
        const GasThermoAdapter<Ns> thermo(p.species);
        ugkwp::gaswall::WallInput<Real,Ns> input;
        input.pressure=context.matchingPressure;input.temperature=temperature;
        input.matchingDistance=context.matchingDistance;input.ownerDistance=context.ownerDistance;
        input.matching=context.matching;input.quadrature=context.quadrature;
        input.model.thermo=thermo.view();input.model.thermo.speciesOrderHash=p.speciesFingerprint;
        Real elements[Ns*Ne]{};input.model.thermo.elementCount=p.nElements;input.model.thermo.elementComposition=elements;
        for(int s=0;s<Ns;++s)for(int e=0;e<p.nElements;++e)elements[s*p.nElements+e]=p.species[s].element[e]*thermo.data[s].molarMass;
        input.model.mechanism=context.mechanism;input.model.mode=p.gasMode;
        input.model.viscosity=p.gasViscosity;input.model.conductivity=p.gasConductivity;
        auto& coefficients=input.model.sstCoefficients;const auto& sst=p.sst;
        coefficients.a1=sst.a1;coefficients.betaStar=sst.betaStar;coefficients.beta1=sst.beta1;coefficients.beta2=sst.beta2;
        coefficients.gamma1=sst.gamma1;coefficients.gamma2=sst.gamma2;coefficients.alphaK1=sst.sigmaK1;coefficients.alphaK2=sst.sigmaK2;
        coefficients.alphaOmega1=sst.sigmaOmega1;coefficients.alphaOmega2=sst.sigmaOmega2;coefficients.c1=sst.productionLimit;
        velocity=velocity-normal*dot(velocity,normal)+normal*normalSpeed;
        input.velocity[0]=velocity.x;input.velocity[1]=velocity.y;input.velocity[2]=velocity.z;
        input.normal[0]=normal.x;input.normal[1]=normal.y;input.normal[2]=normal.z;
        for(int s=0;s<Ns;++s){input.massFlux[s]=species[s];input.model.diffusivity[s]=p.gasDiffusivity[s];}
        auto config=p.wallModel;config.enableSst=p.enableSst;
        config.turbulentPrandtl=p.sst.turbulentPrandtl;config.turbulentSchmidt=p.sst.turbulentSchmidt;
        ugkwp::gaswall::WallStatus status;
        bool accepted=false;
        if(config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport)
            accepted=ugkwp::gaswall::evaluateConstantTransportWall(input,config,result.layer,status);
        else if(context.workspace){
            if(context.workspaceCapacity==24)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,24>*>(context.workspace),result.layer,status);
            else if(context.workspaceCapacity==48)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,48>*>(context.workspace),result.layer,status);
            else if(context.workspaceCapacity==128)accepted=ugkwp::gaswall::evaluateWallModel(input,config,*static_cast<ugkwp::gaswall::WallWorkspace<Real,Ns,128>*>(context.workspace),result.layer,status);
        }
        if(!accepted){if(context.failureStatus)*context.failureStatus=status;return false;}
        Real mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!makeInterfacePrimitive(result.layer.traceMassFraction,temperature,input.pressure,velocity,normal,
            normalSpeed,mass,p,result.trace))return false;
        result.traction={result.layer.traction[0],result.layer.traction[1],result.layer.traction[2]};
        result.conductiveHeatFlux=result.layer.conductiveHeatFlux;
    }else{
        // The previous low-Re path stays bitwise unchanged. Ordinary wall
        // functions have no CHMT mass-transfer extension.
        if(p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe)return false;
        Real Y[Ns]{},mass=0;for(int s=0;s<Ns;++s)mass+=species[s];
        if(!interfaceComposition(species,bulk.Y,bulk.pressure,temperature,distance,p,fluxTolerance,Y)
            ||!makeInterfacePrimitive(Y,temperature,bulk.pressure,velocity,normal,normalSpeed,mass,p,result.trace))return false;
        GasGradient gradient=inputGradient;const Vec3 difference=bulk.velocity-result.trace.velocity;
        gradient.velocity[0]+=normal*(difference.x/distance-dot(gradient.velocity[0],normal));
        gradient.velocity[1]+=normal*(difference.y/distance-dot(gradient.velocity[1],normal));
        gradient.velocity[2]+=normal*(difference.z/distance-dot(gradient.velocity[2],normal));
        result.traction=viscousTraction(gradient,normal,p.gasViscosity,0);
        const Real conductance=p.gasConductivity>0?1/(distance/p.gasConductivity+p.material.gasContactResistance):0;
        result.conductiveHeatFlux=conductance*(temperature-bulk.temperature);
    }
    output=result;return true;
}
CHMT_HD inline bool hertzKnudsenFlux(const FilmQ& film, const Real* gasY, Real pressure,
    Real T, const PhysicsConfig& p, Real* flux) {
    if (film.mass<=0 || T<=0 || !finite(T)) return false;
    Real molecularSum=0;
    for (int s=0; s<Ns; ++s) molecularSum+=film.species[s]*p.species[s].R;
    if (!(molecularSum>0)) return false;
    for (int s=0; s<Ns; ++s) {
        const Real alpha=p.material.evaporationAccommodation[s];
        const Real activity=film.species[s]*p.species[s].R/molecularSum;
        const Real saturation=p.material.saturationPressureScale[s]
            *::exp(-p.material.saturationTemperature[s]/T);
        flux[s]=alpha*(activity*saturation-interfacePartialPressure(pressure,gasY,p,s))
            /::sqrt(2*3.14159265358979323846*p.species[s].R*T);
        if (!finite(flux[s])) return false;
    }
    return true;
}

struct SurfacePhysicsInput {
    GasWallClosureContext wallContext{};
    GasPrimitive bulk{};
    GasGradient gradient{};
    GasQ gasInventory{};
    SolidQ solid{}, solidBase{}, solidLocalRate{};
    FilmQ film{}, filmBase{}, filmLocalRate{};
    FilmAux aux{};
    bool hasSolid=false, hasFilm=false, useAcceptedFlux=false;
    Real solidVolume=0, area=0, gasDistance=0, solidDistance=0, dt=0;
    Vec3 normal{}, baseVelocity{};
    Real gasArea=0, solidArea=0;
    Vec3 gasNormal{}, solidNormal{}; // both point from material toward gas.
    bool useSweptGeometry=false;
    Real gasSweptRate=0, solidSweptRate=0; // physical normal swept volume rates, m^3/s.
    Real filmTransportMassRate=0, sideMeshVolumeRate=0;
    Real acceptedGasSpecies[Ns]{}, acceptedPoreSpecies[Ns]{}, acceptedPoreSweep[Ns]{}, acceptedCondensed[Nc]{};
    Real acceptedPhaseMass=0;
};
struct SurfacePhysicsResult {
    GasPrimitive gasState{};
    Real gasSpecies[Ns]{}, poreSpecies[Ns]{}, poreDarcy[Ns]{}, poreSweep[Ns]{}, condensed[Nc]{}, phaseMass=0;
    Real gasEnergy=0, poreEnergy=0, phaseEnergy=0;
    Real gasAdvective=0, gasConductive=0, gasPressureWork=0, gasViscousWork=0;
    Real phaseAdvective=0, phaseConductive=0, phasePressureWork=0, phaseViscousWork=0;
    Real topKineticAdvection=0, bottomKineticAdvection=0;
    Real topTemperature=0, bottomTemperature=0, solidSpeed=0, topSpeed=0;
    Real radiationFlux=0, thermalResidual=0;
    Real gasWallSpeed=0, solidWallSpeed=0;
    Real liquidTopNormal=0, liquidBottomNormal=0; // components along reference normal.
    Real solidRecessionResidual=0, filmVolumeResidual=0; // m^3/s, not heat sources.
    Vec3 shear{}, gasShear{}, traction{};
    bool wet=false;
};
CHMT_HD inline Real phaseRateFromHeat(Real power, Real latent, Real densityJump) {
    if (!(latent>0) || !finite(power)) return undefinedValue();
    Real low=0, high=absValue(power)/latent;
    for (int iteration=0; iteration<80; ++iteration) {
        const Real rate=0.5*(low+high);
        const Real value=rate*(latent+0.5*rate*rate*densityJump*densityJump);
        if (value>absValue(power)) high=rate; else low=rate;
    }
    return (power>=0 ? 1 : -1)*0.5*(low+high);
}
CHMT_HD inline Real surfaceArrheniusRate(const Reaction& r, const Real* gasOrder,
    const SurfacePhysicsInput& in, const GasPrimitive& gas, Real T) {
    Real rate=r.A*::pow(T,r.temperaturePower)*::exp(-r.activationEnergy/(universalGasConstant*T));
    for (int c=0; c<Nc; ++c) if (r.order[c]>0) {
        rate*=::pow(in.solid.condensed[c]/in.solidVolume,r.order[c]);
    }
    for (int s=0; s<Ns; ++s) if (gasOrder[s]>0) {
        rate*=::pow(gas.rho*gas.Y[s],gasOrder[s]);
    }
    return rate;
}
CHMT_HD inline bool evaluateSurfaceTemperature(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, const MaterialPrimitive& material, Real filmT, Real topT,
    bool birth, SurfacePhysicsResult& out) {
    if(p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer)return false;
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:in.bulk.pressure;
    if(!finite(mechanicalPressure)||mechanicalPressure<=0)return false;
    SurfacePhysicsResult result;
    const int phase=p.material.phaseCondensed;
    const Real gasRatio=in.gasArea/in.area;
    const Real solidRatio=in.hasSolid ? in.solidArea/in.area : 0;
    const Real gasProjection=dot(in.normal,in.gasNormal);
    const Real solidProjection=in.hasSolid ? dot(in.normal,in.solidNormal) : 1;
    result.wet=in.hasFilm && (in.film.mass>0 || birth);
    result.topTemperature=topT;
    const Real gasConductance=p.gasConductivity>0
        ? 1/(in.gasDistance/p.gasConductivity+p.material.gasContactResistance) : 0;
    const Real solidConductance=in.hasSolid && material.conductivity>0
        ? 1/(in.solidDistance/material.conductivity+p.material.solidContactResistance) : 0;
    const Real thickness=in.hasFilm && in.film.mass>0 ? in.film.mass/(p.liquid.rho*in.area) : 0;
    const Real liquidConductance=thickness>0 && p.liquid.conductivity>0
        ? 2*p.liquid.conductivity/thickness : 0;
    result.radiationFlux=gasRatio*grayRadiation(topT,p);
    Real bottomT=in.hasSolid ? material.temperature : filmT;
    Real phaseRate=0;
    if (result.wet && in.hasSolid) {
        if (p.material.enableMelting) {
            if (phase<0 || phase>=Nc) return false;
            bottomT=p.meltTemperature;
            const Real latent=liquidH(p,bottomT,in.bulk.pressure)
                -condensedE(p.condensed[phase],bottomT)-in.bulk.pressure/p.condensed[phase].rho;
            Real gasIncomingHeat=gasRatio*gasConductance*(in.bulk.temperature-bottomT);
            if(birth&&p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
                Real zeroFlux[Ns]{};InterfaceGasClosure birthClosure;
                if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,bottomT,in.gasDistance,in.baseVelocity,
                    in.gasNormal,0,zeroFlux,p.tolerances.absoluteMass,p,in.wallContext,birthClosure))return false;
                gasIncomingHeat=-gasRatio*birthClosure.conductiveHeatFlux;
            }
            const Real heat=birth
                ? solidConductance*(material.temperature-bottomT)
                    +(gasIncomingHeat+result.radiationFlux)/solidRatio
                : solidConductance*(material.temperature-bottomT)
                    +solidProjection*liquidConductance*(filmT-bottomT);
            phaseRate=solidRatio*phaseRateFromHeat(heat,latent,1/p.liquid.rho-1/p.condensed[phase].rho);
            if (!finite(phaseRate)) return false;
            if (birth && phaseRate<0) return false;
        } else if (solidConductance+solidProjection*liquidConductance>0) {
            bottomT=(solidConductance*material.temperature+solidProjection*liquidConductance*filmT)
                /(solidConductance+solidProjection*liquidConductance);
        }
    }
    if (in.useAcceptedFlux) phaseRate=in.acceptedPhaseMass;
    result.bottomTemperature=bottomT;
    Real Y[Ns]{};
    for (int s=0; s<Ns; ++s) Y[s]=in.bulk.Y[s];
    bool compositionConverged=false;
    InterfaceGasClosure wallClosure;
    for (int iteration=0; iteration<60; ++iteration) {
        SurfacePhysicsResult trial;
        trial.wet=result.wet;
        trial.phaseMass=phaseRate;
        GasPrimitive provisional;
        if (!makeInterfacePrimitive(Y,topT,in.bulk.pressure,{},in.gasNormal,0,0,p,provisional)) return false;
        if (in.useAcceptedFlux) {
            for (int s=0; s<Ns; ++s) {
                trial.gasSpecies[s]=in.acceptedGasSpecies[s];
                trial.poreSpecies[s]=in.acceptedPoreSpecies[s];
                trial.poreSweep[s]=in.acceptedPoreSweep[s];
                trial.poreDarcy[s]=trial.poreSpecies[s]-trial.poreSweep[s];
            }
            for (int c=0; c<Nc; ++c) trial.condensed[c]=in.acceptedCondensed[c];
        } else {
            if (result.wet && p.material.evaporation==EvaporationLaw::HertzKnudsen && !birth) {
                if (!hertzKnudsenFlux(in.film,Y,in.bulk.pressure,topT,p,trial.gasSpecies)) return false;
                for (int species=0; species<Ns; ++species) trial.gasSpecies[species]*=gasRatio;
            }
            if (!result.wet && p.material.enableSurfaceReactions) {
                if (!in.hasSolid) return false;
                for (int r=0; r<p.material.nSurfaceReactions; ++r) {
                    const Reaction& reaction=p.material.surfaceReactions[r];
                    const Real rate=solidRatio*surfaceArrheniusRate(
                        reaction,p.material.surfaceGasOrder[r],in,provisional,topT);
                    if (!finite(rate) || rate<0) return false;
                    for (int c=0; c<Nc; ++c) trial.condensed[c]-=rate*reaction.condensedNu[c];
                    for (int s=0; s<Ns; ++s) trial.gasSpecies[s]+=rate*reaction.gasNu[s];
                }
            }
        }
        Real consumedVolume=0;
        for (int c=0; c<Nc; ++c) if (trial.condensed[c]!=0) {
            if (!in.hasSolid || !validCondensed(p.condensed[c])) return false;
            consumedVolume+=trial.condensed[c]/p.condensed[c].rho;
        }
        if (trial.phaseMass!=0) {
            if (!in.hasSolid || phase<0 || phase>=Nc || !validCondensed(p.condensed[phase])) return false;
            consumedVolume+=trial.phaseMass/p.condensed[phase].rho;
        }
        const Real packing=in.hasSolid ? 1-material.porosity : 1;
        trial.solidSpeed=in.hasSolid ? -consumedVolume/(solidRatio*packing) : 0;
        trial.solidWallSpeed=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate/in.solidArea : trial.solidSpeed;
        if (!in.useAcceptedFlux && in.hasSolid) {
            Real superficial=0;
            if (p.material.enablePoreOutflow && p.permeability>0) {
                superficial=p.permeability/p.poreViscosity
                    *((material.pressure-in.bulk.pressure)/in.solidDistance
                        +material.poreDensity*dot(p.gravity,in.solidNormal));
            }
            // Recession exposes the pore volume even if permeability is zero.
            const Real poreSweep=-material.porosity*trial.solidWallSpeed;
            if (result.wet && (superficial!=0 || poreSweep!=0)) return false;
            for (int s=0; s<Ns; ++s) {
                const Real poreDensity=material.porosity>0
                    ? in.solid.pore[s]/(material.porosity*in.solidVolume) : 0;
                trial.poreDarcy[s]=solidRatio*(superficial>=0 ? superficial*poreDensity
                    : superficial*provisional.rho*Y[s]);
                trial.poreSweep[s]=solidRatio*(poreSweep>=0 ? poreSweep*poreDensity
                    : poreSweep*provisional.rho*Y[s]);
                trial.poreSpecies[s]=trial.poreDarcy[s]+trial.poreSweep[s];
            }
        }
        Real factor=1;
        if (!in.useAcceptedFlux && in.dt>0) {
            const Real scale=in.area*in.dt;
            for (int c=0; c<Nc; ++c) {
                const Real loss=trial.condensed[c]+(c==phase ? trial.phaseMass : 0);
                if (loss>0) {
                    const Real available=in.solidBase.condensed[c]+in.dt*in.solidLocalRate.condensed[c];
                    factor=minValue(factor,available/(scale*loss));
                }
            }
            for (int s=0; s<Ns; ++s) {
                if (trial.poreSpecies[s]>0) {
                    const Real available=in.solidBase.pore[s]+in.dt*in.solidLocalRate.pore[s];
                    factor=minValue(factor,available/(scale*trial.poreSpecies[s]));
                }
                const Real gasLoss=-trial.gasSpecies[s]-trial.poreSpecies[s];
                if (gasLoss>0) factor=minValue(factor,in.gasInventory.species[s]/(scale*gasLoss));
                if (result.wet) {
                    const Real filmLoss=trial.gasSpecies[s]-trial.phaseMass*p.material.phaseFilmY[s];
                    if (filmLoss>0) {
                        const Real available=in.filmBase.species[s]+in.dt*in.filmLocalRate.species[s];
                        factor=minValue(factor,available/(scale*filmLoss));
                    }
                }
            }
            if (!finite(factor) || factor<0) return false;
        }
        // Actual geometric sweep is immutable during this evaluation. If a
        // donor limit would rescale it, the entire geometry/source trial must be
        // recomputed instead of silently changing only its diagnostic mass.
        if (in.useSweptGeometry && factor<1) {
            for (int species=0; species<Ns; ++species) if (trial.poreSweep[species]!=0) return false;
        }
        trial.phaseMass*=factor;
        trial.solidSpeed*=factor;
        if (!in.useSweptGeometry) trial.solidWallSpeed=trial.solidSpeed;
        Real J=0, filmLoss=0;
        Real allSpecies[Ns]{};
        for (int c=0; c<Nc; ++c) trial.condensed[c]*=factor;
        for (int s=0; s<Ns; ++s) {
            trial.gasSpecies[s]*=factor;
            trial.poreSpecies[s]*=factor;
            trial.poreDarcy[s]*=factor;
            trial.poreSweep[s]*=factor;
            allSpecies[s]=trial.gasSpecies[s]+trial.poreSpecies[s];
            J+=allSpecies[s]; filmLoss+=trial.gasSpecies[s];
        }
        const Real filmVolumeRate=result.wet
            ? (in.filmTransportMassRate+in.area*(trial.phaseMass-filmLoss))/p.liquid.rho : 0;
        trial.topSpeed=result.wet
            ? (in.solidArea*trial.solidSpeed+filmVolumeRate-in.sideMeshVolumeRate)/in.gasArea
            : trial.solidSpeed;
        trial.gasWallSpeed=in.useSweptGeometry ? in.gasSweptRate/in.gasArea : trial.topSpeed;
        trial.solidRecessionResidual=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate-in.solidArea*trial.solidSpeed : 0;
        trial.filmVolumeResidual=in.useSweptGeometry
            ? filmVolumeRate-(in.gasSweptRate-in.solidSweptRate
                +(result.wet ? in.sideMeshVolumeRate : 0)) : 0;
        for (int species=0; species<Ns; ++species) allSpecies[species]/=gasRatio;
        Real nextY[Ns]{};
        const Real fluxTolerance=in.dt>0
            ? p.tolerances.absoluteMass/(in.dt*in.gasArea) : p.tolerances.absoluteMass;
        if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
            Vec3 velocity=in.baseVelocity;
            if(result.wet){const Vec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
                const Real normalVelocity=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)-dot(tangent,in.gasNormal))/gasProjection;
                velocity=tangent+in.normal*normalVelocity;}
            if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,topT,in.gasDistance,velocity,in.gasNormal,
                trial.gasWallSpeed,allSpecies,fluxTolerance,p,in.wallContext,wallClosure))return false;
            for(int s=0;s<Ns;++s)nextY[s]=wallClosure.trace.Y[s];
        }else if (!interfaceComposition(allSpecies,in.bulk.Y,in.bulk.pressure,topT,in.gasDistance,
            p,fluxTolerance,nextY)) return false;
        Real change=0;
        for (int s=0; s<Ns; ++s) change=maxValue(change,absValue(nextY[s]-Y[s]));
        for (int s=0; s<Ns; ++s) Y[s]=0.5*(Y[s]+nextY[s]);
        result.gasState=provisional;
        for (int s=0; s<Ns; ++s) {
            result.gasSpecies[s]=trial.gasSpecies[s]; result.poreSpecies[s]=trial.poreSpecies[s];
            result.poreDarcy[s]=trial.poreDarcy[s]; result.poreSweep[s]=trial.poreSweep[s];
        }
        for (int c=0; c<Nc; ++c) result.condensed[c]=trial.condensed[c];
        result.phaseMass=trial.phaseMass;
        result.solidSpeed=trial.solidSpeed; result.topSpeed=trial.topSpeed;
        result.solidWallSpeed=trial.solidWallSpeed; result.gasWallSpeed=trial.gasWallSpeed;
        result.solidRecessionResidual=trial.solidRecessionResidual;
        result.filmVolumeResidual=trial.filmVolumeResidual;
        Vec3 liquidTrace=in.baseVelocity;
        if (result.wet) {
            const Vec3 tangent=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal);
            result.liquidTopNormal=(trial.gasWallSpeed+filmLoss/(gasRatio*p.liquid.rho)
                -dot(tangent,in.gasNormal))/gasProjection;
            liquidTrace=tangent+in.normal*result.liquidTopNormal;
        }
        if (!makeInterfacePrimitive(nextY,topT,in.bulk.pressure,liquidTrace,in.gasNormal,
            trial.gasWallSpeed,J/gasRatio,p,result.gasState)) return false;
        if (change<=p.tolerances.relativeMass+1e-12) { compositionConverged=true; break; }
    }
    if (!compositionConverged) return false;
    Vec3 liquidTop{};
    if (result.wet) {
        liquidTop=in.aux.topVelocity-in.normal*dot(in.aux.topVelocity,in.normal)
            +in.normal*result.liquidTopNormal;
    }
    GasGradient gradient=in.gradient;
    const Vec3 difference=in.bulk.velocity-result.gasState.velocity;
    gradient.velocity[0]+=in.gasNormal*(difference.x/in.gasDistance-dot(gradient.velocity[0],in.gasNormal));
    gradient.velocity[1]+=in.gasNormal*(difference.y/in.gasDistance-dot(gradient.velocity[1],in.gasNormal));
    gradient.velocity[2]+=in.gasNormal*(difference.z/in.gasDistance-dot(gradient.velocity[2],in.gasNormal));
    // Physical low-Re wall: molecular stress only, no owner nut or k stress.
    result.traction=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.traction:viscousTraction(gradient,in.gasNormal,p.gasViscosity,0);
    result.gasShear=result.traction-in.gasNormal*dot(result.traction,in.gasNormal);
    result.shear=(result.gasShear-in.normal*dot(result.gasShear,in.normal))*gasRatio;
    Real primaryJ=0, poreJ=0, poreDonorEnergy=0, solidDonorEnergy=0;
    for (int s=0; s<Ns; ++s) {
        primaryJ+=result.gasSpecies[s]; poreJ+=result.poreSpecies[s];
        result.gasAdvective+=result.gasSpecies[s]*speciesH(p.species[s],topT);
        result.poreEnergy+=result.poreSpecies[s]*speciesH(p.species[s],topT);
        const Real darcyT=result.poreDarcy[s]>=0 ? material.temperature : topT;
        const Real sweepT=result.poreSweep[s]>=0 ? material.temperature : topT;
        poreDonorEnergy+=result.poreDarcy[s]*speciesH(p.species[s],darcyT)
            +result.poreSweep[s]*speciesE(p.species[s],sweepT);
    }
    const Real gasKE=0.5*dot(result.gasState.velocity,result.gasState.velocity);
    result.gasAdvective+=primaryJ*gasKE;
    result.poreEnergy+=poreJ*gasKE;
    result.gasConductive=gasRatio*(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.conductiveHeatFlux:gasConductance*(topT-in.bulk.temperature));
    result.gasPressureWork=gasRatio*mechanicalPressure*result.gasWallSpeed;
    result.gasViscousWork=-gasRatio*dot(result.traction,result.gasState.velocity);
    result.gasEnergy=result.gasAdvective+result.gasConductive+result.gasPressureWork+result.gasViscousWork;
    const Real solidHeat=solidRatio*solidConductance*(material.temperature-bottomT);
    if (result.wet && in.hasSolid) {
        if (result.phaseMass!=0) {
            if (phase<0 || phase>=Nc || !validCondensed(p.condensed[phase])) return false;
            result.phaseAdvective=result.phaseMass*(condensedE(p.condensed[phase],bottomT)
                +in.bulk.pressure/p.condensed[phase].rho);
        }
        result.phaseConductive=solidHeat;
        result.phasePressureWork=solidRatio*in.bulk.pressure*result.solidWallSpeed;
        result.phaseViscousWork=0; // stationary rigid skeleton.
        result.phaseEnergy=result.phaseAdvective+result.phaseConductive+result.phasePressureWork;
        const Vec3 baseTangent=in.baseVelocity-in.normal*dot(in.baseVelocity,in.normal);
        result.liquidBottomNormal=(result.solidWallSpeed+result.phaseMass/(solidRatio*p.liquid.rho)
            -dot(baseTangent,in.solidNormal))/solidProjection;
        const Vec3 liquidBottom=baseTangent+in.normal*result.liquidBottomNormal;
        result.bottomKineticAdvection=-0.5*result.phaseMass*dot(liquidBottom,liquidBottom);
    }
    if (result.wet) {
        result.topKineticAdvection=0.5*primaryJ*dot(liquidTop,liquidTop);
        const Real liquidOut=gasRatio*physicalInterfaceEnergy(primaryJ/gasRatio,
            primaryJ/gasRatio*liquidH(p,topT,in.bulk.pressure),liquidTop,mechanicalPressure,
            result.gasWallSpeed,dot(result.gasShear,liquidTop),
            gasProjection*liquidConductance*(filmT-topT));
        result.thermalResidual=birth ? 0 : result.gasEnergy-liquidOut-result.radiationFlux;
    } else {
        for (int c=0; c<Nc; ++c) if (result.condensed[c]!=0) {
            solidDonorEnergy+=result.condensed[c]*condensedE(p.condensed[c],topT);
        }
        result.thermalResidual=result.gasEnergy+result.poreEnergy-solidDonorEnergy-poreDonorEnergy
            -solidRatio*solidConductance*(material.temperature-topT)-result.radiationFlux;
    }
    if (!finite(result.thermalResidual) || !finite(result.phaseEnergy)) return false;
    out=result; return true;
}
// Never accept a root by interpolating through an inadmissible composition or
// kinetic state. A failed midpoint abandons this bracket; bounded domain search
// below may find a separate connected feasible interval.
CHMT_HD inline bool bisectFeasibleSurface(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const MaterialPrimitive& material, Real filmT, Real lo, Real hi,
    SurfacePhysicsResult left, SurfacePhysicsResult right, SurfacePhysicsResult& out) {
    if ((left.thermalResidual>0)==(right.thermalResidual>0)
        && left.thermalResidual!=0 && right.thermalResidual!=0) return false;
    if (left.thermalResidual==0) { out=left; return true; }
    if (right.thermalResidual==0) { out=right; return true; }
    for (int iteration=0; iteration<100; ++iteration) {
        const Real T=0.5*(lo+hi);
        SurfacePhysicsResult candidate;
        if (!evaluateSurfaceTemperature(in,p,material,filmT,T,false,candidate)) return false;
        const Real scale=absValue(candidate.gasEnergy)+absValue(candidate.poreEnergy)
            +absValue(candidate.phaseEnergy)+absValue(candidate.radiationFlux);
        const Real fluxTolerance=p.tolerances.absoluteEnergy/(in.area*(in.dt>0 ? in.dt : 1))
            +p.tolerances.relativeEnergy*scale;
        const Real temperatureTolerance=p.tolerances.absoluteTemperature+p.tolerances.relativeTemperature*T;
        if (absValue(candidate.thermalResidual)<=fluxTolerance
            && (hi-lo<=temperatureTolerance || candidate.thermalResidual==0)) {
            out=candidate; return true;
        }
        if (T==lo || T==hi) return false;
        if ((candidate.thermalResidual>0)==(left.thermalResidual>0)) { lo=T; left=candidate; }
        else { hi=T; right=candidate; }
    }
    return false;
}
CHMT_HD inline bool findFeasibleSurfaceRoot(const SurfacePhysicsInput& in, const PhysicsConfig& p,
    const MaterialPrimitive& material, Real filmT, Real lo, Real hi, SurfacePhysicsResult& out) {
    if (!(hi>lo)) return false;
    const Real preferred=maxValue(lo,minValue(hi,in.hasFilm && in.film.mass>0 ? filmT : material.temperature));
    SurfacePhysicsResult seed;
    if (evaluateSurfaceTemperature(in,p,material,filmT,preferred,false,seed)
        && seed.thermalResidual==0) { out=seed; return true; }
    // Refine only if coarser feasible components contain no solvable bracket.
    // Infeasible nodes reset continuation; no state or packet is published here.
    for (int subdivisions=32; subdivisions<=512; subdivisions*=4) {
        SurfacePhysicsResult previous;
        Real previousT=lo;
        bool previousValid=false;
        for (int i=0; i<=subdivisions; ++i) {
            const Real gridT=lo+(hi-lo)*static_cast<Real>(i)/subdivisions;
            const bool insert=i>0 && preferred>lo+(hi-lo)*(i-1)/subdivisions && preferred<gridT;
            for (int part=0; part<(insert ? 2 : 1); ++part) {
                const Real T=insert && part==0 ? preferred : gridT;
                SurfacePhysicsResult current;
                const bool valid=evaluateSurfaceTemperature(in,p,material,filmT,T,false,current);
                if (valid && current.thermalResidual==0) { out=current; return true; }
                if (valid && previousValid
                    && (current.thermalResidual>0)!=(previous.thermalResidual>0)) {
                    SurfacePhysicsResult root;
                    if (bisectFeasibleSurface(in,p,material,filmT,previousT,T,previous,current,root)) {
                        out=root; return true;
                    }
                }
                previousValid=valid;
                if (valid) { previous=current; previousT=T; }
            }
        }
    }
    return false;
}
CHMT_HD inline bool solveSurfaceInterface(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, SurfacePhysicsResult& out, DeviceStatus* status=nullptr) {
    if (!(in.area>0 && in.gasArea>0 && in.gasDistance>0) || !finite(in.normal) || !finite(in.gasNormal)
        || !closeEnough(mag(in.normal),1,1e-12,1e-12)
        || !closeEnough(mag(in.gasNormal),1,1e-12,1e-12) || dot(in.normal,in.gasNormal)<=1e-8
        || (in.hasSolid && (mag(in.baseVelocity)!=0 || in.solidDistance<=0 || in.solidArea<=0
            || !finite(in.solidNormal) || !closeEnough(mag(in.solidNormal),1,1e-12,1e-12)
            || dot(in.normal,in.solidNormal)<=1e-8))
        || (in.useSweptGeometry && (!finite(in.gasSweptRate) || !finite(in.solidSweptRate)))) return false;
    MaterialPrimitive material;
    if (in.hasSolid && !recoverMaterial(in.solid,in.solidVolume,p,material)) return false;
    FilmAux filmAux=in.aux;
    const bool wet=in.hasFilm && in.film.mass>0;
    if (wet && !recoverFilm(in.film,in.area,in.aux.pressure,p,filmAux)) return false;
    if (!wet && !in.hasSolid) return false;
    if (!wet && (!closeEnough(in.gasArea,in.solidArea,p.tolerances.absoluteGeometry,
        p.tolerances.relativeGeometry) || mag(in.gasNormal-in.solidNormal)>1e-10)) return false;
    if (wet && p.material.enableSurfaceReactions) return false;
    // Pore collapse/bubbling into a film needs an additional mechanical model.
    // The first Stefan phase interface is a dense material phase boundary.
    if (in.hasSolid && p.material.enableMelting && material.porosity>p.tolerances.relativeGeometry) return false;
    Real lo=0, hi=std::numeric_limits<Real>::max();
    for (int s=0; s<Ns; ++s) { lo=maxValue(lo,p.species[s].Tmin); hi=minValue(hi,p.species[s].Tmax); }
    if (wet) { lo=maxValue(lo,p.liquid.Tmin); hi=minValue(hi,p.liquid.Tmax); }
    if (in.hasSolid) for (int c=0; c<Nc; ++c) if (in.solid.condensed[c]>0) {
        lo=maxValue(lo,p.condensed[c].Tmin); hi=minValue(hi,p.condensed[c].Tmax);
    }
    const Real filmT=wet ? filmAux.temperature : p.meltTemperature;
    SurfacePhysicsResult candidate;
    if (!findFeasibleSurfaceRoot(in,p,material,filmT,lo,hi,candidate)) return false;
    if (!wet && in.hasFilm && p.material.enableMelting && !in.useAcceptedFlux
        && candidate.topTemperature>p.meltTemperature) {
        if (!evaluateSurfaceTemperature(in,p,material,p.meltTemperature,p.meltTemperature,true,candidate)) return false;
    }
    const bool offsetGeometry=!closeEnough(in.gasArea,in.area,0,p.tolerances.relativeGeometry)
        || mag(in.gasNormal-in.normal)>1e-10
        || (in.hasSolid && (!closeEnough(in.solidArea,in.area,0,p.tolerances.relativeGeometry)
            || mag(in.solidNormal-in.normal)>1e-10));
    const Real rateTolerance=p.tolerances.absoluteGeometry/(in.dt>0 ? in.dt : 1);
    if (offsetGeometry && (absValue(candidate.topSpeed)*in.gasArea>rateTolerance
        || absValue(candidate.solidSpeed)*in.solidArea>rateTolerance
        || (in.useSweptGeometry && (absValue(in.gasSweptRate)>rateTolerance
            || absValue(in.solidSweptRate)>rateTolerance)))) {
        return failStatus(status,ErrorCode::Unsupported,-1,candidate.filmVolumeResidual,ErrorLocation::Face);
    }
    out=candidate; return true;
}
struct SurfacePacketIdentity {
    std::uint64_t step=0, geometry=0, face=0;
    int stage=0, gasCell=-1, solidCell=-1, filmFace=-1;
    Real oldFilmPV=0;
    Vec3 filmPressureGradient{};
};
CHMT_HD inline ExchangePacket surfacePacketIdentity(ExchangeKind kind, const SurfacePacketIdentity& id) {
    ExchangePacket packet;
    packet.kind=kind; packet.step=id.step; packet.geometry=id.geometry;
    packet.face=id.face; packet.stage=id.stage;
    if (requiredConsumers(kind)&ConsumeGas) packet.gasCell=id.gasCell;
    if (requiredConsumers(kind)&ConsumeSolid) packet.solidCell=id.solidCell;
    if (requiredConsumers(kind)&ConsumeFilm) packet.filmFace=id.filmFace;
    return packet;
}
CHMT_HD inline bool assembleSurfacePackets(const SurfacePhysicsInput& input,
    const SurfacePhysicsResult& result, const SurfacePacketIdentity& identity,
    const PhysicsConfig& p, ExchangePacket* output, DeviceStatus* status=nullptr) {
    const int face=identity.filmFace;
    const Real dt=input.dt;
    const Real scale=input.area*dt;
    const ExchangeKind primaryKind=result.wet ? ExchangeKind::GasFilm : ExchangeKind::GasSolid;
    ExchangePacket primary=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket pore=surfacePacketIdentity(primaryKind,identity);
    ExchangePacket phase=surfacePacketIdentity(primaryKind,identity);
    for (int s=0; s<Ns; ++s) {
        primary.species[s]=scale*result.gasSpecies[s];
        primary.mass+=primary.species[s];
    }
    for (int c=0; c<Nc; ++c) primary.condensed[c]=scale*result.condensed[c];
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?input.wallContext.mechanicalPressure:result.gasState.pressure;
    if(!finite(mechanicalPressure)||mechanicalPressure<=0)return false;
    primary.momentum=result.gasState.velocity*primary.mass
        +(input.gasNormal*mechanicalPressure-result.traction)*(input.gasArea*dt);
    primary.advective=scale*result.gasAdvective;
    primary.conductive=scale*result.gasConductive;
    primary.pressureWork=scale*result.gasPressureWork;
    primary.viscousWork=scale*result.gasViscousWork;
    primary.energy=primary.advective+primary.conductive+primary.pressureWork+primary.viscousWork;
    primary.liquidKineticAdvection=result.wet ? scale*result.topKineticAdvection : 0;
    if (input.hasSolid) {
        pore=surfacePacketIdentity(ExchangeKind::PoreGas,identity);
        for (int s=0; s<Ns; ++s) {
            pore.species[s]=pore.pore[s]=scale*result.poreSpecies[s];
            pore.poreSweep[s]=scale*result.poreSweep[s];
            pore.mass+=pore.species[s];
        }
        pore.momentum=result.gasState.velocity*pore.mass;
        pore.energy=pore.advective=scale*result.poreEnergy;
    }
    if (input.hasSolid && result.wet) {
        phase=surfacePacketIdentity(ExchangeKind::SolidFilm,identity);
        phase.mass=scale*result.phaseMass;
        if (phase.mass!=0) phase.condensed[p.material.phaseCondensed]=phase.mass;
        for (int s=0; s<Ns; ++s) phase.species[s]=phase.mass*p.material.phaseFilmY[s];
        phase.advective=scale*result.phaseAdvective;
        phase.conductive=scale*result.phaseConductive;
        phase.pressureWork=scale*result.phasePressureWork;
        phase.viscousWork=scale*result.phaseViscousWork;
        phase.energy=phase.advective+phase.conductive+phase.pressureWork+phase.viscousWork;
        phase.liquidKineticAdvection=scale*result.bottomKineticAdvection;
        Vec3 G=identity.filmPressureGradient;
        G-=(p.gravity-input.normal*dot(p.gravity,input.normal))*p.liquid.rho;
        const FilmProfile profile=filmProfile(input.aux.thickness,p.liquidViscosity,
            input.baseVelocity,result.shear,G);
        const Vec3 bottomShear=profile.bottomShear
            -input.solidNormal*dot(profile.bottomShear,input.solidNormal);
        phase.momentum=input.solidNormal*(result.gasState.pressure*input.solidArea*dt)-bottomShear*scale;
        // Exact event transfer: the disappearing film hands its actual remaining
        // internal energy to the solid. This is not a correction for R_K.
        FilmQ available=input.filmBase;
        available.mass+=dt*input.filmLocalRate.mass-primary.mass;
        for (int s=0; s<Ns; ++s) {
            available.species[s]+=dt*input.filmLocalRate.species[s]-primary.species[s];
        }
        const Real Uavailable=input.filmBase.enthalpy-identity.oldFilmPV
            +dt*input.filmLocalRate.enthalpy-primary.energy+primary.liquidKineticAdvection
            +scale*result.radiationFlux;
        if (phase.mass<0 && -phase.mass>=available.mass && available.mass>=0) {
            for (int s=0; s<Ns; ++s) {
                if (!closeEnough(available.species[s],available.mass*p.material.phaseFilmY[s],
                    p.tolerances.absoluteMass,p.tolerances.relativeMass)) {
                    return failStatus(status,ErrorCode::Composition,face,available.species[s],ErrorLocation::Face);
                }
            }
            phase.mass=-available.mass;
            for (int c=0; c<Nc; ++c) phase.condensed[c]=c==p.material.phaseCondensed ? phase.mass : 0;
            for (int s=0; s<Ns; ++s) phase.species[s]=-available.species[s];
            phase.energy=-Uavailable-phase.liquidKineticAdvection;
            phase.advective=phase.energy-phase.conductive-phase.pressureWork-phase.viscousWork;
        } else if (available.mass+phase.mass==0) {
            phase.energy=-Uavailable-phase.liquidKineticAdvection;
            phase.advective=phase.energy-phase.conductive-phase.pressureWork-phase.viscousWork;
        }
    }
    DeviceStatus local;
    if (!validatePacketMath(primary,p.tolerances,&local)
        || !validatePacketMath(pore,p.tolerances,&local) || !validatePacketMath(phase,p.tolerances,&local)) {
        return failStatus(status,ErrorCode::Packet,face,local.value,ErrorLocation::Face);
    }
    output[0]=primary; output[1]=pore; output[2]=phase;
    return true;
}
CHMT_HD inline bool surfaceDonorCandidate(const SolidQ& base, const SolidQ& localRate,
    Real dt, const ExchangePacket* packets, int count, int cell, SolidQ& out) {
    if (!finite(dt) || dt<=0 || count<0 || (count>0 && !packets)) return false;
    SolidQ candidate=addSolid(base,localRate,dt);
    for (int index=0; index<count; ++index) {
        const ExchangePacket& packet=packets[index];
        if ((requiredConsumers(packet.kind)&ConsumeSolid) && packet.solidCell==cell) {
            candidate=addSolid(candidate,packetDelta(packet).solid);
        }
    }
    if (!validSolidInventory(candidate)) return false;
    out=candidate; return true;
}
} // namespace chmt
#endif
