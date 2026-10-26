#ifndef CHMT_ABLATION_DRYSURFACEINTERFACEMATH_H
#define CHMT_ABLATION_DRYSURFACEINTERFACEMATH_H
#include "ablation/InterfaceMath.H"
#include "coupling/IntervalContracts.H"
#include "materials/dry/DrySurfaceResponse.H"
#include "materials/DryMaterialAdapter.H"
#include "materials/dry/PoreTransport.H"
namespace chmt {
// Shared material adapter: proposals retain WallFaceSample reference-area
// units. The validated dry kernels always see current physical material area.
// No accepted gas ExchangePacket is constructed, changed, or applied here.
CHMT_HD inline dry::SurfaceProduction<Nc,Ns> drySurfaceProduction(const SurfacePhysicsInput& in,
    const GasPrimitive& gas,Real T,const PhysicsConfig& p) {
    std::array<dry::SurfaceMassAction<Nc,Ns>,Nr> reactions{};
    std::array<double,Nc> condensed{};std::array<double,Ns> partial{};
    for(int c=0;c<Nc;++c)condensed[c]=in.solid.condensed[c]/in.solidVolume;
    for(int species=0;species<Ns;++species)partial[species]=gas.rho*gas.Y[species];
    for(int r=0;r<p.material.nSurfaceReactions;++r){
        const auto& old=p.material.surfaceReactions[r];auto& rate=reactions[r];
        rate.prefactor=old.A;rate.temperaturePower=old.temperaturePower;rate.activationEnergy=old.activationEnergy;
        for(int c=0;c<Nc;++c){rate.condensedOrder[c]=old.order[c];rate.stoichiometry.condensed[c]=old.condensedNu[c];}
        for(int species=0;species<Ns;++species){rate.gasOrder[species]=p.material.surfaceGasOrder[r][species];rate.stoichiometry.gas[species]=old.gasNu[species];}
    }
    return dry::surfaceProduction(reactions,T,condensed,partial);
}
// Sharp recession represents one exposed effective phase or a congruent mixture.
// Bulk pyrolysis may contain independent condensed constituents; it does not by
// itself imply their selective surface loss is a closed sharp-front model.
CHMT_HD inline bool evaluateDrySurfaceTemperature(const SurfacePhysicsInput& in,
    const PhysicsConfig& p, const MaterialPrimitive& material, Real filmT, Real topT,
    bool birth, SurfacePhysicsResult& out,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if((p.wallModel.family!=ugkwp::gaswall::WallFamily::LowRe&&p.wallModel.family!=ugkwp::gaswall::WallFamily::BoundaryLayer))return false;
    const Real mechanicalPressure=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?in.wallContext.mechanicalPressure:in.bulk.pressure;
    if((!finite(mechanicalPressure)||mechanicalPressure<=0))return surfacePressureFailure(p,mechanicalPressure);
    SurfacePhysicsResult result;
    (void)filmT; (void)birth;
    const Real gasRatio=in.gasArea/in.area;
    const Real solidRatio=in.hasSolid ? in.solidArea/in.area : 0;
    result.wet=false;
    result.topTemperature=topT;
    const Real gasConductance=p.gasConductivity>0
        ? 1/(in.gasDistance/p.gasConductivity+p.material.gasContactResistance) : 0;
    const Real solidConductance=in.hasSolid && material.conductivity>0
        ? 1/(in.solidDistance/material.conductivity+p.material.solidContactResistance) : 0;
    result.radiationFlux=gasRatio*grayRadiation(topT,p);
    result.bottomTemperature=topT;
    Real Y[Ns]{};
    for (int s=0; s<Ns; ++s) Y[s]=in.bulk.Y[s];
    bool compositionConverged=false;Real compositionChange=0;
    InterfaceGasClosure wallClosure;
    Real physicalPoreEnergy[Ns]{};
    // This cache lives only inside this fixed-temperature evaluation. The bulk,
    // model, geometry and matching state cannot change during this loop.
    bool wallClosureReady=false;
    Real wallClosureSpecies[Ns]{},wallClosureSpeed=0;
    Vec3 wallClosureVelocity{};
    for (int iteration=0; iteration<60; ++iteration) {
        SurfacePhysicsResult trial;
        Real trialPoreEnergy[Ns]{};
        trial.wet=result.wet;
        GasPrimitive provisional;
        if (!makeInterfacePrimitive(Y,topT,in.bulk.pressure,{},in.gasNormal,0,0,p,provisional,numericalFailure)) return false;
        if (in.useAcceptedFlux) {
            for (int s=0; s<Ns; ++s) {
                trial.gasSpecies[s]=in.acceptedGasSpecies[s];
                trial.poreSpecies[s]=in.acceptedPoreSpecies[s];
                trial.poreSweep[s]=in.acceptedPoreSweep[s];
                trial.poreDarcy[s]=trial.poreSpecies[s]-trial.poreSweep[s];
                trialPoreEnergy[s]=trial.poreDarcy[s]*speciesH(p.species[s],trial.poreDarcy[s]>=0?material.temperature:topT);
            }
            for (int c=0; c<Nc; ++c) trial.condensed[c]=in.acceptedCondensed[c];
        } else {
            if (p.material.enableSurfaceReactions) {
                if (!in.hasSolid) {if(numericalFailure)*numericalFailure=false;return false;}
                const auto production=drySurfaceProduction(in,provisional,topT,p);
                // Core uses CURRENT physical material area. Existing wall proposals
                // use reference area; convert exactly here, never in the gas ledger.
                for(int c=0;c<Nc;++c)trial.condensed[c]=-solidRatio*production.condensed[c];
                for(int species=0;species<Ns;++species)trial.gasSpecies[species]=solidRatio*production.gas[species];
            }
        }
        if(in.hasSolid){
            dry::SurfaceProduction<Nc,Ns> geometryProduction;
            for(int c=0;c<Nc;++c)geometryProduction.condensed[c]=-trial.condensed[c]/solidRatio;
            const dry::PoreSurfaceFlux<Ns> noPore;
            const auto geometry=dry::congruentDryInterfaceFlux(geometryProduction,noPore,
                dryAdapter::condensed(p),material.porosity);
            trial.solidSpeed=geometry.normalInterfaceSpeed;
        }
        trial.solidWallSpeed=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate/in.solidArea : trial.solidSpeed;
        if (!in.useAcceptedFlux && in.hasSolid) {
            Real superficial=0;
            if (p.material.enablePoreOutflow && p.permeability>0) {
                superficial=p.permeability/p.poreViscosity
                    *((material.pressure-in.bulk.pressure)/in.solidDistance
                        +material.poreDensity*dot(p.gravity,in.solidNormal));
            }
            auto inside=dryAdapter::poreState(in.solid,material);
            dry::PoreFaceState<Ns> outside;
            outside.density=provisional.rho;outside.temperature=topT;
            std::array<double,Ns> effectiveDiffusivity{};
            for(int species=0;species<Ns;++species){
                outside.massFraction[species]=Y[species];
                effectiveDiffusivity[species]=p.poreEffectiveDiffusivity[species];
            }
            dryAdapter::completePorePair(inside,outside);
            const auto physical=dry::poreFaceFlux(inside,outside,in.solidDistance,
                superficial,effectiveDiffusivity,0,dryAdapter::GasCalorics{p});
            // Recession exposes pore inventory even with zero permeability.
            const Real poreSweep=-material.porosity*trial.solidWallSpeed;
            for (int species=0; species<Ns; ++species) {
                const Real poreDensity=material.porosity>0
                    ? in.solid.pore[species]/(material.porosity*in.solidVolume) : 0;
                // Legacy name: poreDarcy now includes physical diffusion too.
                trial.poreDarcy[species]=solidRatio*physical.species[species];
                trialPoreEnergy[species]=solidRatio*physical.speciesEnergy[species];
                trial.poreSweep[species]=solidRatio*(poreSweep>=0 ? poreSweep*poreDensity
                    : poreSweep*provisional.rho*Y[species]);
                trial.poreSpecies[species]=trial.poreDarcy[species]+trial.poreSweep[species];
            }
        }
        Real factor=1;
        if (!in.useAcceptedFlux && in.dt>0) {
            const Real scale=in.area*in.dt;
            for (int c=0; c<Nc; ++c) {
                const Real loss=trial.condensed[c];
                if (loss>0) {
                    const Real available=in.solidBase.condensed[c]+in.dt*in.solidLocalRate.condensed[c];
                    factor=minValue(factor,surfaceDonorFraction(available,scale,loss));
                }
            }
            for (int s=0; s<Ns; ++s) {
                if (trial.poreSpecies[s]>0) {
                    const Real available=in.solidBase.pore[s]+in.dt*in.solidLocalRate.pore[s];
                    factor=minValue(factor,surfaceDonorFraction(available,scale,trial.poreSpecies[s]));
                }
                const Real gasLoss=-trial.gasSpecies[s]-trial.poreSpecies[s];
                if (gasLoss>0) factor=minValue(factor,surfaceDonorFraction(in.gasInventory.species[s],scale,gasLoss));
            }
            if ((!finite(factor) || factor<0)) return false;
        }
        // Actual geometric sweep is immutable during this evaluation. If a
        // donor limit would rescale it, the entire geometry/source trial must be
        // recomputed instead of silently changing only its diagnostic mass.
        if (in.useSweptGeometry && factor<1) {
            for (int species=0; species<Ns; ++species) if ((trial.poreSweep[species]!=0)) return false;
        }
        trial.solidSpeed*=factor;
        if (!in.useSweptGeometry) trial.solidWallSpeed=trial.solidSpeed;
        Real J=0;
        Real allSpecies[Ns]{};
        for (int c=0; c<Nc; ++c) trial.condensed[c]*=factor;
        for (int s=0; s<Ns; ++s) {
            trial.gasSpecies[s]*=factor;
            trial.poreSpecies[s]*=factor;
            trial.poreDarcy[s]*=factor;
            trial.poreSweep[s]*=factor;
            trialPoreEnergy[s]*=factor;
            allSpecies[s]=trial.gasSpecies[s]+trial.poreSpecies[s];
            J+=allSpecies[s];
        }
        trial.topSpeed=trial.solidSpeed;
        trial.gasWallSpeed=in.useSweptGeometry ? in.gasSweptRate/in.gasArea : trial.topSpeed;
        trial.solidRecessionResidual=in.hasSolid && in.useSweptGeometry
            ? in.solidSweptRate-in.solidArea*trial.solidSpeed : 0;
        for (int species=0; species<Ns; ++species) allSpecies[species]/=gasRatio;
        Real nextY[Ns]{};
        const Real fluxTolerance=in.dt>0
            ? p.tolerances.absoluteMass/(in.dt*in.gasArea) : p.tolerances.absoluteMass;
        if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
            Vec3 velocity=in.baseVelocity;
            bool sameClosure=wallClosureReady && trial.gasWallSpeed==wallClosureSpeed
                && velocity.x==wallClosureVelocity.x && velocity.y==wallClosureVelocity.y
                && velocity.z==wallClosureVelocity.z;
            for(int s=0;s<Ns;++s) sameClosure=sameClosure && allSpecies[s]==wallClosureSpecies[s];
            if(!sameClosure){
                if(!evaluateInterfaceGasClosure(in.bulk,in.gradient,topT,in.gasDistance,velocity,in.gasNormal,
                    trial.gasWallSpeed,allSpecies,fluxTolerance,p,in.wallContext,wallClosure,numericalFailure))return false;
                wallClosureReady=true;wallClosureSpeed=trial.gasWallSpeed;wallClosureVelocity=velocity;
                for(int s=0;s<Ns;++s)wallClosureSpecies[s]=allSpecies[s];
            }
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
            physicalPoreEnergy[s]=trialPoreEnergy[s];
        }
        for (int c=0; c<Nc; ++c) result.condensed[c]=trial.condensed[c];
        result.solidSpeed=trial.solidSpeed; result.topSpeed=trial.topSpeed;
        result.solidWallSpeed=trial.solidWallSpeed; result.gasWallSpeed=trial.gasWallSpeed;
        result.solidRecessionResidual=trial.solidRecessionResidual;
        if (!makeInterfacePrimitive(nextY,topT,in.bulk.pressure,in.baseVelocity,in.gasNormal,
            trial.gasWallSpeed,J/gasRatio,p,result.gasState,numericalFailure)) return false;
        compositionChange=change;
        if (change<=p.tolerances.relativeMass+1e-12) { compositionConverged=true; break; }
    }
    if (!compositionConverged) surfaceCompositionWarning(p,compositionChange,p.tolerances.relativeMass+1e-12);
    GasGradient gradient=in.gradient;
    const Vec3 difference=in.bulk.velocity-result.gasState.velocity;
    gradient.velocity[0]+=in.gasNormal*(difference.x/in.gasDistance-dot(gradient.velocity[0],in.gasNormal));
    gradient.velocity[1]+=in.gasNormal*(difference.y/in.gasDistance-dot(gradient.velocity[1],in.gasNormal));
    gradient.velocity[2]+=in.gasNormal*(difference.z/in.gasDistance-dot(gradient.velocity[2],in.gasNormal));
    // Physical low-Re wall: molecular stress only, no owner nut or k stress.
    result.traction=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer
        ?wallClosure.traction:viscousTraction(gradient,in.gasNormal,gasMolecularViscosity(p,result.gasState.temperature,result.gasState.Y),0);
    result.gasShear=result.traction-in.gasNormal*dot(result.traction,in.gasNormal);
    result.shear=(result.gasShear-in.normal*dot(result.gasShear,in.normal))*gasRatio;
    Real primaryJ=0, poreJ=0, poreDonorEnergy=0, solidDonorEnergy=0;
    for (int s=0; s<Ns; ++s) {
        primaryJ+=result.gasSpecies[s]; poreJ+=result.poreSpecies[s];
        result.gasAdvective+=result.gasSpecies[s]*speciesH(p.species[s],topT);
        result.poreEnergy+=result.poreSpecies[s]*speciesH(p.species[s],topT);
        const Real sweepT=result.poreSweep[s]>=0 ? material.temperature : topT;
        poreDonorEnergy+=physicalPoreEnergy[s]
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
    for (int c=0; c<Nc; ++c) if (result.condensed[c]!=0)
        solidDonorEnergy+=result.condensed[c]*condensedE(p.condensed[c],topT);
    // The accepted gas flux remains authoritative. This residual only closes a
    // constitutive proposal; swept pore inventory carries e, Darcy transport h.
    result.thermalResidual=result.gasEnergy+result.poreEnergy-solidDonorEnergy-poreDonorEnergy
        -solidRatio*solidConductance*(material.temperature-topT)-result.radiationFlux;
    if ((!finite(result.thermalResidual) || !finite(result.phaseEnergy))) return surfaceEnergyFailure(p,result.thermalResidual,result.phaseEnergy);
    out=result; return true;
}
struct DrySurfaceEvaluation {
    CHMT_HD static bool evaluate(const SurfacePhysicsInput& in,const PhysicsConfig& p,
        const MaterialPrimitive& material,Real filmT,Real topT,bool birth,
        SurfacePhysicsResult& out,bool* numericalFailure) {
        return evaluateDrySurfaceTemperature(in,p,material,filmT,topT,birth,out,numericalFailure);
    }
};
CHMT_HD inline bool solveDrySurfaceInterface(const SurfacePhysicsInput& in,const PhysicsConfig& p,
    SurfacePhysicsResult& out,DeviceStatus* status=nullptr,bool* numericalFailure=nullptr) {
    if(numericalFailure)*numericalFailure=true;
    if ((!(in.area>0 && in.gasArea>0 && in.gasDistance>0) || !finite(in.normal) || !finite(in.gasNormal)
        || mag(in.normal)==0
        || mag(in.gasNormal)==0 || dot(in.normal,in.gasNormal)<=0
        || (in.hasSolid && (mag(in.baseVelocity)!=0 || in.solidDistance<=0 || in.solidArea<=0
            || !finite(in.solidNormal) || mag(in.solidNormal)==0
            || dot(in.normal,in.solidNormal)<=0))
        || (in.useSweptGeometry && (!finite(in.gasSweptRate) || !finite(in.solidSweptRate))))) return surfaceGeometryFailure(p,in.area,in.gasDistance);
    const Real normalResidual=maxValue(absValue(mag(in.normal)-1),maxValue(absValue(mag(in.gasNormal)-1),in.hasSolid?absValue(mag(in.solidNormal)-1):0));
    if(normalResidual>1e-12)surfaceNormalWarning(p,normalResidual);
    MaterialPrimitive material;
    if (in.hasSolid && !recoverMaterial(in.solid,in.solidVolume,p,material,nullptr,-1,numericalFailure)) return false;
    if(!in.hasSolid){if(numericalFailure)*numericalFailure=false;return false;}
    if(!closeEnough(in.gasArea,in.solidArea,p.tolerances.absoluteGeometry,
        p.tolerances.relativeGeometry)||mag(in.gasNormal-in.solidNormal)>1e-10)
        surfaceAreaWarning(p,in.gasArea-in.solidArea,mag(in.gasNormal-in.solidNormal));
    Real lo=0,hi=std::numeric_limits<Real>::max();
    for(int species=0;species<Ns;++species){lo=maxValue(lo,p.species[species].Tmin);hi=minValue(hi,p.species[species].Tmax);}
    for(int c=0;c<Nc;++c)if(in.solid.condensed[c]>0){lo=maxValue(lo,p.condensed[c].Tmin);hi=minValue(hi,p.condensed[c].Tmax);}
    SurfacePhysicsResult candidate;
    if(!findFeasibleSurfaceRoot<false,DrySurfaceEvaluation>(in,p,material,material.temperature,
        lo,hi,candidate,nullptr,numericalFailure))return false;
    if(!evaluateDrySurfaceTemperature(in,p,material,material.temperature,
        candidate.topTemperature,false,candidate,numericalFailure))return false;
    // A common current-area/reference-area scale is an exact unit conversion.
    // Other moving offsets retain the existing unsupported-geometry behavior;
    // no tilted/nonconformal moving-front extension is claimed by this adapter.
    const bool scalarAreaNormalization=closeEnough(in.gasArea,in.solidArea,0,p.tolerances.relativeGeometry)
        &&mag(in.gasNormal-in.normal)<=1e-10&&mag(in.solidNormal-in.normal)<=1e-10;
    const bool offsetGeometry=!closeEnough(in.gasArea,in.area,0,p.tolerances.relativeGeometry)
        ||mag(in.gasNormal-in.normal)>1e-10
        ||!closeEnough(in.solidArea,in.area,0,p.tolerances.relativeGeometry)
        ||mag(in.solidNormal-in.normal)>1e-10;
    const Real rateTolerance=p.tolerances.absoluteGeometry/(in.dt>0?in.dt:1);
    if(offsetGeometry&&!scalarAreaNormalization
        &&(absValue(candidate.topSpeed)*in.gasArea>rateTolerance
            ||absValue(candidate.solidSpeed)*in.solidArea>rateTolerance
            ||(in.useSweptGeometry&&(absValue(in.gasSweptRate)>rateTolerance
                ||absValue(in.solidSweptRate)>rateTolerance))))
        return failStatus(status,ErrorCode::Unsupported,-1,candidate.filmVolumeResidual,ErrorLocation::Face);
    out=candidate;return true;
}
// The proposal adapter uses the same reference-area units on CPU and GPU.
CHMT_HD inline WallFaceSample dryWallFaceSample(const SurfacePhysicsInput& input,
    const SurfacePhysicsResult& physics) {
    WallFaceSample wall;wall.temperature=physics.topTemperature;wall.primaryKind=ExchangeKind::GasSolid;
    wall.velocity=physics.gasState.velocity-input.gasNormal*dot(physics.gasState.velocity,input.gasNormal);
    for(int s=0;s<Ns;++s){wall.speciesRate[s]=physics.gasSpecies[s];wall.poreRate[s]=physics.poreSpecies[s];wall.poreSweepRate[s]=physics.poreSweep[s];}
    for(int c=0;c<Nc;++c)wall.condensedRate[c]=physics.condensed[c];
    wall.radiationFlux=physics.radiationFlux;wall.solidNormalVelocity=physics.solidSpeed;wall.normalVelocity=physics.topSpeed;
    return wall;
}

}
#endif
