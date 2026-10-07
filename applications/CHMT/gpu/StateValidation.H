#ifndef CHMT_GPU_STATEVALIDATION_H
#define CHMT_GPU_STATEVALIDATION_H
#include "core/HostState.H"
#include "core/ThermalBoundary.H"
#include "coupling/Coordinator.H"
#include "materials/Reaction.H"
#include "film/ResolvedNormalThermal.H"
#include "particles/ParticleMath.H"
#include "mesh/Geometry.H"
#include "mesh/SweepConstraints.H"
#include <climits>
#include <set>
namespace chmt {
inline bool validateRuntimeState(const ModelConfig& model,HostState& h,std::string& error){
    const auto& p=model.physics;
    auto fail=[&](const std::string& message){error=message;return false;};
    if(!validateSpeciesSet(model,error))return false;
    if(!validateThermalBoundaries(h.gasMesh,error)||!validateThermalBoundaries(h.solidMesh,error))return false;
    if(h.gas.size()>INT_MAX||h.solid.size()>INT_MAX||h.film.size()>INT_MAX||h.particles.size()>INT_MAX
        ||h.surface.area.size()>INT_MAX/3||h.gasMesh.owner.size()>INT_MAX||h.solidMesh.owner.size()>INT_MAX)return fail("state exceeds compiled index range");
    const bool normal=p.filmThermalMode==FilmThermalMode::ResolvedNormal;
    ValidationContext context;context.gasThermo=p.enableGas||p.enableReactions||p.material.enableSurfaceReactions||p.material.enablePoreOutflow||p.material.evaporation!=EvaporationLaw::Disabled;
    for(const auto& q:h.solid)for(int s=0;s<Ns;++s)if(q.pore[s]!=0)context.gasThermo=true;
    context.condensedMask=h.solid.empty()?0:((1u<<Nc)-1);context.liquid=p.enableFilm;
    if(normal||p.material.enableMelting){if(p.material.phaseCondensed<0||p.material.phaseCondensed>=Nc)return fail("missing phase constituent");context.condensedMask|=1u<<p.material.phaseCondensed;}
    if(p.enableParticles){if(p.particle.condensedIndex<0||p.particle.condensedIndex>=Nc)return fail("missing particle constituent");context.condensedMask|=1u<<p.particle.condensedIndex;}
    if(!validatePhysicsConfig(p,context,error))return false;
    if(p.enableGas){if(h.gas.empty()||h.gas.size()!=h.gasMesh.volumes.size())return fail("enabled gas requires matching populated mesh/inventory");}
    else if(!h.gas.empty()||!h.gasMesh.points.empty()||!h.gasMesh.owner.empty())return fail("disabled gas must not allocate gas inventory/mesh");
    if(h.gas.empty()&&h.solid.empty()&&h.film.empty()&&h.normalEnthalpy.empty())return fail("no physical inventory to advance");
    const std::size_t declaredGasCells=h.gasMesh.volumes.size(),declaredSolidCells=h.solidMesh.volumes.size();
    const HostMesh* addressedMeshes[]={&h.gasMesh,&h.solidMesh};const std::size_t inventoryCounts[]={h.gas.size(),h.solid.size()};
    for(int m=0;m<2;++m)for(int owner:addressedMeshes[m]->owner)if(owner<0||static_cast<std::size_t>(owner)>=inventoryCounts[m])return fail("mesh owner exceeds actual inventory count");
    if(!h.gas.empty()&&!rebuildGeometry(h.gasMesh,error))return false;
    if(!h.solid.empty()&&!rebuildGeometry(h.solidMesh,error))return false;
    if(h.gasMesh.volumes.size()!=declaredGasCells||h.gas.size()!=h.gasMesh.volumes.size())return fail("gas inventory/declared mesh count differs from rebuilt topology");
    if(h.solidMesh.volumes.size()!=declaredSolidCells||h.solid.size()!=h.solidMesh.volumes.size())return fail("material inventory/declared mesh count differs from rebuilt topology");
    if(!validateMaterialSweepRemainder(h,error))return false;
    if(!h.solid.empty()||p.enableFilm||p.material.enableSurfaceReactions||p.material.enablePoreOutflow)
        if(!validateMaterialPolicy(p,error))return false;
    if(p.enableSst&&h.sst.size()!=h.gas.size())return fail("SST inventory missing");
    if(!p.enableSst&&!h.sst.empty())return fail("disabled SST has allocated inventory");
    if(!p.enableParticles&&(!h.particles.empty()||!h.gasVoidFraction.empty()))return fail("disabled particles have allocated state");
    const std::size_t nf=h.surface.area.size(),ne=h.surface.edgeOwner.size();
    if(!normal&&nf){
        if(h.surface.gasFace.size()!=nf||h.surface.solidFace.size()!=nf||h.surface.solidCell.size()!=nf||h.surface.persistentId.size()!=nf
            ||h.surface.centre.size()!=nf||h.surface.normal.size()!=nf||h.surface.baseVelocity.size()!=nf)return fail("incomplete mandatory surface geometry arrays");
        if(!p.enableFilm&&h.filmAux.empty()){
            h.filmAux.resize(nf);for(std::size_t f=0;f<nf;++f){h.filmAux[f].area=h.surface.area[f];h.filmAux[f].normal=h.surface.normal[f];h.filmAux[f].temperature=undefinedValue();}
        }
        if(h.filmAux.size()!=nf)return fail("incomplete surface auxiliary arrays");
        if(h.surface.edgeNeighbour.size()!=ne||h.surface.edgeLength.size()!=ne||h.surface.edgeConormal.size()!=ne||h.surface.edgeOwnerOffset.size()!=ne||h.surface.edgeNeighbourOffset.size()!=ne)return fail("incomplete surface edge arrays");
        if(h.surface.oldArea.empty())h.surface.oldArea=h.surface.area;
        if(h.surface.sweptEdgeArea.empty())h.surface.sweptEdgeArea.assign(ne,0);
        if(h.surface.meshVelocity.empty())h.surface.meshVelocity.assign(nf,Vec3{});
        if(!h.surface.gasMapOffsets.empty()||!h.surface.gasMapFaces.empty()||!h.surface.gasMapWeights.empty())return fail("nonconformal interface mapping is unsupported");
        std::set<int> receivers;bool coupled=false,standalone=false;
        for(std::size_t f=0;f<nf;++f){if(!finite(h.surface.area[f])||h.surface.area[f]<=0||!finite(h.surface.normal[f])||absValue(mag(h.surface.normal[f])-1)>1e-10)return fail("invalid surface area/normal");
            const int g=h.surface.gasFace[f],solid=h.surface.solidCell[f];coupled|=g>=0;standalone|=g<0;
            if(g>=0){if(g>=static_cast<int>(h.gasMesh.owner.size())||!receivers.insert(g).second||h.gasMesh.boundaryKind[g]!=BoundaryKind::Interface)return fail("invalid or duplicate conformal gas receiver");
                if(h.surface.gasDistance.size()!=nf||h.surface.gasDistance[f]<=0)return fail("missing actual gas interface distance");}
            if(solid>=0){if(solid>=static_cast<int>(h.solid.size())||mag(h.surface.baseVelocity[f])!=0)return fail("coupled SolidQ requires a stationary rigid substrate");
                if(h.surface.solidDistance.size()!=nf||h.surface.solidDistance[f]<=0)return fail("missing actual solid interface distance");}
        }
        if(coupled&&standalone)return fail("mixed coupled/standalone surface batch unsupported");
        if(coupled&&p.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal)return fail("prescribed gas motion cannot override coupled recession geometry");
        if(coupled&&(p.enableFilm||p.material.enableMelting||p.material.enableSurfaceReactions)
            &&p.meshMotion.policy!=MeshMotionPolicy::CoupledRecession)return fail("coupled film/receding-surface physics requires CoupledRecession mesh policy");
    }
    if(p.enableFilm&&!normal&&h.film.size()!=nf)return fail("film inventory/surface count mismatch");
    if((!p.enableFilm||normal)&&!h.film.empty())return fail("inactive averaged film inventory must be absent");
    if(normal){const std::size_t layers=h.normalEnthalpy.size(),columns=h.surface.normalOffsets.empty()?0:h.surface.normalOffsets.size()-1;
        if(!p.enableFilm||p.enableGas||!h.solid.empty()||!h.film.empty()||!h.filmAux.empty()||p.meshMotion.policy!=MeshMotionPolicy::Static)return fail("normal thermal instance requires separate fixed-volume columns without gas/solid/film inventory");
        if(!layers||!columns||h.surface.normalOffsets.front()!=0||h.surface.normalOffsets.back()!=static_cast<int>(layers)||h.normalLiquidFraction.size()!=layers||h.surface.normalLayerMass.size()!=layers||h.surface.normalLayerVolume.size()!=layers||h.surface.oldNormalLayerVolume.size()!=layers||h.surface.normalCoordinates.size()!=layers||h.surface.normalThickness.size()!=layers||h.surface.area.size()!=columns||h.surface.normalPressure.size()!=columns||h.surface.normalPressureRate.size()!=columns||h.normalAcceptedWallHeat.size()!=columns)return fail("incomplete normal thermal column state");
        for(std::size_t col=0;col<columns;++col){int begin=h.surface.normalOffsets[col],end=h.surface.normalOffsets[col+1];if(begin<0||end<=begin||end>static_cast<int>(layers))return fail("invalid normal layer offsets");
            for(int i=begin;i<end;++i){Real T=0;if(!recoverNormalSensibleAverage(h.surface.normalLayerMass[i],h.normalEnthalpy[i],h.normalLiquidFraction[i],h.surface.normalPressure[col],p,T))return fail("invalid supplied normal phase/enthalpy inventory");
                if(h.acceptedSteps){Real fraction=0;if(!recoverNormalEnthalpy(h.surface.normalLayerMass[i],h.normalEnthalpy[i],h.surface.normalLayerVolume[i],h.surface.normalPressure[col],p,T,fraction)||!closeEnough(fraction,h.normalLiquidFraction[i],1e-12,p.tolerances.relativeMass))return fail("restored normal fraction disagrees with accepted equilibrium enthalpy");}}}
    }
    if(p.meshMotion.policy==MeshMotionPolicy::PrescribedSinusoidal){
        if(h.gasMesh.referencePoints.size()!=h.gasMesh.points.size()||!h.solid.empty()||nf)return fail("prescribed sinusoidal motion requires immutable gas-only reference points");
        Real volume=0;for(Real v:h.gasMesh.volumes)volume+=v;const Real length=::cbrt(volume),area=length*length;
        const Real temporal=::sin(p.meshMotion.angularFrequency*(h.time-p.meshMotion.timeOrigin));
        for(std::size_t i=0;i<h.gasMesh.points.size();++i){const Vec3 X=h.gasMesh.referencePoints[i];if(!finite(X))return fail("nonfinite immutable reference point");
            const Vec3 expected{X.x+p.meshMotion.amplitude.x*::sin(p.meshMotion.spatialWaveNumber.x*(X.x-p.meshMotion.spatialOrigin.x))*temporal,X.y+p.meshMotion.amplitude.y*::sin(p.meshMotion.spatialWaveNumber.y*(X.y-p.meshMotion.spatialOrigin.y))*temporal,X.z+p.meshMotion.amplitude.z*::sin(p.meshMotion.spatialWaveNumber.z*(X.z-p.meshMotion.spatialOrigin.z))*temporal};
            if(mag(expected-h.gasMesh.points[i])*area>p.tolerances.absoluteGeometry+p.tolerances.relativeGeometry*volume)return fail("mesh points disagree with declared reference trajectory/time");}
    }
    if(p.enableParticles){const auto& t=p.condensed[p.particle.condensedIndex];if(t.cp1!=0||p.particle.maxFaceCrossings<1||p.particle.heatTransferCoefficient<0||p.particle.contactHeatTransferCoefficient<0||p.particle.contactDuration<0||p.particle.emissivity<0||p.particle.emissivity>1)return fail("invalid or variable-cp dilute particle policy");
        for(int s=0;s<Ns;++s)if(p.species[s].cp1!=0)return fail("explicit finite-capacity particle instance requires constant gas cv");
        if(h.gasVoidFraction.size()!=h.gas.size())return fail("particles require explicit dilute void fraction");
        for(Real fraction:h.gasVoidFraction)if(fraction!=1)return fail("occupied-volume/dense particles unsupported");
        const Real particleValues[]={p.particle.heatTransferCoefficient,p.particle.contactHeatTransferCoefficient,p.particle.contactDuration,p.particle.emissivity};
        for(Real value:particleValues)if(!finite(value))return fail("nonfinite particle policy");
        if(p.particle.contact!=ParticleContactMode::ElasticRebound&&p.particle.contact!=ParticleContactMode::Deposit)return fail("unsupported particle contact policy");
        for(std::size_t face=0;face<h.gasMesh.boundaryKind.size();++face){const auto kind=h.gasMesh.boundaryKind[face];
            if(kind==BoundaryKind::Inlet||kind==BoundaryKind::Outlet)return fail("first particle instance requires closed or periodic boundaries; particle reservoir/escape tracking is not enabled");
            if((kind==BoundaryKind::NoSlip||kind==BoundaryKind::Slip||kind==BoundaryKind::Interface)
                &&(p.particle.contactDuration>0||p.particle.contact==ParticleContactMode::Deposit)){
                bool receiver=false;for(std::size_t f=0;f<nf;++f)receiver=receiver||(h.surface.gasFace[f]==static_cast<int>(face)&&h.surface.solidCell[f]>=0);
                if(!receiver)return fail("particle dwell/deposition requires a mapped solid receiver for every reachable wall");
            }
        }
        for(const auto& q:h.particles)if(!finite(q.mass)||q.mass<0||!finite(q.energy)||!finite(q.velocity)||!finite(q.position)||q.cell<0||q.cell>=static_cast<int>(h.gas.size())||!finite(q.contactMechanicalEnergy)||q.contactMechanicalEnergy<0||!finite(q.contactRelativeVelocity))return fail("invalid particle inventory/location");
    }
    for(std::size_t c=0;c<h.gas.size();++c){GasPrimitive w;if(!recoverGas(h.gas[c],h.gasMesh.volumes[c],p,w))return fail("invalid gas inventory/property range");}
    for(std::size_t c=0;c<h.solid.size();++c){MaterialPrimitive w;if(!recoverMaterial(h.solid[c],h.solidMesh.volumes[c],p,w))return fail("invalid solid/pore inventory/property range");}
    for(std::size_t f=0;f<h.film.size();++f){FilmAux a=h.filmAux[f];if(!recoverFilm(h.film[f],h.surface.area[f],a.pressure,p,a))return fail("invalid film inventory/property range");}
    for(const auto& q:h.sst)if(!finite(q.rhoK)||!finite(q.rhoOmega)||q.rhoK<0||q.rhoOmega<=0)return fail("invalid SST inventory");
    if(!finite(h.time)||h.time<0||!validBudget(h.budget))return fail("invalid initial time or cumulative budget");
    error.clear();return true;
}
}
#endif
