#include "ablation/CpuSurfaceInterface.H"
namespace chmt {
template<bool Terminal>
bool evaluateCpuSurfaceImpl(const HostState& h,const PhysicsConfig& p,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
    Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
    const std::vector<Real>* sideMeshVolumeRate,
    const std::vector<GasWallClosureContext>* wallContexts,bool* numericalFailure){
    if(numericalFailure)*numericalFailure=true;
    const auto& surface=h.surface;const std::size_t nf=surface.area.size();
    if((!Terminal&&(!finite(dt)||dt<=0))||(Terminal?bulk.size()<nf:bulk.size()!=nf)||(!gradients.empty()&&(Terminal?gradients.size()<nf:gradients.size()!=nf))
        ||(Terminal?surface.gasFace.size()<nf:surface.gasFace.size()!=nf)||(Terminal?surface.normal.size()<nf:surface.normal.size()!=nf)||(Terminal?surface.gasDistance.size()<nf:surface.gasDistance.size()!=nf)
        ||(!surface.solidCell.empty()&&(Terminal?surface.solidCell.size()<nf:surface.solidCell.size()!=nf))
        ||(!surface.baseVelocity.empty()&&(Terminal?surface.baseVelocity.size()<nf:surface.baseVelocity.size()!=nf))
        ||(Terminal?h.filmAux.size()<nf:h.filmAux.size()!=nf)||(!materialRate.empty()&&(Terminal?materialRate.size()<h.solid.size():materialRate.size()!=h.solid.size()))
        ||(!filmRate.empty()&&(Terminal?filmRate.size()<h.film.size():filmRate.size()!=h.film.size()))){
        if(numericalFailure)*numericalFailure=bulk.size()>=nf&&(gradients.empty()||gradients.size()>=nf)
            &&surface.gasFace.size()>=nf&&surface.normal.size()>=nf&&surface.gasDistance.size()>=nf
            &&(surface.solidCell.empty()||surface.solidCell.size()>=nf)&&(surface.baseVelocity.empty()||surface.baseVelocity.size()>=nf)
            &&h.filmAux.size()>=nf&&(materialRate.empty()||materialRate.size()>=h.solid.size())&&(filmRate.empty()||filmRate.size()>=h.film.size());
        error="CPU surface input layout/interval mismatch";return false;
    }
    if(!Terminal&&(!surface.gasMapOffsets.empty()||!surface.gasMapFaces.empty()||!surface.gasMapWeights.empty())){
        error="CPU surface requires the supported conformal interface map";return false;}
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer&&(!wallContexts||(Terminal?wallContexts->size()<nf:wallContexts->size()!=nf))){
        if(numericalFailure)*numericalFailure=wallContexts&&wallContexts->size()>=nf;
        error="CPU boundaryLayer requires current matching snapshot and certified stage geometry";return false;}
    CpuSurfaceResult result;result.physics.resize(nf);result.wall.resize(nf);
    for(std::size_t f=0;f<nf;++f){const int gf=surface.gasFace[f];
        if(gf<0)continue;
        if(!Terminal&&(static_cast<std::size_t>(gf)>=h.gasMesh.owner.size()||static_cast<std::size_t>(gf)>=h.gasMesh.areaVectors.size()
            ||static_cast<std::size_t>(gf)>=h.gasMesh.faceIds.size())){error="CPU surface gas face outside real mesh";return false;}
        const int gc=h.gasMesh.owner[gf];if(!Terminal&&(gc<0||static_cast<std::size_t>(gc)>=h.gasMesh.volumes.size())){error="CPU surface gas owner outside real mesh";return false;}
        SurfacePhysicsInput input;input.bulk=bulk[f];if(!gradients.empty())input.gradient=gradients[f];
        input.gasInventory=conservativeGas(bulk[f],h.gasMesh.volumes[gc],p);
        if(wallContexts){input.wallContext=(*wallContexts)[f];input.bulk.pressure=input.wallContext.matchingPressure;}input.area=surface.area[f];input.normal=surface.normal[f];
        input.gasArea=mag(h.gasMesh.areaVectors[gf]);input.gasNormal=-h.gasMesh.areaVectors[gf]/input.gasArea;
        input.gasDistance=surface.gasDistance[f];input.dt=dt;input.aux=h.filmAux[f];input.aux.pressure=input.bulk.pressure;
        input.baseVelocity=surface.baseVelocity.empty()?Vec3{}:surface.baseVelocity[f];
        const int sc=surface.solidCell.empty()?-1:surface.solidCell[f];input.hasSolid=sc>=0;
        if(input.hasSolid){if((!Terminal&&(static_cast<std::size_t>(sc)>=h.solid.size()||static_cast<std::size_t>(sc)>=h.solidMesh.volumes.size()))
                ||(Terminal?surface.solidFace.size()<nf:surface.solidFace.size()!=nf)||(Terminal?surface.solidDistance.size()<nf:surface.solidDistance.size()!=nf)){if(numericalFailure)*numericalFailure=surface.solidFace.size()>=nf&&surface.solidDistance.size()>=nf;error="CPU interface material owner invalid";return false;}
            const int sf=surface.solidFace[f];if(!Terminal&&(sf<0||static_cast<std::size_t>(sf)>=h.solidMesh.areaVectors.size())){error="CPU interface solid face invalid";return false;}
            input.solid=input.solidBase=h.solid[sc];input.solidVolume=h.solidMesh.volumes[sc];input.solidDistance=surface.solidDistance[f];
            input.solidArea=mag(h.solidMesh.areaVectors[sf]);input.solidNormal=h.solidMesh.areaVectors[sf]/input.solidArea;
            if(!materialRate.empty())input.solidLocalRate=materialRate[sc];}
        if(sideMeshVolumeRate){if(Terminal?sideMeshVolumeRate->size()<nf:sideMeshVolumeRate->size()!=nf){if(numericalFailure)*numericalFailure=sideMeshVolumeRate->size()>=nf;error="surface side sweep layout mismatch";return false;}input.sideMeshVolumeRate=(*sideMeshVolumeRate)[f];}
        input.hasFilm=p.enableFilm;
        if(input.hasFilm){if((Terminal?h.film.size()<nf:h.film.size()!=nf)){if(numericalFailure)*numericalFailure=h.film.size()>=nf;error="CPU surface film inventory layout invalid";return false;}
            input.film=input.filmBase=h.film[f];input.film.enthalpy+=pressureVolumeProduct(h.filmAux[f].pressure,h.film[f].mass/p.liquid.rho,input.aux.pressure,h.film[f].mass/p.liquid.rho);if(!filmRate.empty()){input.filmLocalRate=filmRate[f];input.filmTransportMassRate=filmRate[f].mass;}}
        SurfacePhysicsResult physics;DeviceStatus status;
        if(!solveSurfaceInterface<Terminal>(input,p,physics,&status,numericalFailure)){error="CPU surface constitutive solve rejected face "+std::to_string(f)+" code "+std::to_string(status.code);return false;}
        SurfacePacketIdentity id;id.step=sequence;id.stage=1;id.geometry=h.gasMesh.geometryVersion;id.face=h.gasMesh.faceIds[gf];
        id.gasCell=gc;id.solidCell=sc;id.filmFace=f;id.oldFilmPV=input.hasFilm?h.filmAux[f].pressure*h.film[f].mass/p.liquid.rho:0;
        if(input.hasSolid&&physics.wet){
            // The CPU consumes actual recorded primary flux, so no hypothetical
            // gas packet may choose a terminal film-energy remainder here.
            ExchangePacket phase=surfacePacketIdentity(ExchangeKind::SolidFilm,id);
            const Real scale=input.area*dt;phase.mass=scale*physics.phaseMass;
            for(int c=0;c<Nc;++c)phase.condensed[c]=c==p.material.phaseCondensed?phase.mass:0;
            for(int species=0;species<Ns;++species)phase.species[species]=phase.mass*p.material.phaseFilmY[species];
            phase.energy=scale*physics.phaseEnergy;phase.advective=scale*physics.phaseAdvective;phase.conductive=scale*physics.phaseConductive;
            phase.pressureWork=scale*physics.phasePressureWork;phase.viscousWork=scale*physics.phaseViscousWork;phase.liquidKineticAdvection=scale*physics.bottomKineticAdvection;

            if(!Terminal&&!validatePacketMath(phase,p.tolerances,&status)){error="CPU raw phase packet invalid";return false;}
            result.phasePackets.push_back(phase);
        }
        auto& w=result.wall[f];w.temperature=physics.topTemperature;w.primaryKind=physics.wet?ExchangeKind::GasFilm:ExchangeKind::GasSolid;
        w.velocity=physics.gasState.velocity-input.gasNormal*dot(physics.gasState.velocity,input.gasNormal);
        for(int s=0;s<Ns;++s){w.speciesRate[s]=physics.gasSpecies[s];w.poreRate[s]=physics.poreSpecies[s];w.poreSweepRate[s]=physics.poreSweep[s];}
        for(int c=0;c<Nc;++c)w.condensedRate[c]=physics.condensed[c];
        w.radiationFlux=physics.radiationFlux;w.solidNormalVelocity=physics.solidSpeed;w.normalVelocity=physics.topSpeed;
        result.physics[f]=physics;
    }
    output=std::move(result);error.clear();return true;
}
bool evaluateCpuSurface(const HostState& h,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,const std::vector<Real>* sideVolume,const std::vector<GasWallClosureContext>* contexts,bool* numerical){return evaluateCpuSurfaceImpl<false>(h,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,sideVolume,contexts,numerical);}
namespace detail {
bool completeCpuSurface(const HostState& h,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,const std::vector<Real>* sideVolume,const std::vector<GasWallClosureContext>* contexts,bool* numerical){return evaluateCpuSurfaceImpl<true>(h,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,sideVolume,contexts,numerical);}
}

}
