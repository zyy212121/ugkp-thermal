#include "ablation/CpuSurfaceInterface.H"
namespace chmt {
bool evaluateCpuSurface(const HostState& h,const PhysicsConfig& p,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
    Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
    const std::vector<Vec3>* filmPressureGradient,const std::vector<Real>* sideMeshVolumeRate){
    const auto& surface=h.surface;const std::size_t nf=surface.area.size();
    if(!finite(dt)||dt<=0||bulk.size()!=nf||(!gradients.empty()&&gradients.size()!=nf)
        ||surface.gasFace.size()!=nf||surface.normal.size()!=nf||surface.gasDistance.size()!=nf
        ||(!surface.solidCell.empty()&&surface.solidCell.size()!=nf)
        ||(!surface.baseVelocity.empty()&&surface.baseVelocity.size()!=nf)
        ||h.filmAux.size()!=nf||(!materialRate.empty()&&materialRate.size()!=h.solid.size())
        ||(!filmRate.empty()&&filmRate.size()!=h.film.size())){error="CPU surface input layout/interval mismatch";return false;}
    if(!surface.gasMapOffsets.empty()||!surface.gasMapFaces.empty()||!surface.gasMapWeights.empty()){
        error="CPU surface requires the supported conformal interface map";return false;}
    CpuSurfaceResult result;result.physics.resize(nf);result.wall.resize(nf);
    for(std::size_t f=0;f<nf;++f){const int gf=surface.gasFace[f];
        if(gf<0)continue;
        if(static_cast<std::size_t>(gf)>=h.gasMesh.owner.size()||static_cast<std::size_t>(gf)>=h.gasMesh.areaVectors.size()
            ||static_cast<std::size_t>(gf)>=h.gasMesh.faceIds.size()){error="CPU surface gas face outside real mesh";return false;}
        const int gc=h.gasMesh.owner[gf];if(gc<0||static_cast<std::size_t>(gc)>=h.gasMesh.volumes.size()){error="CPU surface gas owner outside real mesh";return false;}
        SurfacePhysicsInput input;input.bulk=bulk[f];if(!gradients.empty())input.gradient=gradients[f];
        input.gasInventory=conservativeGas(bulk[f],h.gasMesh.volumes[gc],p);input.area=surface.area[f];input.normal=surface.normal[f];
        input.gasArea=mag(h.gasMesh.areaVectors[gf]);input.gasNormal=-h.gasMesh.areaVectors[gf]/input.gasArea;
        input.gasDistance=surface.gasDistance[f];input.dt=dt;input.aux=h.filmAux[f];input.aux.pressure=bulk[f].pressure;
        input.baseVelocity=surface.baseVelocity.empty()?Vec3{}:surface.baseVelocity[f];
        const int sc=surface.solidCell.empty()?-1:surface.solidCell[f];input.hasSolid=sc>=0;
        if(input.hasSolid){if(static_cast<std::size_t>(sc)>=h.solid.size()||static_cast<std::size_t>(sc)>=h.solidMesh.volumes.size()
                ||surface.solidFace.size()!=nf||surface.solidDistance.size()!=nf){error="CPU interface material owner invalid";return false;}
            const int sf=surface.solidFace[f];if(sf<0||static_cast<std::size_t>(sf)>=h.solidMesh.areaVectors.size()){error="CPU interface solid face invalid";return false;}
            input.solid=input.solidBase=h.solid[sc];input.solidVolume=h.solidMesh.volumes[sc];input.solidDistance=surface.solidDistance[f];
            input.solidArea=mag(h.solidMesh.areaVectors[sf]);input.solidNormal=h.solidMesh.areaVectors[sf]/input.solidArea;
            if(!materialRate.empty())input.solidLocalRate=materialRate[sc];}
        if(sideMeshVolumeRate){if(sideMeshVolumeRate->size()!=nf){error="surface side sweep layout mismatch";return false;}input.sideMeshVolumeRate=(*sideMeshVolumeRate)[f];}
        input.hasFilm=p.enableFilm;
        if(input.hasFilm){if(h.film.size()!=nf){error="CPU surface film inventory layout invalid";return false;}
            input.film=input.filmBase=h.film[f];input.film.enthalpy+=pressureVolumeProduct(h.filmAux[f].pressure,h.film[f].mass/p.liquid.rho,input.aux.pressure,h.film[f].mass/p.liquid.rho);if(!filmRate.empty()){input.filmLocalRate=filmRate[f];input.filmTransportMassRate=filmRate[f].mass;}}
        SurfacePhysicsResult physics;DeviceStatus status;
        if(!solveSurfaceInterface(input,p,physics,&status)){error="CPU surface constitutive solve rejected face "+std::to_string(f)+" code "+std::to_string(status.code);return false;}
        SurfacePacketIdentity id;id.step=sequence;id.stage=1;id.geometry=h.gasMesh.geometryVersion;id.face=h.gasMesh.faceIds[gf];
        id.gasCell=gc;id.solidCell=sc;id.filmFace=f;id.oldFilmPV=input.hasFilm?h.filmAux[f].pressure*h.film[f].mass/p.liquid.rho:0;
        if(filmPressureGradient){if(filmPressureGradient->size()!=nf){error="film pressure-gradient layout mismatch";return false;}id.filmPressureGradient=(*filmPressureGradient)[f];}
        if(input.hasSolid&&physics.wet){
            // The CPU consumes actual recorded primary flux, so no hypothetical
            // gas packet may choose a terminal film-energy remainder here.
            ExchangePacket phase=surfacePacketIdentity(ExchangeKind::SolidFilm,id);
            const Real scale=input.area*dt;phase.mass=scale*physics.phaseMass;
            for(int c=0;c<Nc;++c)phase.condensed[c]=c==p.material.phaseCondensed?phase.mass:0;
            for(int species=0;species<Ns;++species)phase.species[species]=phase.mass*p.material.phaseFilmY[species];
            phase.energy=scale*physics.phaseEnergy;phase.advective=scale*physics.phaseAdvective;phase.conductive=scale*physics.phaseConductive;
            phase.pressureWork=scale*physics.phasePressureWork;phase.viscousWork=scale*physics.phaseViscousWork;phase.liquidKineticAdvection=scale*physics.bottomKineticAdvection;
            Vec3 gradient=id.filmPressureGradient;gradient-=(p.gravity-input.normal*dot(p.gravity,input.normal))*p.liquid.rho;
            const FilmProfile profile=filmProfile(input.aux.thickness,p.liquidViscosity,input.baseVelocity,physics.shear,gradient);
            const Vec3 bottomShear=profile.bottomShear-input.solidNormal*dot(profile.bottomShear,input.solidNormal);
            phase.momentum=input.solidNormal*(physics.gasState.pressure*input.solidArea*dt)-bottomShear*scale;
            if(!validatePacketMath(phase,p.tolerances,&status)){error="CPU raw phase packet invalid";return false;}
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
}
