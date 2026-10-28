#include "ablation/CpuSurfaceInterface.H"
#include "ablation/DrySurfaceInterfaceMath.H"
namespace chmt {
bool evaluateCpuSurfaceImpl(const HostState& h,const PhysicsConfig& p,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,
    Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,
    const std::vector<Real>* sideMeshVolumeRate,
    const std::vector<GasWallClosureContext>* wallContexts,bool* numericalFailure){
    (void)filmRate; (void)sequence;
    if(numericalFailure)*numericalFailure=true;
    const auto& surface=h.surface;const std::size_t nf=surface.area.size();
    if(((!finite(dt)||dt<=0))||(bulk.size()!=nf)||(!gradients.empty()&&(gradients.size()!=nf))
        ||(surface.gasFace.size()!=nf)||(surface.normal.size()!=nf)||(surface.gasDistance.size()!=nf)
        ||(!surface.solidCell.empty()&&(surface.solidCell.size()!=nf))
        ||(!surface.baseVelocity.empty()&&(surface.baseVelocity.size()!=nf))
        ||(!materialRate.empty()&&(materialRate.size()!=h.solid.size()))){
        if(numericalFailure)*numericalFailure=bulk.size()>=nf&&(gradients.empty()||gradients.size()>=nf)
            &&surface.gasFace.size()>=nf&&surface.normal.size()>=nf&&surface.gasDistance.size()>=nf
            &&(surface.solidCell.empty()||surface.solidCell.size()>=nf)&&(surface.baseVelocity.empty()||surface.baseVelocity.size()>=nf)
            &&(materialRate.empty()||materialRate.size()>=h.solid.size());
        error="CPU surface input layout/interval mismatch";return false;
    }
    if((!surface.gasMapOffsets.empty()||!surface.gasMapFaces.empty()||!surface.gasMapWeights.empty())){
        error="CPU surface requires the supported conformal interface map";return false;}
    if(p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer&&(!wallContexts||(wallContexts->size()!=nf))){
        if(numericalFailure)*numericalFailure=wallContexts&&wallContexts->size()>=nf;
        error="CPU boundaryLayer requires current matching snapshot and certified stage geometry";return false;}
    CpuSurfaceResult result;result.physics.resize(nf);result.wall.resize(nf);
    for(std::size_t f=0;f<nf;++f){const int gf=surface.gasFace[f];
        if(gf<0)continue;
        if((static_cast<std::size_t>(gf)>=h.gasMesh.owner.size()||static_cast<std::size_t>(gf)>=h.gasMesh.areaVectors.size()
            ||static_cast<std::size_t>(gf)>=h.gasMesh.faceIds.size())){error="CPU surface gas face outside real mesh";return false;}
        const int gc=h.gasMesh.owner[gf];if((gc<0||static_cast<std::size_t>(gc)>=h.gasMesh.volumes.size())){error="CPU surface gas owner outside real mesh";return false;}
        SurfacePhysicsInput input;input.bulk=bulk[f];if(!gradients.empty())input.gradient=gradients[f];
        input.gasInventory=conservativeGas(bulk[f],h.gasMesh.volumes[gc],p);
        if(wallContexts){input.wallContext=(*wallContexts)[f];input.bulk.pressure=input.wallContext.matchingPressure;}input.area=surface.area[f];input.normal=surface.normal[f];
        input.gasArea=mag(h.gasMesh.areaVectors[gf]);input.gasNormal=-h.gasMesh.areaVectors[gf]/input.gasArea;
        input.gasDistance=surface.gasDistance[f];input.dt=dt;input.aux.pressure=input.bulk.pressure;
        input.baseVelocity=surface.baseVelocity.empty()?Vec3{}:surface.baseVelocity[f];
        const int sc=surface.solidCell.empty()?-1:surface.solidCell[f];input.hasSolid=sc>=0;
        if(input.hasSolid){if(((static_cast<std::size_t>(sc)>=h.solid.size()||static_cast<std::size_t>(sc)>=h.solidMesh.volumes.size()))
                ||(surface.solidFace.size()!=nf)||(surface.solidDistance.size()!=nf)){if(numericalFailure)*numericalFailure=surface.solidFace.size()>=nf&&surface.solidDistance.size()>=nf;error="CPU interface material owner invalid";return false;}
            const int sf=surface.solidFace[f];if((sf<0||static_cast<std::size_t>(sf)>=h.solidMesh.areaVectors.size())){error="CPU interface solid face invalid";return false;}
            input.solid=input.solidBase=h.solid[sc];input.solidVolume=h.solidMesh.volumes[sc];input.solidDistance=surface.solidDistance[f];
            input.solidArea=mag(h.solidMesh.areaVectors[sf]);input.solidNormal=h.solidMesh.areaVectors[sf]/input.solidArea;
            if(!materialRate.empty())input.solidLocalRate=materialRate[sc];}
        if(sideMeshVolumeRate){if(sideMeshVolumeRate->size()!=nf){if(numericalFailure)*numericalFailure=sideMeshVolumeRate->size()>=nf;error="surface side sweep layout mismatch";return false;}input.sideMeshVolumeRate=(*sideMeshVolumeRate)[f];}
        SurfacePhysicsResult physics;DeviceStatus status;
        if(!solveDrySurfaceInterface(input,p,physics,&status,numericalFailure)){error="CPU surface constitutive solve rejected face "+std::to_string(f)+" code "+std::to_string(status.code);return false;}
        result.wall[f]=dryWallFaceSample(input,physics);
        result.physics[f]=physics;
    }
    output=std::move(result);error.clear();return true;
}
bool evaluateCpuSurface(const HostState& h,const PhysicsConfig& p,const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,const std::vector<SolidQ>& materialRate,const std::vector<FilmQ>& filmRate,Real dt,std::uint64_t sequence,CpuSurfaceResult& output,std::string& error,const std::vector<Real>* sideVolume,const std::vector<GasWallClosureContext>* contexts,bool* numerical){return evaluateCpuSurfaceImpl(h,p,bulk,gradients,materialRate,filmRate,dt,sequence,output,error,sideVolume,contexts,numerical);}

}
