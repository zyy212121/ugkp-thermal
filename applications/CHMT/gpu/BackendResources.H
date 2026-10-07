#ifndef CHMT_GPU_BACKENDRESOURCES_H
#define CHMT_GPU_BACKENDRESOURCES_H
#include "Buffer.H"
#include "gpu/DeviceViews.H"
#include "core/HostState.H"
namespace chmt {
struct StateBuffers {
    Buffer<PhysicsConfig> physics;
    Buffer<GasQ> gas,gasRhs,gasFaceFlux;
    Buffer<GasPrimitive> primitive;Buffer<GasGradient> gradient;
    Buffer<SolidQ> solid,solidRhs,solidFaceFlux;
    Buffer<Real> solidT,porePressure,reactionRate;
    Buffer<Vec3> solidGradient,poreVelocity,solidPressureGradient;
    Buffer<Real> solidPressureGradientSensitivity;
    Buffer<FilmQ> film,filmRhs,filmEdgeFlux;
    Buffer<FilmAux> filmAux;
    Buffer<Vec3> pressureGradient,gasShear;
    Buffer<Real> pressureRate,bottomT,topT,reducedResidual;
    Buffer<FilmRateBudget> filmRates;
    Buffer<FilmPressureVolumeUpdate> storage;
    Buffer<SstQ> sst,sstRhs,sstFaceFlux;
    Buffer<SstPrimitive> sstPrimitive,boundarySst;
    Buffer<Vec3> gradientK,gradientOmega;
    Buffer<Real> production,dissipation,omegaConstraint,voidFraction;
    Buffer<Real> normalH,normalRhs,normalT,normalFraction,normalWallHeat,normalBottomRate,normalTopRate;
    Buffer<ParticleQ> particles;
    Buffer<Real> particleRadiation,particleBodyWork,particleSupportWork;
    Buffer<Vec3> particleSupportImpulse;
    Buffer<ExchangePacket> packets;
    Buffer<Budget> trialBudget;
    void rebindFault(CudaFault& fault)noexcept{
        physics.rebindFault(fault);
        gas.rebindFault(fault);
        gasRhs.rebindFault(fault);
        gasFaceFlux.rebindFault(fault);
        primitive.rebindFault(fault);
        gradient.rebindFault(fault);
        solid.rebindFault(fault);
        solidRhs.rebindFault(fault);
        solidFaceFlux.rebindFault(fault);
        solidT.rebindFault(fault);
        porePressure.rebindFault(fault);
        reactionRate.rebindFault(fault);
        solidGradient.rebindFault(fault);
        poreVelocity.rebindFault(fault);
        solidPressureGradient.rebindFault(fault);
        solidPressureGradientSensitivity.rebindFault(fault);
        film.rebindFault(fault);
        filmRhs.rebindFault(fault);
        filmEdgeFlux.rebindFault(fault);
        filmAux.rebindFault(fault);
        pressureGradient.rebindFault(fault);
        gasShear.rebindFault(fault);
        pressureRate.rebindFault(fault);
        bottomT.rebindFault(fault);
        topT.rebindFault(fault);
        reducedResidual.rebindFault(fault);
        filmRates.rebindFault(fault);
        storage.rebindFault(fault);
        sst.rebindFault(fault);
        sstRhs.rebindFault(fault);
        sstFaceFlux.rebindFault(fault);
        sstPrimitive.rebindFault(fault);
        boundarySst.rebindFault(fault);
        gradientK.rebindFault(fault);
        gradientOmega.rebindFault(fault);
        production.rebindFault(fault);
        dissipation.rebindFault(fault);
        omegaConstraint.rebindFault(fault);
        voidFraction.rebindFault(fault);
        normalH.rebindFault(fault);
        normalRhs.rebindFault(fault);
        normalT.rebindFault(fault);
        normalFraction.rebindFault(fault);
        normalWallHeat.rebindFault(fault);
        normalBottomRate.rebindFault(fault);
        normalTopRate.rebindFault(fault);
        particles.rebindFault(fault);
        particleRadiation.rebindFault(fault);
        particleBodyWork.rebindFault(fault);
        particleSupportWork.rebindFault(fault);
        particleSupportImpulse.rebindFault(fault);
        packets.rebindFault(fault);
        trialBudget.rebindFault(fault);
    }
    bool initialize(const HostState& h,const PhysicsConfig& p,CudaFault& fault,int packetCount){
        if(!physics.uploadOne(p,fault))return false;
        const std::size_t ng=h.gas.size(),nf=h.gasMesh.owner.size(),ns=h.solid.size(),sf=h.solidMesh.owner.size();
        const std::size_t fl=h.film.size(),edges=h.surface.edgeOwner.size(),aux=h.filmAux.size();
        const std::size_t nl=h.normalEnthalpy.size(),nc=h.normalAcceptedWallHeat.size(),np=h.particles.size();
        if(!gas.upload(h.gas,fault)||!solid.upload(h.solid,fault)||!film.upload(h.film,fault)||!filmAux.upload(h.filmAux,fault)
            ||!particles.upload(h.particles,fault)||!sst.upload(h.sst,fault)||!voidFraction.upload(h.gasVoidFraction,fault)
            ||!normalH.upload(h.normalEnthalpy,fault)||!normalFraction.upload(h.normalLiquidFraction,fault)
            ||!normalWallHeat.upload(h.normalAcceptedWallHeat,fault)||!boundarySst.upload(h.gasMesh.boundarySst,fault))return false;
#define CHMT_ALLOC(buffer,count) if(!buffer.resize(count,fault))return false
        CHMT_ALLOC(gasRhs,ng);CHMT_ALLOC(primitive,ng);CHMT_ALLOC(gradient,ng);CHMT_ALLOC(gasFaceFlux,nf);
        CHMT_ALLOC(solidRhs,ns);CHMT_ALLOC(solidT,ns);CHMT_ALLOC(porePressure,ns);CHMT_ALLOC(solidGradient,ns);
        CHMT_ALLOC(poreVelocity,ns);CHMT_ALLOC(solidFaceFlux,sf);CHMT_ALLOC(reactionRate,p.enableReactions?ns*Nr:0);
        CHMT_ALLOC(solidPressureGradient,ns);CHMT_ALLOC(solidPressureGradientSensitivity,ns);
        CHMT_ALLOC(filmRhs,fl);CHMT_ALLOC(filmEdgeFlux,fl?edges:0);CHMT_ALLOC(pressureGradient,fl);
        CHMT_ALLOC(gasShear,aux);CHMT_ALLOC(pressureRate,fl);CHMT_ALLOC(bottomT,aux);CHMT_ALLOC(topT,aux);
        CHMT_ALLOC(reducedResidual,fl);CHMT_ALLOC(filmRates,fl);CHMT_ALLOC(storage,fl);
        CHMT_ALLOC(sstRhs,p.enableSst?ng:0);CHMT_ALLOC(sstFaceFlux,p.enableSst?nf:0);CHMT_ALLOC(sstPrimitive,p.enableSst?ng:0);
        CHMT_ALLOC(gradientK,p.enableSst?ng:0);CHMT_ALLOC(gradientOmega,p.enableSst?ng:0);CHMT_ALLOC(production,p.enableSst?ng:0);
        CHMT_ALLOC(dissipation,p.enableSst?ng:0);CHMT_ALLOC(omegaConstraint,p.enableSst?ng:0);
        CHMT_ALLOC(normalRhs,nl);CHMT_ALLOC(normalT,nl);CHMT_ALLOC(normalBottomRate,nc);CHMT_ALLOC(normalTopRate,nc);
        CHMT_ALLOC(particleRadiation,np);CHMT_ALLOC(particleBodyWork,np);CHMT_ALLOC(particleSupportWork,np);CHMT_ALLOC(particleSupportImpulse,np);
        CHMT_ALLOC(packets,packetCount);CHMT_ALLOC(trialBudget,1);
#undef CHMT_ALLOC
        return true;
    }
    bool copyInventory(const StateBuffers& from,CudaFault& fault,cudaStream_t stream){
#define CHMT_COPY(x) if(!x.copy(from.x,fault,stream))return false
        CHMT_COPY(gas);CHMT_COPY(solid);CHMT_COPY(film);CHMT_COPY(filmAux);CHMT_COPY(particles);CHMT_COPY(sst);
        CHMT_COPY(normalH);CHMT_COPY(normalFraction);CHMT_COPY(normalWallHeat);
#undef CHMT_COPY
        return true;
    }
    bool resetScratch(cudaStream_t s){
        return gasRhs.zero(s)&&gradient.zero(s)&&solidRhs.zero(s)&&filmRhs.zero(s)&&gasShear.zero(s)
            &&pressureGradient.zero(s)&&pressureRate.zero(s)&&filmRates.zero(s)&&reducedResidual.zero(s)
            &&packets.zero(s)&&storage.zero(s)&&omegaConstraint.zero(s)&&particleRadiation.zero(s)
            &&particleBodyWork.zero(s)&&particleSupportWork.zero(s)&&particleSupportImpulse.zero(s);
    }
    GasView gasView(const StateBuffers& base,DeviceStatus* status,const Real* distance){
        GasView v;v.nCells=static_cast<int>(gas.size());v.nFaces=static_cast<int>(gasFaceFlux.size());
        v.q=gas.data();v.stageBase=base.gas.data();v.rhs=gasRhs.data();v.primitive=primitive.data();v.gradient=gradient.data();v.faceFlux=gasFaceFlux.data();v.status=status;v.voidFraction=voidFraction.data();
        if(sst.size()){auto& t=v.sst;t.nCells=v.nCells;t.nFaces=v.nFaces;t.q=sst.data();t.rhs=sstRhs.data();t.primitive=sstPrimitive.data();t.gradientK=gradientK.data();t.gradientOmega=gradientOmega.data();t.faceFlux=sstFaceFlux.data();t.wallDistance=distance;t.boundary=boundarySst.data();t.production=production.data();t.dissipation=dissipation.data();t.omegaConstraint=omegaConstraint.data();}return v;
    }
    SolidView solidView(const StateBuffers& base,DeviceStatus* status){SolidView v;v.nCells=solid.size();v.nFaces=solidFaceFlux.size();v.q=solid.data();v.stageBase=base.solid.data();v.rhs=solidRhs.data();v.temperature=solidT.data();v.porePressure=porePressure.data();v.gradientTemperature=solidGradient.data();v.poreVelocity=poreVelocity.data();v.gradientPressure=solidPressureGradient.data();v.pressureGradientSensitivity=solidPressureGradientSensitivity.data();v.faceFlux=solidFaceFlux.data();v.reactionRate=reactionRate.data();v.status=status;return v;}
    FilmView filmView(const StateBuffers& base,DeviceStatus* status){FilmView v;v.nFaces=film.size();v.nEdges=filmEdgeFlux.size();v.q=film.data();v.stageBase=base.film.data();v.rhs=filmRhs.data();v.aux=filmAux.data();v.edgeFlux=filmEdgeFlux.data();v.pressureGradient=pressureGradient.data();v.gasShear=gasShear.data();v.pressureRate=pressureRate.data();v.bottomTemperature=bottomT.data();v.topTemperature=topT.data();v.reducedResidual=reducedResidual.data();v.rateBudget=filmRates.data();v.storage=storage.data();v.status=status;return v;}
    PacketView packetView(){PacketView p;p.count=packets.size();p.packets=packets.data();return p;}
    bool download(HostState& h,cudaStream_t s)const{return gas.download(h.gas,s)&&solid.download(h.solid,s)&&film.download(h.film,s)&&filmAux.download(h.filmAux,s)&&particles.download(h.particles,s)&&sst.download(h.sst,s)&&voidFraction.download(h.gasVoidFraction,s)&&normalH.download(h.normalEnthalpy,s)&&normalFraction.download(h.normalLiquidFraction,s)&&normalWallHeat.download(h.normalAcceptedWallHeat,s)&&packets.download(h.ledger,s)&&storage.download(h.filmStorage,s);}
};
struct GeometryBuffers {
    Buffer<int> owner,neighbour,periodicPartner,cellFaceOffsets,cellFaces,cellFaceSigns,faceOffsets,facePoints;
    Buffer<Vec3> cellCentre,faceCentre,areaVector,oldPoints,newPoints,meshVelocity,endpointCellCentre,endpointFaceCentre,endpointAreaVector;
    Buffer<Real> oldVolume,newVolume,evaluationVolume,sweptVolume,wallDistance,endpointWallDistance;
    Buffer<BoundaryKind> boundaryKind;
    Buffer<GasPrimitive> gasBoundary,wallTrace;
    Buffer<std::uint64_t> faceIds;
    int nCells=0,nFaces=0,nPoints=0;std::uint64_t geometryVersion=0,topologyHash=0;
    bool upload(const HostMesh& base,const HostMesh& evaluation,const HostMesh& endpoint,const std::vector<Real>& sweeps,const std::vector<Vec3>& areas,CudaFault& fault){
        nCells=endpoint.volumes.size();nFaces=endpoint.owner.size();nPoints=endpoint.points.size();geometryVersion=endpoint.geometryVersion;topologyHash=endpoint.topologyHash;
        if(!owner.upload(endpoint.owner,fault))return false;
        if(!neighbour.upload(endpoint.neighbour,fault))return false;
        if(!periodicPartner.upload(endpoint.periodicPartner,fault))return false;
        if(!cellFaceOffsets.upload(endpoint.cellFaceOffsets,fault))return false;
        if(!cellFaces.upload(endpoint.cellFaces,fault))return false;
        if(!cellFaceSigns.upload(endpoint.cellFaceSigns,fault))return false;
        if(!faceOffsets.upload(endpoint.faceOffsets,fault))return false;
        if(!facePoints.upload(endpoint.facePoints,fault))return false;
        if(!cellCentre.upload(evaluation.cellCentres,fault))return false;
        if(!faceCentre.upload(evaluation.faceCentres,fault))return false;
        if(!areaVector.upload(areas,fault))return false;
        if(!oldPoints.upload(base.points,fault))return false;
        if(!newPoints.upload(endpoint.points,fault))return false;
        if(!meshVelocity.upload(endpoint.meshVelocity,fault))return false;
        if(!endpointCellCentre.upload(endpoint.cellCentres,fault))return false;
        if(!endpointFaceCentre.upload(endpoint.faceCentres,fault))return false;
        if(!endpointAreaVector.upload(endpoint.areaVectors,fault))return false;
        if(!oldVolume.upload(base.volumes,fault))return false;
        if(!newVolume.upload(endpoint.volumes,fault))return false;
        if(!evaluationVolume.upload(evaluation.volumes,fault))return false;
        if(!sweptVolume.upload(sweeps,fault))return false;
        if(!wallDistance.upload(evaluation.wallDistance,fault))return false;
        if(!endpointWallDistance.upload(endpoint.wallDistance,fault))return false;
        if(!boundaryKind.upload(endpoint.boundaryKind,fault))return false;
        if(!gasBoundary.upload(evaluation.boundaryPrimitive,fault))return false;
        if(!wallTrace.upload(endpoint.boundaryPrimitive,fault))return false;
        if(!faceIds.upload(endpoint.faceIds,fault))return false;
        return true;
    }
    GeometryView view(bool endpoint=false,bool sstWallTrace=false)const{GeometryView v;v.nCells=nCells;v.nFaces=nFaces;v.nPoints=nPoints;v.geometryVersion=geometryVersion;v.topologyHash=topologyHash;
        v.owner=owner.data();
        v.neighbour=neighbour.data();
        v.periodicPartner=periodicPartner.data();
        v.cellFaceOffsets=cellFaceOffsets.data();
        v.cellFaces=cellFaces.data();
        v.cellFaceSigns=cellFaceSigns.data();
        v.faceOffsets=faceOffsets.data();
        v.facePoints=facePoints.data();
        v.cellCentre=cellCentre.data();
        v.faceCentre=faceCentre.data();
        v.areaVector=areaVector.data();
        v.oldPoints=oldPoints.data();
        v.newPoints=newPoints.data();
        v.meshVelocity=meshVelocity.data();
        v.oldVolume=oldVolume.data();
        v.newVolume=newVolume.data();
        v.evaluationVolume=evaluationVolume.data();
        v.sweptVolume=sweptVolume.data();
        v.wallDistance=wallDistance.data();
        v.boundaryKind=boundaryKind.data();
        v.faceIds=faceIds.data();
        v.boundaryPrimitive=sstWallTrace?wallTrace.data():gasBoundary.data();
        if(endpoint){v.cellCentre=endpointCellCentre.data();v.faceCentre=endpointFaceCentre.data();v.areaVector=endpointAreaVector.data();v.evaluationVolume=newVolume.data();v.wallDistance=endpointWallDistance.data();}
        return v;
    }
};
struct SurfaceBuffers {
    Buffer<int> gasFace,solidFace,solidCell,edgeOwner,edgeNeighbour,gasMapOffsets,gasMapFaces,normalOffsets;
    Buffer<std::uint64_t> persistentId;
    Buffer<Vec3> prescribedTopTraction,prescribedPressureGradient,edgeOwnerOffset,edgeNeighbourOffset,edgeConormal,centre,normal,meshVelocity,baseVelocity;
    Buffer<Real> gasDistance,solidDistance,edgeLength,area,oldArea,sweptEdgeArea,gasMapWeights,normalCoordinates,normalLayerMass,normalLayerVolume,oldNormalLayerVolume,normalThickness,normalPressure,normalPressureRate,bottomTemperature,topTemperature,bottomHeatFlux,topHeatFlux,radiationPower,solidRecessionResidual,filmVolumeResidual;
    Buffer<GasPrimitive> interfaceGasState;
    int nFaces=0,nEdges=0,nLayers=0,nColumns=0;
    bool upload(const SurfaceMesh& h,CudaFault& fault){nFaces=h.area.size();nEdges=h.edgeOwner.size();nLayers=h.normalLayerMass.size();nColumns=h.normalOffsets.empty()?0:h.normalOffsets.size()-1;
        if(!gasFace.upload(h.gasFace,fault))return false;
        if(!solidFace.upload(h.solidFace,fault))return false;
        if(!solidCell.upload(h.solidCell,fault))return false;
        if(!edgeOwner.upload(h.edgeOwner,fault))return false;
        if(!edgeNeighbour.upload(h.edgeNeighbour,fault))return false;
        if(!gasMapOffsets.upload(h.gasMapOffsets,fault))return false;
        if(!gasMapFaces.upload(h.gasMapFaces,fault))return false;
        if(!normalOffsets.upload(h.normalOffsets,fault))return false;
        if(!persistentId.upload(h.persistentId,fault))return false;
        if(!prescribedTopTraction.upload(h.prescribedTopTraction,fault))return false;
        if(!prescribedPressureGradient.upload(h.prescribedPressureGradient,fault))return false;
        if(!edgeOwnerOffset.upload(h.edgeOwnerOffset,fault))return false;
        if(!edgeNeighbourOffset.upload(h.edgeNeighbourOffset,fault))return false;
        if(!edgeConormal.upload(h.edgeConormal,fault))return false;
        if(!centre.upload(h.centre,fault))return false;
        if(!normal.upload(h.normal,fault))return false;
        if(!meshVelocity.upload(h.meshVelocity,fault))return false;
        if(!baseVelocity.upload(h.baseVelocity,fault))return false;
        if(!gasDistance.upload(h.gasDistance,fault))return false;
        if(!solidDistance.upload(h.solidDistance,fault))return false;
        if(!edgeLength.upload(h.edgeLength,fault))return false;
        if(!area.upload(h.area,fault))return false;
        if(!oldArea.upload(h.oldArea,fault))return false;
        if(!sweptEdgeArea.upload(h.sweptEdgeArea,fault))return false;
        if(!gasMapWeights.upload(h.gasMapWeights,fault))return false;
        if(!normalCoordinates.upload(h.normalCoordinates,fault))return false;
        if(!normalLayerMass.upload(h.normalLayerMass,fault))return false;
        if(!normalLayerVolume.upload(h.normalLayerVolume,fault))return false;
        if(!oldNormalLayerVolume.upload(h.oldNormalLayerVolume,fault))return false;
        if(!normalThickness.upload(h.normalThickness,fault))return false;
        if(!normalPressure.upload(h.normalPressure,fault))return false;
        if(!normalPressureRate.upload(h.normalPressureRate,fault))return false;
        if(!bottomTemperature.upload(h.bottomTemperature,fault))return false;
        if(!topTemperature.upload(h.topTemperature,fault))return false;
        if(!bottomHeatFlux.upload(h.bottomHeatFlux,fault))return false;
        if(!topHeatFlux.upload(h.topHeatFlux,fault))return false;
        if(!radiationPower.upload(std::vector<Real>(nFaces,0),fault))return false;
        if(!solidRecessionResidual.upload(std::vector<Real>(nFaces,0),fault))return false;
        if(!filmVolumeResidual.upload(std::vector<Real>(nFaces,0),fault))return false;
        if(!interfaceGasState.upload(std::vector<GasPrimitive>(nFaces),fault))return false;
        return true;
    }
    SurfaceView view(const GeometryBuffers& solid)const{SurfaceView v;v.nFaces=nFaces;v.nEdges=nEdges;v.nMapEntries=gasMapFaces.size();v.nSolidFaces=solid.nFaces;v.solidAreaVector=solid.areaVector.data();v.solidSweptVolume=solid.sweptVolume.data();
        v.gasFace=gasFace.data();
        v.solidFace=solidFace.data();
        v.solidCell=solidCell.data();
        v.edgeOwner=edgeOwner.data();
        v.edgeNeighbour=edgeNeighbour.data();
        v.gasMapOffsets=gasMapOffsets.data();
        v.gasMapFaces=gasMapFaces.data();
        v.persistentId=persistentId.data();
        v.prescribedTopTraction=prescribedTopTraction.data();
        v.prescribedPressureGradient=prescribedPressureGradient.data();
        v.edgeOwnerOffset=edgeOwnerOffset.data();
        v.edgeNeighbourOffset=edgeNeighbourOffset.data();
        v.edgeConormal=edgeConormal.data();
        v.centre=centre.data();
        v.normal=normal.data();
        v.meshVelocity=meshVelocity.data();
        v.baseVelocity=baseVelocity.data();
        v.gasDistance=gasDistance.data();
        v.solidDistance=solidDistance.data();
        v.edgeLength=edgeLength.data();
        v.area=area.data();
        v.oldArea=oldArea.data();
        v.sweptEdgeArea=sweptEdgeArea.data();
        v.gasMapWeights=gasMapWeights.data();
        v.radiationPower=radiationPower.data();
        v.solidRecessionResidual=solidRecessionResidual.data();
        v.filmVolumeResidual=filmVolumeResidual.data();
        v.interfaceGasState=interfaceGasState.data();
        return v;
    }
    NormalThermalView normalView(StateBuffers& state)const{NormalThermalView v;v.nFaces=nColumns;v.nLayers=nLayers;
        v.offsets=normalOffsets.data();
        v.coordinate=normalCoordinates.data();
        v.thickness=normalThickness.data();
        v.area=area.data();
        v.pressure=normalPressure.data();
        v.pressureRate=normalPressureRate.data();
        v.mass=normalLayerMass.data();
        v.oldVolume=oldNormalLayerVolume.data();
        v.newVolume=normalLayerVolume.data();
        v.bottomTemperature=bottomTemperature.data();
        v.topTemperature=topTemperature.data();
        v.bottomHeatFlux=bottomHeatFlux.data();
        v.topHeatFlux=topHeatFlux.data();
        v.enthalpy=state.normalH.data();
        v.rhs=state.normalRhs.data();
        v.temperature=state.normalT.data();
        v.liquidFraction=state.normalFraction.data();
        v.bottomHeatRate=state.normalBottomRate.data();
        v.topHeatRate=state.normalTopRate.data();
        v.acceptedWallHeat=state.normalWallHeat.data();
        return v;
    }
};
}
#endif
