#include "ablation/InterfaceKernels.H"
#include "gas/KernelSupport.cuh"
namespace chmt {
namespace {
__device__ bool gatherSurfaceInput(int face, GasView gas, SolidView solid, FilmView film,
    GeometryView geometry, SurfaceView surface, PhysicsConfig p, Real dt,
    bool accepted, PacketView packets, SurfacePhysicsInput& in, DeviceStatus* status) {
    const int gasFace=surface.gasFace[face];
    if (gasFace<0 || gasFace>=geometry.nFaces || !film.aux) {
        gasDeviceError(status,ErrorCode::Unsupported,face,gasFace,ErrorLocation::Face); return false;
    }
    if (surface.gasMapOffsets || surface.gasMapFaces || surface.gasMapWeights) {
        gasDeviceError(status,ErrorCode::Unsupported,face,surface.nMapEntries,ErrorLocation::Face);
        return false;
    }
    const int gasCell=geometry.owner[gasFace];
    if (gasCell<0 || gasCell>=gas.nCells || geometry.boundaryKind[gasFace]!=BoundaryKind::Interface) {
        gasDeviceError(status,ErrorCode::InvalidInput,face,gasCell,ErrorLocation::Face); return false;
    }
    const Real volume=accepted ? geometry.newVolume[gasCell]
        : (geometry.evaluationVolume ? geometry.evaluationVolume[gasCell] : geometry.newVolume[gasCell]);
    const Real voidFraction=gas.voidFraction ? gas.voidFraction[gasCell] : 1;
    DeviceStatus local;
    if (!recoverGas(gas.q[gasCell],volume*voidFraction,p,in.bulk,&local,gasCell)) {
        gasDeviceError(status,ErrorCode::PropertyRange,gasCell,local.value,ErrorLocation::Cell); return false;
    }
    if (gas.gradient) in.gradient=gas.gradient[gasCell];
    in.gasInventory=gas.stageBase[gasCell];
    in.area=surface.area[face];
    in.gasArea=mag(geometry.areaVector[gasFace]);
    if (!(in.gasArea>0)) return false;
    in.gasNormal=-geometry.areaVector[gasFace]/in.gasArea;
    in.useSweptGeometry=dt>0;
    in.gasSweptRate=dt>0 ? -geometry.sweptVolume[gasFace]/dt : 0;
    in.gasDistance=surface.gasDistance[face];
    in.normal=surface.normal[face];
    in.baseVelocity=surface.baseVelocity ? surface.baseVelocity[face] : Vec3{};
    in.aux=film.aux[face];
    in.dt=dt;
    const int solidCell=surface.solidCell ? surface.solidCell[face] : -1;
    in.hasSolid=solidCell>=0;
    if (in.hasSolid) {
        if (solidCell>=solid.nCells || !solid.q || !solid.stageBase || !solid.rhs
            || !surface.solidDistance) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,solidCell,ErrorLocation::Face); return false;
        }
        const int solidFace=surface.solidFace ? surface.solidFace[face] : -1;
        if (solidFace<0 || solidFace>=surface.nSolidFaces || surface.nSolidFaces!=solid.nFaces
            || !surface.solidAreaVector || (dt>0 && !surface.solidSweptVolume)) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,solidFace,ErrorLocation::Face); return false;
        }
        in.solidArea=mag(surface.solidAreaVector[solidFace]);
        if (!(in.solidArea>0)) return false;
        in.solidNormal=surface.solidAreaVector[solidFace]/in.solidArea;
        in.solidSweptRate=dt>0 ? surface.solidSweptVolume[solidFace]/dt : 0;
        in.solid=solid.q[solidCell];
        in.solidBase=solid.stageBase[solidCell];
        in.solidLocalRate=solid.rhs[solidCell];
        in.solidDistance=surface.solidDistance[face];
        Real occupied=0;
        for (int c=0; c<Nc; ++c) if (in.solid.condensed[c]!=0) {
            if (!validCondensed(p.condensed[c])) return false;
            occupied+=in.solid.condensed[c]/p.condensed[c].rho;
        }
        if (!finite(in.solid.porosity) || in.solid.porosity<0 || in.solid.porosity>=1) {
            gasDeviceError(status,ErrorCode::Inventory,solidCell,in.solid.porosity,ErrorLocation::Cell);
            return false;
        }
        // Material recovery has already set phi from its exact evaluation
        // geometry. This inversion recovers that volume without a second hidden mesh.
        in.solidVolume=occupied/(1-in.solid.porosity);
    }
    in.hasFilm=p.enableFilm && p.filmThermalMode==FilmThermalMode::ThicknessAveraged;
    if (in.hasFilm) {
        if (!film.q || !film.stageBase || !film.rhs || !film.storage || face>=film.nFaces) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,0,ErrorLocation::Face); return false;
        }
        in.film=film.q[face]; in.filmBase=film.stageBase[face];
        in.filmLocalRate=film.rhs[face];
        in.filmTransportMassRate=film.rhs[face].mass;
        for (int edge=0; edge<surface.nEdges; ++edge) {
            int sign=0;
            if (surface.edgeOwner[edge]==face) sign=1;
            else if (surface.edgeNeighbour[edge]==face) sign=-1;
            if (!sign || !surface.sweptEdgeArea || dt<=0) continue;
            const int owner=surface.edgeOwner[edge], neighbour=surface.edgeNeighbour[edge];
            const Real massFlux=film.edgeFlux ? film.edgeFlux[edge].mass : 0;
            const int donor=massFlux>=0 || neighbour<0 ? owner : neighbour;
            in.sideMeshVolumeRate+=sign*film.aux[donor].thickness*surface.sweptEdgeArea[edge]/dt;
        }
    }
    if ((in.hasSolid && mag(in.baseVelocity)!=0)
        || (in.hasSolid && p.material.enableMelting && in.solid.porosity>p.tolerances.relativeGeometry)
        || (in.hasFilm && in.film.mass>0 && p.material.enableSurfaceReactions)
        || (in.hasSolid && in.hasFilm && in.film.mass>0
            && p.material.enablePoreOutflow && p.permeability>0)) {
        gasDeviceError(status,ErrorCode::Unsupported,face,in.solid.porosity,ErrorLocation::Face);
        return false;
    }
    in.useAcceptedFlux=accepted;
    if (accepted) {
        if (packets.count<InterfacePacketsPerFace*surface.nFaces || dt<=0) return false;
        for (int slot=0; slot<InterfacePacketsPerFace; ++slot) {
            const ExchangePacket& packet=packets.packets[3*face+slot];
            if (packet.face!=geometry.faceIds[gasFace]
                || !validatePacketMath(packet,p.tolerances,&local)) return false;
            const Real rate=1/(dt*in.area);
            if (slot==InterfacePrimarySlot) {
                for (int s=0; s<Ns; ++s) in.acceptedGasSpecies[s]=packet.species[s]*rate;
                for (int c=0; c<Nc; ++c) in.acceptedCondensed[c]=packet.condensed[c]*rate;
            } else if (slot==InterfacePoreSlot && packet.kind==ExchangeKind::PoreGas) {
                for (int s=0; s<Ns; ++s) {
                    in.acceptedPoreSpecies[s]=packet.species[s]*rate;
                    in.acceptedPoreSweep[s]=packet.poreSweep[s]*rate;
                }
            } else if (slot==InterfacePhaseSlot && packet.kind==ExchangeKind::SolidFilm) {
                in.acceptedPhaseMass=packet.mass*rate;
            }
        }
    }
    return true;
}
__global__ void interfacePacketKernel(GasView gas, SolidView solid, FilmView film,
    GeometryView geometry, SurfaceView surface, PacketView packets, PhysicsConfig p,
    Real dt, std::uint64_t step, int stage, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=surface.nFaces || gasStopped(status)) return;
    SurfacePhysicsInput input;
    SurfacePhysicsResult result;
    DeviceStatus physicsStatus;
    if (!gatherSurfaceInput(face,gas,solid,film,geometry,surface,p,dt,false,PacketView{},input,status)
        || !solveSurfaceInterface(input,p,result,&physicsStatus)) {
        gasDeviceError(status,physicsStatus.code ? static_cast<ErrorCode>(physicsStatus.code)
            : ErrorCode::PropertyRange,face,physicsStatus.value,ErrorLocation::Face); return;
    }
    SurfacePacketIdentity identity;
    identity.step=step; identity.stage=stage; identity.geometry=geometry.geometryVersion;
    identity.face=geometry.faceIds[surface.gasFace[face]];
    identity.gasCell=geometry.owner[surface.gasFace[face]];
    identity.solidCell=surface.solidCell ? surface.solidCell[face] : -1;
    identity.filmFace=face;
    identity.oldFilmPV=input.hasFilm ? film.storage[face].oldPV : 0;
    identity.filmPressureGradient=film.pressureGradient ? film.pressureGradient[face] : Vec3{};
    ExchangePacket output[InterfacePacketsPerFace];
    DeviceStatus local;
    if (!assembleSurfacePackets(input,result,identity,p,output,&local)) {
        gasDeviceError(status,static_cast<ErrorCode>(local.code),face,local.value,ErrorLocation::Face);
        return;
    }
    FilmAux aux=film.aux[face];
    aux.solidNormalVelocity=result.solidSpeed;
    aux.normalVelocity=result.topSpeed;
    if (!setCoupledFilmNormalTrace(aux,input.normal,result.wet,
        result.liquidBottomNormal,result.liquidTopNormal,&local,face)) {
        gasDeviceError(status,static_cast<ErrorCode>(local.code),face,local.value,ErrorLocation::Face);
        return;
    }
    // A thread publishes all three roles together only after local validation.
    // Any later aggregate failure rejects the entire provisional batch.
    packets.packets[3*face]=output[0];
    packets.packets[3*face+1]=output[1];
    packets.packets[3*face+2]=output[2];
    surface.interfaceGasState[face]=result.gasState;
    surface.radiationPower[face]=input.area*result.radiationFlux;
    surface.solidRecessionResidual[face]=result.solidRecessionResidual;
    surface.filmVolumeResidual[face]=result.filmVolumeResidual;
    // Front displacement and thermodynamic pressure are committed by Backend,
    // not accumulated or changed in this candidate rate evaluation.
    film.aux[face]=aux;
    if (film.gasShear) film.gasShear[face]=result.shear;
    if (film.topTemperature) film.topTemperature[face]=result.topTemperature;
    if (film.bottomTemperature) film.bottomTemperature[face]=result.bottomTemperature;
}
__global__ void materialDonorKernel(SolidView solid, SurfaceView surface, PacketView packets,
    Real dt, DeviceStatus* status) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=solid.nCells || gasStopped(status)) return;
    SolidQ candidate;
    if (!surfaceDonorCandidate(solid.stageBase[cell],solid.rhs[cell],dt,packets.packets,
        InterfacePacketsPerFace*surface.nFaces,cell,candidate)) {
        gasDeviceError(status,ErrorCode::Inventory,cell,0,ErrorLocation::Cell);
    }
}
__global__ void gasDonorKernel(GasView gas, SurfaceView surface, PacketView packets, DeviceStatus* status) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=gas.nCells || gasStopped(status)) return;
    GasQ candidate=gas.stageBase[cell];
    for (int index=0; index<InterfacePacketsPerFace*surface.nFaces; ++index) {
        const ExchangePacket& packet=packets.packets[index];
        if ((requiredConsumers(packet.kind)&ConsumeGas) && packet.gasCell==cell) {
            candidate+=packetDelta(packet).gas;
        }
    }
    if (!validGasInventory(candidate)) {
        gasDeviceError(status,ErrorCode::Inventory,cell,candidate.mass,ErrorLocation::Cell);
    }
}
__global__ void filmDonorKernel(FilmView film, SurfaceView surface, PacketView packets,
    Real dt, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=film.nFaces || gasStopped(status)) return;
    FilmQ candidate=film.stageBase[face];
    candidate.mass+=dt*film.rhs[face].mass;
    for (int species=0; species<Ns; ++species) candidate.species[species]+=dt*film.rhs[face].species[species];
    for (int slot=0; slot<InterfacePacketsPerFace; ++slot) {
        const ExchangePacket& packet=packets.packets[3*face+slot];
        if (!(requiredConsumers(packet.kind)&ConsumeFilm)) continue;
        const FilmQ delta=packetDelta(packet).film;
        candidate.mass+=delta.mass;
        for (int species=0; species<Ns; ++species) candidate.species[species]+=delta.species[species];
    }
    if (!validFilmInventory(candidate,true)) {
        gasDeviceError(status,ErrorCode::Inventory,face,candidate.mass,ErrorLocation::Face);
    }
    (void)surface;
}
__global__ void interfaceStateKernel(GasView gas, SolidView solid, FilmView film, GeometryView geometry,
    SurfaceView surface, PhysicsConfig p, InterfaceTraceMode mode, PacketView packets,
    Real interval, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=surface.nFaces || gasStopped(status)) return;
    SurfacePhysicsInput input;
    SurfacePhysicsResult result;
    const bool accepted=mode==InterfaceTraceMode::DiscreteAcceptedFlux;
    DeviceStatus physicsStatus;
    if (!gatherSurfaceInput(face,gas,solid,film,geometry,surface,p,accepted ? interval : 0,
        accepted,packets,input,status) || !solveSurfaceInterface(input,p,result,&physicsStatus)) {
        gasDeviceError(status,physicsStatus.code ? static_cast<ErrorCode>(physicsStatus.code)
            : ErrorCode::PropertyRange,face,physicsStatus.value,ErrorLocation::Face); return;
    }
    // Deliberately the ONLY physical output of the read-only trace launch.
    surface.interfaceGasState[face]=result.gasState;
}
bool interfacePointers(GasView gas, SolidView solid, FilmView film, GeometryView geometry,
    SurfaceView surface) {
    return surface.nFaces>=0 && gas.nCells>0 && gas.q && gas.stageBase && geometry.owner && geometry.boundaryKind
        && geometry.faceIds && geometry.areaVector && (geometry.evaluationVolume || geometry.newVolume)
        && surface.gasFace && surface.area && surface.normal && surface.gasDistance
        && surface.interfaceGasState && film.aux
        && (solid.nCells==0 || (solid.q && solid.stageBase && solid.rhs));
}
}
int launchInterfacePackets(GasView gas, SolidView solid, FilmView film, GeometryView geometry,
    SurfaceView surface, PacketView packets, const PhysicsConfig& p, Real dt,
    std::uint64_t step, int stage, DeviceStatus* status, void* stream) {
    if (surface.nFaces==0) return 0;
    if (p.filmThermalMode==FilmThermalMode::ResolvedNormal) return static_cast<int>(cudaErrorNotSupported);
    if (!status || !geometry.evaluationVolume
        || !interfacePointers(gas,solid,film,geometry,surface) || !surface.radiationPower
        || !geometry.sweptVolume || !surface.solidRecessionResidual || !surface.filmVolumeResidual
        || packets.count<InterfacePacketsPerFace*surface.nFaces || !packets.packets
        || !finite(dt) || dt<=0 || stage<0 || stage>1) return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t cudaStream=static_cast<cudaStream_t>(stream);
    interfacePacketKernel<<<(surface.nFaces+127)/128,128,0,cudaStream>>>
        (gas,solid,film,geometry,surface,packets,p,dt,step,stage,status);
    int code=launchError(); if (code) return code;
    if (solid.nCells>0) {
        materialDonorKernel<<<(solid.nCells+127)/128,128,0,cudaStream>>>(solid,surface,packets,dt,status);
        code=launchError(); if (code) return code;
    }
    gasDonorKernel<<<(gas.nCells+127)/128,128,0,cudaStream>>>(gas,surface,packets,status);
    code=launchError(); if (code) return code;
    if (p.enableFilm) {
        filmDonorKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>(film,surface,packets,dt,status);
    }
    return launchError();
}
int launchInterfaceState(GasView gas, SolidView solid, FilmView film, GeometryView geometry,
    SurfaceView surface, const PhysicsConfig& p, InterfaceTraceMode mode, PacketView packets,
    Real interval, DeviceStatus* status, void* stream) {
    if (surface.nFaces==0) return 0;
    if (p.filmThermalMode==FilmThermalMode::ResolvedNormal) return static_cast<int>(cudaErrorNotSupported);
    if (!status || !interfacePointers(gas,solid,film,geometry,surface)
        || (mode==InterfaceTraceMode::ProvisionalModel && packets.count!=0)
        || (mode==InterfaceTraceMode::DiscreteAcceptedFlux
            && (packets.count<InterfacePacketsPerFace*surface.nFaces || !packets.packets
                || !geometry.newVolume || !geometry.sweptVolume || !finite(interval) || interval<=0))
        || (mode!=InterfaceTraceMode::ProvisionalModel && mode!=InterfaceTraceMode::DiscreteAcceptedFlux)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    interfaceStateKernel<<<(surface.nFaces+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>
        (gas,solid,film,geometry,surface,p,mode,packets,interval,status);
    return launchError();
}
} // namespace chmt
