#include "gpu/Backend.H"
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/ParticleCouplingStorage.cuh"
#include "gpu/CourantReductionStorage.cuh"
#include "gpu/CoupledSurfaceTransfer.cuh"
#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "gpu/CoupledSurfacePipeline.cuh"
#include "gpu/SingleGasContract.H"
#include "configuration/CouplingEntry.H"
#include "coupling/IntervalAudit.H"
#include "mesh/TrajectorySurface.H"
#include "../../../common/GpuGasOperatorDependencies.cuh"
#include "../../../common/GpuPrecisionTypes.H"
#include "../../../common/gasNumerics/CharacteristicMuscl.cuh"
#include "../../../common/gasNumerics/OpenFoamLimitedLinear.cuh"
#include "../../../common/gasNumerics/OpenFoamViscousFlux.cuh"
#include "../../../common/gasNumerics/OpenFoamWallFunctions.cuh"
#include "../../../common/gasNumerics/RiemannBoundaryState.cuh"
#include "../../../common/gasNumerics/RiemannGasFlux.cuh"
#include "../../../common/gasNumerics/GpuSstAlgebra.cuh"
#include "../../../common/gasNumerics/GpuLesAlgebra.cuh"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <memory>
#include <cerrno>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include "../../../common/gasTransport/ResidentWallHistory.inl"
namespace chmt {
struct Backend {
    CudaFault executionFault;
    DeviceStream executionStream;
    ugkp::diagnostics::GpuOwner diagnostics;
    ugkp::diagnostics::DeviceContext hostDiagnostics{};
    ugkp::diagnostics::HostSink diagnosticSink;
    ModelConfig model;GasExecutionOptions options;HostState accepted,trial;
    HostMesh endpointGas,endpointSolid;SurfaceMesh endpointSurface;HostStageGeometry gasStage,solidStage;
    SharedGasDeviceStorage storage;ParticleCouplingStorage particles;WallProgram program;IntervalHistory history;MaterialDonorReserve reserve;
    CoupledCourantStorage courantReductionStorage;
    CudaFault surfaceTransferFault;
    Buffer<int> interfaceFaces,interfaceFaceSlots,interfaceCells;
    Buffer<Real> interfaceProgress;
    Buffer<unsigned char> interfaceWallChanged;
    Buffer<WallFaceSample> interfaceStageWall;
    Buffer<CoupledSurfaceInput> interfaceInputs;
    Buffer<WallFaceSample> interfaceWallProgram;
    GasWallProgramPosition stageWallPosition{};
    bool interfaceGeometryReady=false;
    Buffer<CoupledSurfaceRecord> interfaceRecords;
    Buffer<CoupledSurfaceDownload> interfaceDownloads;
    Buffer<CoupledWallDownload> interfaceWallDownloads;
    Buffer<PhysicsConfig> interfacePhysics;
    bool poisoned=false;std::string error;
    GasIntervalRecord record;
    Real microDt=0,microTime=0,stageTime=0,stagePacketDt=0;
    WindowTime elapsed{};Real nextGasSeed=0;bool microGeometryPrepared=false;int gasStageIndex=0;
    int nCells=0,nFaces=0,nInternalFaces=0,fixedCellBlockThreads=128,fixedFaceBlockThreads=128;
    int hostTurbulenceModel=0,hostGasFluxScheme=1,hostGasTimeIntegrator=1,hasPeriodicFaces=0;
    ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
    ugkwp::GasGeometryState<Real> gasGeometry;
    ugkwp::GasBoundaryLayerState<Real> gasBoundaryLayer;
    Real preparedWallTime=0,preparedWallDt=0;bool preparedWallReady=false,terminalWallFailure=false;
    SharedGasDeviceView* deviceState=nullptr;cudaStream_t gasCaptureStream=nullptr;
    ~Backend(){if(gasCaptureStream)executionStream.sync();}
};
inline void bindSharedGasStorage(Backend& b){
    const auto& view=b.storage.hostView();
    b.preparedWallReady=false;
    b.gasSpecies=view.gasSpecies;b.gasBoundaryLayer=view.gasBoundaryLayer;b.gasGeometry=view.gasGeometry;
    b.deviceState=b.storage.deviceView();b.nCells=view.nCells;b.nFaces=view.nFaces;b.nInternalFaces=view.nInternalFaces;
}
inline WindowTime coupledStageTime(const Backend& b){return b.elapsed+(b.gasStageIndex==0?0:b.gasStageIndex==1?b.microDt:.5*b.microDt);}
bool loadGasWindowState(Backend&,const HostState&,std::string&);
int prepareCoupledGeometry(Backend*);
int finishCoupledGeometry(Backend*);
int advanceCoupledParticles(Backend*,Real);
int prepareCoupledBoundaryLayer(Backend*,Real,Real);
int applyCoupledFaces(Backend*,Real,Real);
int accountCoupledFaces(Backend*,Real);
bool downloadCoupledRecord(Backend&);
namespace transport {
using std::isfinite;
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) static_cast<double>(x)
#define GPU_OPERATOR_TINY(x) x
constexpr double OfSmall=2.22044604925031308085e-16,OfVSmall=2.22507385850720138309e-308,OfGreat=1/OfSmall;
struct GasPrimDevice {double rho,ux,uy,uz,p,T;};
#include "../../../common/operators/clampMin.cuh"
#include "../../../common/operators/clampRange.cuh"
#include "../../../common/operators/linearScheduledValueDevice.cuh"
#include "../../../common/operators/makeGasPrimDevice.cuh"
#include "../../../common/operators/riemannFacePrimitiveForGradient.cuh"
#include "../../../common/operators/computeGasPrimitiveGradientsKernel.cuh"
#include "../../../common/operators/computeSstGradientsKernel.cuh"
#include "../../../common/operators/sstVelocityInvariants.cuh"
#include "../../../common/operators/computeGasHllcAdcSensorKernel.cuh"
#include "../../../common/operators/updateBarthLimiter.cuh"
#include "../../../common/operators/computeGasGradientLimiterKernel.cuh"
#include "../../../common/operators/computeGasEddyViscosityKernel.cuh"
#include "../../../common/operators/updateWaveTransmissivePressureBoundaryKernel.cuh"
#include "../../../common/operators/updateLegacyGasBoundaryMirrorKernel.cuh"
#include "../../../common/operators/gasFaceSubgridTransportProperties.cuh"
#include "../../../common/operators/computeRiemannGasFaceFluxDevice.cuh"
#include "../../../common/operators/computeGasInternalFaceFluxKernel.cuh"
#include "../../../common/operators/computeGasFluxPositivityScaleKernel.cuh"
#include "../../../common/operators/computeSstFaceFluxKernel.cuh"
#include "../../../common/operators/evaluateGasBoundaryLayerKernel.cuh"


template<class State>
__global__ void refreshCoupledWallFluxKernel(State* state){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=state->gasBoundaryLayer.count)return;
    const int face=state->gasBoundaryLayerModel.faces[slot];
    computeRiemannGasFaceFluxDevice<false>(*state,face,state->gasPhiRho[face],
        state->gasPhiRhoUx[face],state->gasPhiRhoUy[face],state->gasPhiRhoUz[face],state->gasPhiRhoE[face]);
}
} // transport
} // chmt
#include "../../../common/GpuGasHostPolicy.cuh"
namespace chmt { namespace transport {
thread_local std::string lastError;
inline void setLastError(const char* text,cudaError_t code){lastError=std::string(text)+": "+cudaGetErrorString(code);}
inline void setLastErrorText(const char* text){lastError=text;}
struct GasHostPolicy:GasHostWithWallEnergy<Real,Real,&accountCoupledFaces> {
    static int prepareBoundaryLayer(Backend* s,Real dt,Real time){return prepareCoupledBoundaryLayer(s,dt,time);}
    static int applyFaceSources(Backend* s,Real dt,Real time){return applyCoupledFaces(s,dt,time);}
};
#include "../../../common/GpuGasAdvance.cuh"
#include "../../../common/GpuGasCourant.cuh"
struct CoupledTrialPolicy {
    static int begin(Backend* b){
        b->preparedWallReady=false;b->terminalWallFailure=false;ugkwp::invalidateGasBoundaryLayerPreparation(*b);
        b->gasStageIndex=0;b->record={};b->record.microSequence=b->history.records().size()+1;
        b->record.begin=b->microTime;b->record.actualDt=b->microDt;b->record.localBegin=b->elapsed;b->record.localEnd=b->elapsed+b->microDt;b->record.end=windowAbsolute(b->program.interval.begin,b->record.localEnd);b->record.gasGeometry=b->trial.gasMesh.geometryVersion;b->record.solidGeometry=b->trial.solidMesh.geometryVersion;
        if(!b->storage.clearTrialStatus())return 1;
        if(!b->microGeometryPrepared&&prepareCoupledGeometry(b)!=0)return 1;
        if(b->model.physics.meshMotion.policy!=MeshMotionPolicy::Static){
            b->record.gasGeometry=b->endpointGas.geometryVersion;b->record.solidGeometry=b->endpointSolid.geometryVersion;
        }
        b->microGeometryPrepared=false;
        if(b->particles.active()&&!b->particles.result(ugkp::thermal::prepareParticleGasInterval(
            b->particles.owner(),b->gasCaptureStream),b->error))return 1;
        return 0;
    }
    static const Real* stageVolumes(Backend* b,bool after){const auto& g=b->storage.hostView().gasGeometry;return g.enabled?(after?g.newVolume:g.oldVolume):b->storage.hostView().V;}

    static int validateTimeStep(Backend*,Real){return 0;}
    static int applySources(Backend* b,Real dt){
        if(finishCoupledGeometry(b))return 1;
        return advanceCoupledParticles(b,dt);
    }
    static int validate(Backend* b){
        if(!b->storage.checkStatus(b->error))return 1;
        return 0;
    }
    static int commit(Backend* b){
        if(!downloadCoupledRecord(*b))return 1;
        const auto& model=b->storage.hostView().gasBoundaryLayerModel;
        if(model.acceptedHistory&&b->gasBoundaryLayer.count){
            ::commitAcceptedWallHistory<<<(b->gasBoundaryLayer.count+63)/64,64,0,b->gasCaptureStream>>>(b->deviceState);
            const auto error=cudaGetLastError();if(error!=cudaSuccess){b->error=cudaGetErrorString(error);return 1;}
        }
        // Material consumes the sparse accepted history only after all RK stages.
        // The complete gas endpoint stays resident until downloadGasWindow.
        b->trial.time=b->record.end;return 0;
    }

};
}}
namespace chmt {
bool downloadCoupledRecord(Backend& b){
    std::vector<CoupledSurfaceDownload> records;
    std::vector<CoupledWallDownload> walls;
    const int count=int(b.interfaceRecords.size());
    if(count){
        packCoupledSurfaceKernel<<<(count+127)/128,128,0,b.gasCaptureStream>>>(b.interfaceRecords.data(),count,b.interfaceFaces.data(),b.particles.wallExchange(),b.interfaceDownloads.data(),b.interfaceWallDownloads.data());
        if(!b.surfaceTransferFault.check(cudaGetLastError(),"pack accepted coupled exchange")){
            b.error=b.surfaceTransferFault.message;return false;
        }
    }
    if(!b.storage.enqueueRead(b.interfaceDownloads.data(),b.interfaceDownloads.size(),records)
        ||!b.storage.enqueueRead(b.interfaceWallDownloads.data(),b.interfaceWallDownloads.size(),walls)
        ||!b.storage.observeDiagnostics()){b.error=b.storage.error();return false;}
    b.record.gasTraceTime=b.stageTime;
    for(std::size_t i=0;i<records.size();++i){const auto& r=records[i];
        if(r.failed){
            if(r.failed==2){
                b.terminalWallFailure=true;const auto& wall=walls[i];
                b.error="boundaryLayer rejected face "+std::to_string(b.trial.surface.gasFace[i])+" slot "+std::to_string(i)
                    +" transport code "+std::to_string(wall.transportCode)+" wall code "+std::to_string(int(wall.status.code))
                    +" node "+std::to_string(wall.status.node)+" iteration "+std::to_string(wall.status.iteration)
                    +" residual "+(std::isfinite(wall.status.residual)?std::to_string(wall.status.residual):"NOT_AVAILABLE");
            }else b.error="coupled interface trace/packet failed at face "+std::to_string(i);
            return false;
        }
        b.record.packets.push_back(r.primary);b.record.packets.push_back(r.pore);
        if(b.particles.active()){
            ExchangePacket contact{};contact.kind=ExchangeKind::ParticleWall;
            contact.step=r.primary.step;contact.geometry=r.primary.geometry;contact.face=r.primary.face;
            contact.stage=2;contact.gasCell=r.primary.gasCell;contact.solidCell=r.primary.solidCell;
            contact.filmFace=r.primary.filmFace;contact.consumerMask=ConsumeParticle;
            contact.energy=contact.conductive=r.particleWallEnergy;
            b.record.packets.push_back(contact);
        }
        b.record.gasTrace.push_back(r.bulk);b.record.gasGradient.push_back(r.gradient);
        b.record.gasAcceptedWall.push_back(r.acceptedWall);
        b.record.gasTraction.push_back(r.traction);
        b.record.radiationEnergy.push_back(r.radiationEnergy);b.record.radiationReceiver.push_back(r.radiationReceiver);
        if(b.gasBoundaryLayer.enabled)b.record.gasWallMatching.push_back(walls[i].matching);
    }
    if(b.gasBoundaryLayer.enabled){
        b.trial.gasWallDiagnostics.resize(records.size());
        for(std::size_t i=0;i<records.size();++i)static_cast<ugkwp::gaswall::WallExchangeOutput<Real,Ns>&>(b.trial.gasWallDiagnostics[i])=walls[i].layer;
        b.trial.gasWallDiagnosticTime=b.stageTime;
    }
    return true;
}
int prepareCoupledGeometry(Backend* b){
    // CFL and RK stages leave host geometry unchanged until finishCoupledGeometry.
    // Consume that pre-step view directly; endpoint outputs have separate owners.
    const auto& base=b->trial;const bool moving=b->model.physics.meshMotion.policy!=MeshMotionPolicy::Static;
    // Static stages borrow the accepted window geometry. Owned endpoint and
    // stage arrays exist only for motion or an explicit history/export request.
    std::vector<Real> gasSweep,solidSweep;
    if(moving){
        b->endpointGas=base.gasMesh;b->endpointSolid=base.solidMesh;b->endpointSurface=base.surface;
        gasSweep.assign(base.gasMesh.owner.size(),0);solidSweep.assign(base.solidMesh.owner.size(),0);
        WallKnot endpoint;if(!sampleGasWallProgramLocal(b->program,b->elapsed+b->microDt,endpoint,b->error))return 1;
        const auto& tolerance=b->model.physics.tolerances;
        auto stage=[&](const HostMesh& old,const std::vector<Vec3>& points,HostMesh& next,std::vector<Real>& sweep){
            return makeStageGeometry(old,points,b->microDt,next,sweep,b->error,tolerance);
        };
        if(!stage(base.gasMesh,endpoint.gasPoints,b->endpointGas,gasSweep)
            ||!stage(base.solidMesh,endpoint.solidPoints,b->endpointSolid,solidSweep))return 1;
        if(!rebuildTrajectorySurface(base,b->endpointGas,b->endpointSolid,b->microDt,b->endpointSurface,b->error,tolerance))return 1;
    }
    auto trace=[&](const HostMesh& old,const HostMesh& end,const std::vector<Real>& sweep,HostStageGeometry& out){
        out.interval=b->microDt;out.geometryVersion=end.geometryVersion;out.topologyHash=end.topologyHash;
        out.oldVolume=old.volumes;out.newVolume=end.volumes;out.evaluationVolume=old.volumes;out.sweptVolume=sweep;
        out.areaVector=old.areaVectors;out.cellCentre=old.cellCentres;out.faceCentre=old.faceCentres;out.oldPoints=old.points;out.newPoints=end.points;
    };
    const auto& endpointGas=moving?b->endpointGas:base.gasMesh;
    const auto& endpointSolid=moving?b->endpointSolid:base.solidMesh;
    const auto& endpointSurface=moving?b->endpointSurface:base.surface;
    b->record.gasGeometry=endpointGas.geometryVersion;b->record.solidGeometry=endpointSolid.geometryVersion;
    if(moving){
        trace(base.gasMesh,endpointGas,gasSweep,b->gasStage);trace(base.solidMesh,endpointSolid,solidSweep,b->solidStage);
        if(!b->storage.bindStageGeometry(b->gasStage)){b->error=b->storage.error();return 1;}}
    else if(!b->storage.clearGeometry()){b->error=b->storage.error();return 1;}
    if(moving||!b->interfaceGeometryReady){
        const auto& surface=base.surface;std::vector<CoupledSurfaceInput> inputs(surface.area.size());
        for(std::size_t i=0;i<inputs.size();++i){const int f=surface.gasFace[i];auto& in=inputs[i];
            in.area=surface.area[i];in.gasDistance=surface.gasDistance[i];
            in.normal=surface.normal[i];in.sweptVolume=moving?-b->gasStage.sweptVolume[f]:0;
            const Vec3 support=surface.baseVelocity.empty()?Vec3{}:surface.baseVelocity[i];
            const Vec3 motion=endpointSurface.meshVelocity.empty()?Vec3{}:endpointSurface.meshVelocity[i];
            in.particleMaterialVelocity=support+in.normal*dot(motion-support,in.normal);
            in.identity.face=surface.persistentId[i];in.identity.stage=1;in.identity.gasCell=base.gasMesh.owner[f];
            in.identity.solidCell=surface.solidCell[i];in.identity.filmFace=int(i);
        }
        if(!b->storage.write(b->interfaceInputs.data(),inputs)){b->error=b->storage.error();return 1;}
        b->interfaceGeometryReady=true;
    }
    b->gasGeometry=b->storage.hostView().gasGeometry;return 0;
}
int finishCoupledGeometry(Backend* b){
    // Static geometry and boundary topology were published at window start.
    // Re-uploading every mesh array here only creates synchronization work.
    if(b->model.physics.meshMotion.policy==MeshMotionPolicy::Static)return 0;
    if(!b->particles.mapGeometry(b->trial.gasMesh,b->endpointGas,b->storage,b->gasCaptureStream,b->error))return 1;
    b->trial.gasMesh=b->endpointGas;b->trial.solidMesh=b->endpointSolid;b->trial.surface=b->endpointSurface;
    if(!b->storage.uploadGeometry(b->trial.gasMesh,false)||!b->storage.refreshView()){b->error=b->storage.error();return 1;}
    bindSharedGasStorage(*b);
    if(!b->particles.refreshGeometry(b->storage,b->gasCaptureStream,b->error))return 1;
    return 0;
}
int advanceCoupledParticles(Backend* b,Real dt){
    if(!b->particles.active())return 0;
    const int count=b->interfaceFaces.size();
    GasWallProgramPosition endpoint;
    if(!locateGasWallProgramLocal(b->program,b->elapsed+dt,endpoint,b->error))return 1;
    if(count){const auto surface=b->particles.surface();
        publishParticleMaterialSurfaceKernel<<<(count+127)/128,128,0,b->gasCaptureStream>>>(count,
            b->interfaceFaces.data(),b->interfaceInputs.data(),b->interfaceWallProgram.data(),endpoint,
            const_cast<double*>(surface.temperature),const_cast<double*>(surface.velocityX),
            const_cast<double*>(surface.velocityY),const_cast<double*>(surface.velocityZ),const_cast<double*>(surface.effusivity));
        if(!b->surfaceTransferFault.check(cudaGetLastError(),"publish resident particle material surface")){
            b->error=b->surfaceTransferFault.message;return 1;}
    }
    return b->particles.result(ugkp::thermal::advanceBoundParticlePhase(b->particles.owner(),dt,
        windowAbsolute(b->program.interval.begin,b->elapsed+dt),b->gasCaptureStream),b->error)?0:1;
}
int prepareResidentSurface(Backend* b,Real dt,bool transportReady){
    const int count=int(b->interfaceFaces.size());if(!count)return 0;
    if(!locateGasWallProgramLocal(b->program,coupledStageTime(*b),b->stageWallPosition,b->error))return 1;
    const bool reactive=b->model.physics.material.enableSurfaceReactions||b->model.physics.material.enablePoreOutflow;
    if(reactive)boundSurfaceProgressKernel<<<(b->interfaceCells.size()+127)/128,128,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceCells.data(),int(b->interfaceCells.size()),b->interfaceFaceSlots.data(),b->interfaceInputs.data(),b->interfaceWallProgram.data(),count,
        b->stageWallPosition,dt,transportReady,b->interfaceProgress.data());
    prepareResidentSurfaceKernel<<<(count+127)/128,128,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceFaces.data(),count,b->interfaceInputs.data(),b->interfaceWallProgram.data(),
        b->stageWallPosition,reactive?b->interfaceProgress.data():nullptr,b->interfaceStageWall.data(),b->interfaceWallChanged.data());
    if(!b->surfaceTransferFault.check(cudaGetLastError(),"bound resident surface reaction progress")){
        b->error=b->surfaceTransferFault.message;return 1;}
    return 0;
}
int evaluateResidentSurfaceLayer(Backend* b,Real dt,Real stageTime,bool changedOnly=false){
    const auto& model=b->storage.hostView().gasBoundaryLayerModel;
    const int workers=model.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport
        ?b->gasBoundaryLayer.count:model.workspaceCount;
    transport::evaluateGasBoundaryLayerKernel<<<(workers+31)/32,32,0,b->gasCaptureStream>>>(b->deviceState,
        changedOnly?b->interfaceWallChanged.data():nullptr);
    const auto error=cudaGetLastError();if(error!=cudaSuccess){b->error=std::string("boundaryLayer CUDA launch: ")+cudaGetErrorString(error);return 1;}
    b->preparedWallTime=stageTime;b->preparedWallDt=dt;b->preparedWallReady=true;return 0;
}
int prepareCoupledBoundaryLayer(Backend* b,Real dt,Real stageTime){
    if(!b->gasBoundaryLayer.enabled)return 0;
    b->preparedWallReady=false;
    if(prepareResidentSurface(b,dt,false))return 1;
    return evaluateResidentSurfaceLayer(b,dt,stageTime);
}
int applyCoupledFaces(Backend* b,Real dt,Real stageTime){
    const int count=int(b->interfaceFaces.size());if(!count)return 0;
    // Transport fluxes now exist. Re-evaluate the physical wall closure using
    // the remaining donor budget, before constructing any accepted packet.
    const bool reactive=b->model.physics.material.enableSurfaceReactions||b->model.physics.material.enablePoreOutflow;
    const bool staged=reactive||b->gasBoundaryLayer.enabled;
    if(staged&&prepareResidentSurface(b,dt,true))return 1;
    if(!staged&&!locateGasWallProgramLocal(b->program,coupledStageTime(*b),b->stageWallPosition,b->error))return 1;
    if(b->gasBoundaryLayer.enabled){
        if((reactive||!b->preparedWallReady||b->preparedWallTime!=stageTime||b->preparedWallDt!=dt)
            &&evaluateResidentSurfaceLayer(b,dt,stageTime,b->preparedWallReady
                &&b->preparedWallTime==stageTime&&b->preparedWallDt==dt))return 1;
        transport::refreshCoupledWallFluxKernel<<<(b->gasBoundaryLayer.count+31)/32,32,0,b->gasCaptureStream>>>(b->deviceState);
        const auto code=cudaGetLastError();
        if(code!=cudaSuccess){b->error=std::string("refresh current wall face flux: ")+cudaGetErrorString(code);return 1;}
    }
    b->stageTime=windowAbsolute(b->program.interval.begin,coupledStageTime(*b));b->stagePacketDt=dt;
    applyCoupledSurfaceKernel<<<(count+127)/128,128,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceFaces.data(),count,b->interfaceInputs.data(),staged?b->interfaceStageWall.data():b->interfaceWallProgram.data(),
        staged?GasWallProgramPosition{}:b->stageWallPosition,b->interfacePhysics.data(),dt,b->gasStageIndex==0,b->record.microSequence,b->record.gasGeometry,b->interfaceRecords.data());
    if(!b->surfaceTransferFault.check(cudaGetLastError(),"evaluate resident coupled interface")){
        b->error=b->surfaceTransferFault.message;return 1;}
    return 0;
}
int accountCoupledFaces(Backend* b,Real ledgerDt){
    const int count=int(b->interfaceFaces.size());
    if(count){
        accountCoupledSurfaceKernel<<<(count+127)/128,128,0,b->gasCaptureStream>>>(b->deviceState,
            b->interfaceFaces.data(),count,b->interfaceInputs.data(),b->interfaceWallProgram.data(),b->stageWallPosition,
            b->interfacePhysics.data(),b->stagePacketDt,ledgerDt,b->interfaceRecords.data());
        if(!b->surfaceTransferFault.check(cudaGetLastError(),"accumulate resident coupled exchange")){
            b->error=b->surfaceTransferFault.message;return 1;}
    }
    ++b->gasStageIndex;return 0;
}
bool loadGasWindowState(Backend& b,const HostState& snapshot,std::string& error){
    b.storage.beginOperation();
    b.microGeometryPrepared=false;b.preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(b);
    if(b.gasBoundaryLayer.enabled&&b.model.physics.meshMotion.policy!=MeshMotionPolicy::Static)b.storage.invalidateWallGeometry();
    if(!b.storage.uploadGeometry(snapshot.gasMesh,false)||!b.storage.clearGeometry()
        ||!b.storage.uploadState(snapshot,error)){
        b.poisoned=b.storage.runtimeFailure();if(error.empty())error=b.storage.error();
        if(error.empty())error="failed to load gas-window geometry/state";return false;
    }
    bindSharedGasStorage(b);
    if(!b.particles.load(b.model,snapshot,b.storage,b.gasCaptureStream,error))return false;
    b.trial=snapshot;error.clear();return true;
}
Backend* createBackend(const ModelConfig& model,const ugkwp::GasModelConfiguration& gas,
 const ugkwp::GasMechanismConfiguration& mechanism,const GasExecutionOptions& options,const HostState& state,std::string& error){
    if(mag(model.physics.gravity)!=0){error="coupled gas gravity source/work ledger is not enabled";return nullptr;}
    if(options.turbulenceModel!=(model.physics.enableSst?3:0))
        std::fprintf(stderr,"CHMT requested physics SST=%d execution turbulence=%d; effective SST=%d\n",
            int(model.physics.enableSst),options.turbulenceModel,int(options.turbulenceModel==3));
    ugkwp::GasCapabilityRequest request;request.mode=gas.mode;request.fluxScheme=options.fluxScheme;request.reconstruction=options.reconstruction;request.limiter=options.limiter;request.timeIntegrator=options.timeIntegrator;request.turbulenceModel=options.turbulenceModel;request.sstWallTreatment=int(model.physics.wallModel.family);request.boundaryLayerModel=int(model.physics.wallModel.model);request.movingGeometry=model.physics.meshMotion.policy!=MeshMotionPolicy::Static;request.particleCoupling=model.physics.enableParticles;
    request.pairedMixtureParticleCoupling=true;
    request.residentParticleGeometry=true;request.residentParticleWallCoupling=true;
    const auto supported=ugkwp::validateGasCapabilities(request);if(!supported){error=supported.message;return nullptr;}
    if(gas.mode!=ugkwp::GasMode::SingleLegacy
        && ((options.fluxScheme!=1&&options.fluxScheme!=2)||(options.reconstruction!=0&&options.reconstruction!=1))) {
        error="selected mixture gas operator is unavailable";return nullptr;
    }
    // Outlined calls in CUDA fast-compile builds may exceed the default
    // per-thread stack. Match the existing gasUGKP opt-in runtime control;
    // normal release builds retain the CUDA default.
    if(const char* requested=std::getenv("CHMT_CUDA_STACK_BYTES")){
        char* end=nullptr;errno=0;
        const auto bytes=std::strtoull(requested,&end,10);
        if(errno||end==requested||*end||requested[0]<'0'||requested[0]>'9'
            ||bytes==0||bytes>std::numeric_limits<std::size_t>::max()){
            error="invalid CHMT_CUDA_STACK_BYTES (expected positive byte count)";return nullptr;
        }
        std::size_t current=0;auto code=cudaDeviceGetLimit(&current,cudaLimitStackSize);
        if(code==cudaSuccess&&bytes>current)code=cudaDeviceSetLimit(cudaLimitStackSize,static_cast<std::size_t>(bytes));
        if(code!=cudaSuccess){error=std::string("configure CHMT CUDA stack: ")+cudaGetErrorString(code);return nullptr;}
    }
    std::unique_ptr<Backend> b(new Backend);b->model=model;b->options=options;b->accepted=b->trial=state;
    if(!b->executionStream.create(b->executionFault)){error=b->executionFault.message;return nullptr;}
    b->gasCaptureStream=b->executionStream.get();b->storage.bindStream(b->gasCaptureStream);
    const auto diagnosticInit=b->diagnostics.initialise(b->gasCaptureStream);
    if(diagnosticInit!=cudaSuccess){error=std::string("allocate gas diagnostics: ")+cudaGetErrorString(diagnosticInit);return nullptr;}
    ugkp::diagnostics::reset(b->hostDiagnostics,state.commitSequence,b->diagnostics.instanceId(),state.acceptedSteps,state.time);
    b->model.physics.diagnostics=&b->hostDiagnostics;
    b->model.physics.tolerances.diagnostics=&b->hostDiagnostics;
    b->storage.bindDiagnostics(b->diagnostics);
    b->model.physics.enableSst=options.turbulenceModel==3;
    if(!b->storage.configure(b->model,gas,mechanism,state,error))return nullptr;
    std::vector<int> faceSlots(state.gasMesh.owner.size(),-1);
    for(std::size_t i=0;i<state.surface.gasFace.size();++i)faceSlots[state.surface.gasFace[i]]=int(i);
    std::vector<int> cells;for(int face:state.surface.gasFace)cells.push_back(state.gasMesh.owner[face]);
    std::sort(cells.begin(),cells.end());cells.erase(std::unique(cells.begin(),cells.end()),cells.end());
    if(!b->interfaceCells.resize(cells.size(),b->surfaceTransferFault)
        ||!b->storage.write(b->interfaceCells.data(),cells)
        ||!b->interfaceFaceSlots.resize(faceSlots.size(),b->surfaceTransferFault)
        ||!b->storage.write(b->interfaceFaceSlots.data(),faceSlots)
        ||!b->interfaceProgress.resize(state.gasMesh.volumes.size(),b->surfaceTransferFault)
        ||!b->interfaceStageWall.resize(state.surface.gasFace.size(),b->surfaceTransferFault)
        ||!b->interfaceWallChanged.resize(b->storage.hostView().gasBoundaryLayer.count,b->surfaceTransferFault)
        ||!b->interfaceFaces.resize(state.surface.gasFace.size(),b->surfaceTransferFault)
        ||!b->storage.write(b->interfaceFaces.data(),state.surface.gasFace)
        ||!b->interfaceInputs.resize(state.surface.gasFace.size(),b->surfaceTransferFault)
        ||!b->interfaceRecords.resize(state.surface.gasFace.size(),b->surfaceTransferFault)
        ||!b->interfaceDownloads.resize(state.surface.gasFace.size(),b->surfaceTransferFault)
        ||!b->interfaceWallDownloads.resize(b->storage.hostView().gasBoundaryLayer.enabled?state.surface.gasFace.size():0,b->surfaceTransferFault)
        ||!b->interfacePhysics.resize(1,b->surfaceTransferFault)){
        error=b->surfaceTransferFault.message;return nullptr;
    }
    auto devicePhysics=b->model.physics;devicePhysics.diagnostics=b->diagnostics.device();
    devicePhysics.tolerances.diagnostics=b->diagnostics.device();
    if(!b->storage.write(b->interfacePhysics.data(),std::vector<PhysicsConfig>{devicePhysics})){
        error=b->storage.error();return nullptr;
    }
    auto& v=b->storage.hostView();v.gasFluxScheme=options.fluxScheme;v.gasReconstruction=options.reconstruction;v.gasLimiter=options.limiter;v.turbulenceModel=options.turbulenceModel;v.sstConfigured=options.turbulenceModel==3;
    if(!b->storage.refreshView()){error=b->storage.error();return nullptr;}
    bindSharedGasStorage(*b);
    if(!b->particles.create(b->model,state,b->storage,b->gasCaptureStream,error))return nullptr;
    if(b->accepted.gasTemperature.empty()){
        if(!b->storage.read(v.Tgas,b->nCells,b->accepted.gasTemperature)){error=b->storage.error();return nullptr;}
        b->trial.gasTemperature=b->accepted.gasTemperature;
    }
    b->hostTurbulenceModel=options.turbulenceModel;b->hostGasFluxScheme=options.fluxScheme;b->hostGasTimeIntegrator=options.timeIntegrator;
    for(auto kind:state.gasMesh.boundaryKind)if(kind==BoundaryKind::Periodic)b->hasPeriodicFaces=1;
    error.clear();return b.release();
}
void destroyBackend(Backend* backend)noexcept{delete backend;}
static bool dispatchHostDiagnostics(Backend&);
void setDiagnosticsSink(Backend& b,const ugkp::diagnostics::HostSink& sink){
    b.diagnosticSink=sink;b.diagnostics.setSink(sink);
    dispatchHostDiagnostics(b);
}
static bool dispatchHostDiagnostics(Backend& b){
    const auto snapshot=ugkp::diagnostics::takeSnapshot(b.hostDiagnostics,ugkp::diagnostics::sourceCatalogue(),"ugkp-diagnostics-v1");
    if(!b.diagnosticSink.fatal&&!b.diagnosticSink.warning)return !snapshot.hasFatal;
    ugkp::diagnostics::dispatch(snapshot,b.diagnosticSink);
    if(snapshot.hasFatal)return false;
    ugkp::diagnostics::reset(b.hostDiagnostics,b.hostDiagnostics.operationSequence,b.diagnostics.instanceId(),b.hostDiagnostics.step,b.hostDiagnostics.simulationTime);
    return true;
}
static bool beginGasWindowImpl(Backend& b,const WallProgram& program,std::string& error){
    if(b.poisoned){error="gas backend is unusable after a failed device operation";return false;}
    b.trial=b.accepted;b.program=program;
    std::vector<WallFaceSample> faces;
    for(const auto& knot:program.knots)faces.insert(faces.end(),knot.faces.begin(),knot.faces.end());
    if(!b.interfaceWallProgram.resize(faces.size(),b.surfaceTransferFault)
        ||!b.storage.write(b.interfaceWallProgram.data(),faces)){
        error=b.surfaceTransferFault.message.empty()?b.storage.error():b.surfaceTransferFault.message;return false;
    }
    b.interfaceGeometryReady=false;
    if(!b.history.begin(program.interval,error)||!b.reserve.reset(b.accepted.solid,b.accepted.film,b.accepted.time,error))return false;
    if(!program.donorPlan.knots.empty()&&!b.reserve.setPlan(program.donorPlan,error))return false;
    if(!loadGasWindowState(b,b.accepted,error))return false;b.elapsed=WindowTime{};b.nextGasSeed=b.accepted.nextDt>0?b.accepted.nextDt:b.accepted.lastAcceptedDt;return true;
}

bool beginGasWindow(Backend& b,const WallProgram& program,std::string& error){
    ugkp::diagnostics::beginOperation(b.hostDiagnostics,program.interval.identity.sequence,b.accepted.acceptedSteps,program.interval.begin);
    const bool success=beginGasWindowImpl(b,program,error);
    const bool healthy=dispatchHostDiagnostics(b);
    if(!healthy&&error.empty())error="fatal host gas-window diagnostic";
    return success&&healthy;
}
struct CoupledCourantMaximum {
    __host__ __device__ double operator()(double a,double b)const{return a!=a?a:b!=b?b:fmax(a,b);}
};
struct CoupledCourantValue {double value;bool unavailable;unsigned char initializedTail[7]{};};
struct CoupledCourantRead {
    const double* field;const int* status;const ugkp::diagnostics::DeviceContext* diagnostics;
    __host__ __device__ CoupledCourantValue operator()(int cell)const{
        // This is the reduction identity, never a replacement physical state.
        // The host discards the reduction after observing the fatal latch.
        if(ugkp::diagnostics::hasFatal(diagnostics))return {-std::numeric_limits<double>::infinity(),false};
        return {field[cell],status&&status[cell]==int(ugkwp::GasTransportCode::InvalidStorage)};
    }
};
struct CoupledCourantReduce {
    __host__ __device__ CoupledCourantValue operator()(CoupledCourantValue a,CoupledCourantValue b)const{
        return {CoupledCourantMaximum{}(a.value,b.value),a.unavailable||b.unavailable};
    }
};
static bool currentGasCourant(Backend& b,Real referenceDt,transport::GasCourantMaxima& result){
    const auto& v=b.storage.hostView();
    struct View {
        int nCells,nFaces,nInternalFaces;SharedGasDeviceView* deviceState;
        Real* gasFluxPositivityScale;
    } view{b.nCells,b.nFaces,b.nInternalFaces,b.deviceState,v.gasFluxPositivityScale};
    const auto maximum=[&](const double* field,int count,cudaStream_t stream,double& value){
        try {
            const auto first=thrust::make_counting_iterator(0);
            const auto reduced=thrust::transform_reduce(thrust::cuda::par(b.courantReductionStorage).on(stream),first,first+count,
                CoupledCourantRead{field,v.gasSpecies.cellStatus,b.diagnostics.device()},
                CoupledCourantValue{-std::numeric_limits<double>::infinity(),false},CoupledCourantReduce{});
            if(!b.storage.observeDiagnostics())return 1;
            if(reduced.unavailable){transport::setLastErrorText("current Co consumed input unavailable");return 1;}
            value=reduced.value;
        }
        catch(const std::exception& e){transport::setLastErrorText(e.what());return 1;}
        const auto code=cudaGetLastError();if(code!=cudaSuccess){transport::setLastError("gas Courant maximum reduction",code);return 1;}return 0;
    };
    return transport::computeGasCurrentCourant(&view,referenceDt,b.microTime,b.fixedCellBlockThreads,b.fixedFaceBlockThreads,b.gasCaptureStream,result,maximum)==0;
}
static bool advanceGasMicrostepImpl(Backend& b,Real cap,GasMicroReport& report,std::string& error){
    report={};report.recoverable=false;
    if(b.poisoned){error="gas backend is unusable after a failed device operation";return false;}
    if(b.history.complete()||b.elapsed==intervalDuration(b.program.interval)){
        report.time=b.trial.time;report.localTime=b.elapsed;report.nextGasDt=b.nextGasSeed;
        report.microSequence=b.history.records().size();error.clear();return true;
    }
    b.microTime=windowAbsolute(b.program.interval.begin,b.elapsed);
    const auto diagnosticSequence=b.history.records().size()+1;
    // acceptedSteps is the zero-based completed global-step count. This launch
    // precedes all producers on their stream and adds no synchronization/copy.
    ugkp::diagnostics::beginOperation(b.hostDiagnostics,diagnosticSequence,b.trial.acceptedSteps,b.microTime);
    ugkp::diagnostics::beginOperationKernel<<<1,1,0,b.gasCaptureStream>>>(b.diagnostics.device(),diagnosticSequence,b.trial.acceptedSteps,b.microTime);
    const auto metadataLaunch=cudaGetLastError();
    if(metadataLaunch!=cudaSuccess){error=std::string("gas diagnostic metadata launch: ")+cudaGetErrorString(metadataLaunch);return false;}
    const Real remaining=boundedWindowDuration(nextGasWallKnotLocal(b.program,b.elapsed),b.elapsed);
    if(!finite(cap)||!finite(b.model.physics.maxDt)||!finite(remaining)||!finite(b.nextGasSeed)||b.nextGasSeed<0
        ||!finite(b.trial.lastAcceptedDt)||b.trial.lastAcceptedDt<0){error="invalid gas timestep input or seed";return false;}
    Real referenceDt=minValue(cap,minValue(b.model.physics.maxDt,remaining));
    if(b.nextGasSeed>0)referenceDt=minValue(referenceDt,b.nextGasSeed);
    if(!finite(referenceDt)||!(referenceDt>0)||b.elapsed+referenceDt==b.elapsed){error="gas reference timestep is nonfinite, nonpositive or nonadvancing";return false;}
    b.microDt=referenceDt;b.error.clear();transport::lastError.clear();
    if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}
    transport::GasCourantMaxima co;
    if(!currentGasCourant(b,referenceDt,co)){
        error=b.error.empty()?transport::lastError:b.error;return false;
    }
    const Real growthBase=b.trial.lastAcceptedDt>0?b.trial.lastAcceptedDt:referenceDt;
    const Real selectedDt=gasCoMaxStep(referenceDt,b.model.physics.cfl,co.maximum,growthBase);


    if(!finite(selectedDt)||!(selectedDt>0)||b.elapsed+selectedDt==b.elapsed){error="gas CFL-selected timestep is nonfinite, nonpositive or nonadvancing";return false;}
    const Real dt=selectedDt;
    if(dt!=referenceDt){b.microDt=dt;if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}}
    b.microGeometryPrepared=true;
    if(transport::advanceGasTrial<transport::CoupledTrialPolicy>(&b,dt,b.microTime)!=0){
        error=b.error.empty()?transport::lastError:b.error;return false;
    }
    MaterialDonorReserve::Transaction reservation;
    if(!b.reserve.prepare(b.record,reservation,b.error)){error=b.error;return false;}
    if(!b.history.appendAccepted(b.record,error))return false;
    b.reserve.commitPrepared(std::move(reservation));b.elapsed=b.record.localEnd;
    if(b.model.physics.meshMotion.policy!=MeshMotionPolicy::Static){
        b.trial.gasStages={{b.gasStage,b.gasStage}};b.trial.solidStages={{b.solidStage,b.solidStage}};
    }
    b.nextGasSeed=minValue(b.model.physics.maxDt,1.05*dt);b.trial.nextDt=b.nextGasSeed;b.trial.lastAcceptedDt=dt;
    report.acceptedDt=dt;report.nextGasDt=b.nextGasSeed;report.time=b.trial.time;report.localTime=b.elapsed;report.microSequence=b.record.microSequence;error.clear();return true;
}
bool advanceGasMicrostep(Backend& b,Real cap,GasMicroReport& report,std::string& error){
    const bool success=advanceGasMicrostepImpl(b,cap,report,error);
    const bool hostHealthy=dispatchHostDiagnostics(b);
    if(!hostHealthy && error.empty())error="fatal host gas diagnostic";
    return success&&hostHealthy;
}
const IntervalHistory& gasWindowHistory(const Backend& b){return b.history;}
bool gasWindowStageGeometry(const Backend& b,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error){
    if(b.model.physics.meshMotion.policy==MeshMotionPolicy::Static){
        const auto snapshot=[&](const HostMesh& mesh,std::array<HostStageGeometry,2>& stages){
            auto& stage=stages[0];stage.interval=b.microDt;stage.geometryVersion=mesh.geometryVersion;stage.topologyHash=mesh.topologyHash;
            stage.oldVolume=stage.newVolume=stage.evaluationVolume=mesh.volumes;
            stage.sweptVolume.assign(mesh.owner.size(),0);stage.areaVector=mesh.areaVectors;
            stage.cellCentre=mesh.cellCentres;stage.faceCentre=mesh.faceCentres;stage.oldPoints=stage.newPoints=mesh.points;
            stages[1]=stage;
        };
        snapshot(b.trial.gasMesh,gas);snapshot(b.trial.solidMesh,solid);
    }else{gas=b.trial.gasStages;solid=b.trial.solidStages;}
    error.clear();return true;
}
bool downloadGasWindow(Backend& b,HostState& out,std::string& error){
    if(!b.storage.downloadState(b.trial,error)||!b.particles.download(b.trial,error))return false;
    out=b.trial;return gasWindowStageGeometry(b,out.gasStages,out.solidStages,error);
}
static bool commitGasWindowImpl(Backend& b,HostState& candidate,std::string& error){
    if(b.poisoned){error="gas backend is unusable after a failed device operation";return false;}
    // Publish the synchronized material/gas endpoint on the host. The next
    // beginGasWindow imports it once; an immediate upload here was overwritten
    // by that same import before any device consumer could use it.
    // The material owner may correct the predicted macro trajectory. Map the
    // actual accepted endpoint too, before publishing particles/checkpoints.
    // A microstep-only map would otherwise leave attached parcels on the old
    // predicted wall and free parcels associated with displaced internal cells.
    bool corrected=b.trial.gasMesh.points.size()!=candidate.gasMesh.points.size();
    for(std::size_t i=0;!corrected&&i<candidate.gasMesh.points.size();++i){
        const auto a=b.trial.gasMesh.points[i],c=candidate.gasMesh.points[i];
        corrected=a.x!=c.x||a.y!=c.y||a.z!=c.z;
    }
    if(corrected&&b.particles.active()){
        if(!b.particles.mapGeometry(b.trial.gasMesh,candidate.gasMesh,b.storage,b.gasCaptureStream,error))return false;
        if(!b.storage.uploadGeometry(candidate.gasMesh,false)||!b.storage.refreshView()){
            error=b.storage.error();return false;
        }
        bindSharedGasStorage(b);
        if(!b.particles.refreshGeometry(b.storage,b.gasCaptureStream,error)
            ||!b.particles.result(ugkp::thermal::relocateMobileParticles(b.particles.owner(),b.gasCaptureStream),error))return false;
        if(!b.storage.observeDiagnostics()){error=b.storage.error();return false;}
        if(!b.particles.download(candidate,error))return false;
    }
    HostState newAccepted=candidate,newTrial=candidate;
    b.accepted=std::move(newAccepted);b.trial=std::move(newTrial);error.clear();return true;
}
bool commitGasWindow(Backend& b,HostState& candidate,std::string& error){
    const bool success=commitGasWindowImpl(b,candidate,error);
    const bool healthy=dispatchHostDiagnostics(b);
    if(!healthy&&error.empty())error="fatal host gas-window commit diagnostic";
    return success&&healthy;
}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;error.clear();return true;}
}
