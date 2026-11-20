#include "gpu/GasTimeStepControl.H"
#include "../../../common/thermal/ParticleRadiationApi.H"
#include "../../../common/gpu/WorkPlan.H"
#include "gasNumerics/GasWindowStep.H"
#include "gpu/Backend.H"
#include "materials/MaterialWindowExchange.H"
#include "gpu/AcceptedStateSnapshot.H"
#include "../../../common/gpu/FrozenReadback.cuh"
#include "gpu/MaterialBackend.H"
#include "../../../common/thermal/ThermalEventApi.H"
#include "../../../common/gpu/DiscreteOrdinateRadiation.H"
#include "../../../common/gpu/ParticleRadiationMaterial.H"
#include "gpu/CoupledHistoryArchive.cuh"
#include "gpu/GasWindowControl.cuh"
#include "../../../common/gpu/ExecutionGraph.cuh"
#include "gpu/SharedGasDeviceStorage.cuh"
#include "gpu/ParticleCouplingStorage.cuh"
#include "gpu/CourantReductionStorage.cuh"
#include "gpu/CoupledSurfaceTransfer.cuh"
#include "gpu/GasWallMath.H"
#include "gpu/GasWindowProgram.H"
#include "gpu/GasWindowGraphStorage.H"
#include "gpu/CoupledSurfacePipeline.cuh"
#include "gpu/ResidentCoupledGeometry.cuh"
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
#include "CoupledWindowCommit.cuh"
struct HostGasPublication {GasQ gas;SstQ sst;};
struct Backend {
    CudaFault executionFault;
    DeviceStream executionStream;
    ugkp::diagnostics::GpuOwner diagnostics;
    ugkp::diagnostics::DeviceContext hostDiagnostics{};
    ugkp::diagnostics::HostSink diagnosticSink;
    ModelConfig model;GasExecutionOptions options;HostState accepted,trial;
    bool residentAcceptedEndpoint=false,gasEndpointPublished=false,hostEndpointCurrent=true;
    Buffer<int> hostGasCells;
    Buffer<HostGasPublication> hostGasValues;
    std::vector<int> hostGasCellAddressing;
    std::vector<HostGasPublication> hostGasStaging;
    std::unique_ptr<WallClosureHost> hostWallGeometry;
    std::vector<GpuReal> hostGasFraction;
    Buffer<Real> acceptedOldPhaseVolume,acceptedNewPhaseVolume;
    Buffer<AcceptedWindowSlot> coupledSlot;
    Buffer<WindowReceipt> coupledReceipt;
    bool hasCoupledReceipt=false;
    std::uint64_t endpointGasVersion=0,endpointSolidVersion=0;
    SharedGasDeviceStorage storage;ParticleCouplingStorage particles;WallProgram program;IntervalHistory history;
    CoupledCourantStorage courantReductionStorage;
    std::unique_ptr<ResidentCoupledGeometry> residentGeometry;
    CudaFault surfaceTransferFault;
    Buffer<int> interfaceFaces,interfaceFaceSlots,interfaceCells;
    Buffer<EnergyReal> windowRadiation;
    Buffer<PhysicalReal> interfaceProgress;
    Buffer<unsigned char> interfaceWallChanged;
    Buffer<WallFaceSample> interfaceStageWall;
    Buffer<CoupledSurfaceInput> interfaceInputs;
    Buffer<WallFaceSample> interfaceWallProgram;
    GasWallProgramPosition stageWallPosition{};
    Buffer<CoupledGasInterval> coupledInterval;
    Buffer<ResidentWallKnotTime> interfaceKnotTime;Buffer<ResidentWallClock> wallClock;
    bool interfaceGeometryReady=false;
    Buffer<CoupledSurfaceRecord> interfaceRecords;
    CoupledHistoryArchive::TransferPool historyTransfers;
    std::shared_ptr<CoupledHistoryArchive> historyArchive;
    Buffer<CoupledRecordFailure> interfaceFailure;
    std::size_t historyFrameCount=0,historyPublishedFrames=0;
    Buffer<ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns>> interfaceWallDiagnostics;
    std::vector<ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns>> wallDiagnosticStaging;
    bool wallDiagnosticsReady=false;Real wallDiagnosticTime=0;
    Buffer<PhysicsConfig> interfacePhysics;

    using WindowGraph=chmt::WindowGraph;
    WindowGraph windowGraph[2];
    Buffer<GasWindowControls> windowControls;Buffer<GasWindowProgress> windowProgress;
    Buffer<GasWindowCourant> windowCourant;
    Buffer<ResidentWallClock> windowGeometryClock;
    Buffer<ugkwp::ParticleResidentInterval> windowParticleInterval;
    Buffer<ugkp::thermal::BoundParticleMirror> windowParticleMirror;
    CoupledHistoryArchive::Batch windowBatch{};
    bool windowBatchPrepared=false,windowBatchPending=false,windowBatchRetired=false,windowMaterialHistoryReady=false;
    MaterialBackend* windowMaterial=nullptr;
    ugkp::gpu::DiscreteOrdinateRadiation* windowTransport=nullptr;
    ugkp::gpu::ParticleRadiationMaterial* windowRadiationMaterial=nullptr;
    Buffer<int> windowRadiationStage;
    Buffer<double> windowRadiationElapsed;
    bool poisoned=false;std::string error;
    GasIntervalRecord record;
    Real microDt=0,microTime=0,stageTime=0,stagePacketDt=0;
    bool movingWindow=false,residentTrajectoryAdvanced=false,acceptedGeometryChanged=false;
    std::uint64_t gasSteps=0;Real cachedCourantRate=0;bool courantReady=false;
    WindowTime elapsed{};Real nextGasSeed=0;bool microGeometryPrepared=false;int gasStageIndex=0;
    int nCells=0,nFaces=0,nInternalFaces=0,fixedCellBlockThreads=ugkp::gpu::workPlan().cellBlockThreads,fixedFaceBlockThreads=ugkp::gpu::workPlan().faceBlockThreads;
    int hostTurbulenceModel=0,hostGasFluxScheme=1,hostGasTimeIntegrator=1,hasPeriodicFaces=0;
    ugkwp::GasSpeciesState<PhysicalReal,Ns> gasSpecies;
    ugkwp::GasGeometryState<Real> gasGeometry;
    ugkwp::GasBoundaryLayerState<PhysicalReal> gasBoundaryLayer;
    Real gasBoundaryLayerStageTime=0;
    Real preparedWallTime=0,preparedWallDt=0;bool preparedWallReady=false;
    SharedGasDeviceView* deviceState=nullptr;cudaStream_t gasCaptureStream=nullptr;
    cudaGraph_t gasGraph=nullptr;cudaGraphExec_t gasGraphExec=nullptr;
    cudaGraphNode_t gasGraphIntervalNode=nullptr;cudaKernelNodeParams gasGraphIntervalParams{};
    ugkwp::GasResidentInterval gasGraphLastInterval{};
    ugkwp::GasResidentInterval* gasGraphClock=nullptr;
    bool gasGraphIntervalSubmitted=false,gasGraphAdvancesInterval=false;
    bool gasGraphIncludesBoundary=false,gasGraphPreparedBoundary=false,gasGraphFrozenBoundary=false;
    Real gasGraphDt=-1;int gasGraphMode=-1;
    std::vector<const void*> graphPointers;std::vector<int> graphShape;
    ~Backend(){
        if(gasCaptureStream)executionStream.sync();
        if(gasGraphExec)cudaGraphExecDestroy(gasGraphExec);
        if(gasGraph)cudaGraphDestroy(gasGraph);
        if(gasGraphClock)cudaFree(gasGraphClock);
    }
};
inline void bindSharedGasStorage(Backend& b){
    const auto& view=b.storage.hostView();
    b.preparedWallReady=false;
    if(b.deviceState!=b.storage.deviceView()||b.nCells!=view.nCells
        ||b.nFaces!=view.nFaces||b.nInternalFaces!=view.nInternalFaces)
        b.gasGraphDt=std::numeric_limits<Real>::quiet_NaN();
    b.gasSpecies=view.gasSpecies;b.gasBoundaryLayer=view.gasBoundaryLayer;b.gasGeometry=view.gasGeometry;
    b.deviceState=b.storage.deviceView();b.nCells=view.nCells;b.nFaces=view.nFaces;b.nInternalFaces=view.nInternalFaces;
}


inline void validateCoupledGraphBindings(Backend& b){
    const auto& model=b.storage.hostView().gasBoundaryLayerModel;
    const std::vector<const void*> pointers={b.deviceState,b.interfaceFaces.data(),b.interfaceCells.data(),
        b.interfaceFaceSlots.data(),b.interfaceInputs.data(),b.interfaceWallProgram.data(),
        b.interfaceProgress.data(),b.interfaceStageWall.data(),b.interfaceWallChanged.data(),
        b.interfacePhysics.data(),b.interfaceRecords.data(),b.coupledInterval.data()};
    const std::vector<int> shape={b.nCells,b.nFaces,b.nInternalFaces,b.fixedCellBlockThreads,
        b.fixedFaceBlockThreads,b.hostTurbulenceModel,b.hostGasFluxScheme,b.hostGasTimeIntegrator,
        b.hasPeriodicFaces,int(b.interfaceFaces.size()),int(b.interfaceCells.size()),
        int(b.gasBoundaryLayer.enabled),b.gasBoundaryLayer.count,model.workspaceCount,
        model.workspaceCapacity,int(model.config.model),model.config.nodes,int(model.conservative.enabled),
        int(b.gasSpecies.mode),int(b.model.physics.material.enableSurfaceReactions),
        int(b.model.physics.material.enablePoreOutflow)};
    if(pointers!=b.graphPointers||shape!=b.graphShape){
        b.gasGraphDt=std::numeric_limits<Real>::quiet_NaN();
        b.graphPointers=pointers;b.graphShape=shape;
    }
}
inline WindowTime coupledStageTime(const Backend& b){return b.elapsed+(b.gasStageIndex==0?0:b.gasStageIndex==1?b.microDt:.5*b.microDt);}
bool loadGasWindowState(Backend&,const HostState&,std::string&);
int prepareCoupledGeometry(Backend*);
int finishCoupledGeometry(Backend*);
int advanceCoupledParticles(Backend*,Real);
int prepareCoupledBoundaryLayer(Backend*,Real,Real);
int applyCoupledFaces(Backend*,Real,Real);
int accountCoupledFaces(Backend*,Real);
bool archiveCoupledRecord(Backend&);
namespace transport {
using std::isfinite;
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
constexpr GpuReal OfSmall=GPU_R(2.22044604925031308085e-16),OfVSmall=GPU_REAL_MIN,OfGreat=GPU_R(1)/OfSmall;
struct GasPrimDevice {GpuReal rho,ux,uy,uz,p,T;};
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
}
}
#include "../../../common/GpuGasHostPolicy.cuh"
namespace chmt { namespace transport {
thread_local std::string lastError;
inline void setLastError(const char* text,cudaError_t code){lastError=std::string(text)+": "+cudaGetErrorString(code);}
inline void setLastErrorText(const char* text){lastError=text;}
struct GasHostPolicy:GasHostWithWallEnergy<Real,Real,&accountCoupledFaces> {
    static int prepareBoundaryLayer(Backend* s,Real dt,Real time){return prepareCoupledBoundaryLayer(s,dt,time);}
    static int applyFaceSources(Backend* s,Real dt,Real time){return applyCoupledFaces(s,dt,time);}
    static void prepareGraphCapture(Backend* b,Real dt,Real time,bool prepared){


        b->preparedWallReady=prepared&&b->preparedWallReady
            &&b->preparedWallTime==time&&b->preparedWallDt==dt;
        if(b->preparedWallReady){b->preparedWallTime=0;b->preparedWallDt=1;}
        b->gasStageIndex=0;
    }
    static void publishGraphReplay(Backend* b,Real dt,Real){
        const int stages=b->hostGasTimeIntegrator==1?1:b->hostGasTimeIntegrator==2?2:3;
        b->gasStageIndex=stages;
        const WindowTime local=b->elapsed+(stages==1?0:stages==2?dt:.5*dt);
        b->stageTime=windowAbsolute(b->program.interval.begin,local);
        b->stageWallPosition={};b->stagePacketDt=dt;
        b->preparedWallReady=false;
    }
};
#include "../../../common/GpuGasAdvance.cuh"
#include "../../../common/GpuGasCourant.cuh"
struct CoupledTrialPolicy {
    static int begin(Backend* b){
        b->preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(*b);
        b->gasStageIndex=0;b->record={};b->record.microSequence=b->history.records().size()+1;
        b->record.begin=b->microTime;b->record.actualDt=b->microDt;b->record.localBegin=b->elapsed;b->record.localEnd=b->elapsed+b->microDt;b->record.end=windowAbsolute(b->program.interval.begin,b->record.localEnd);b->record.gasGeometry=b->trial.gasMesh.geometryVersion;b->record.solidGeometry=b->trial.solidMesh.geometryVersion;
        if(!b->storage.clearTrialStatus())return 1;
        if(!b->microGeometryPrepared&&prepareCoupledGeometry(b)!=0)return 1;
        if(b->movingWindow){
            b->record.gasGeometry=b->endpointGasVersion;b->record.solidGeometry=b->endpointSolidVersion;
        }
        b->microGeometryPrepared=false;
        advanceCoupledWallClock<<<1,1,0,b->gasCaptureStream>>>(b->wallClock.data(),b->interfaceKnotTime.data(),
            int(b->interfaceKnotTime.size()),b->microDt,b->record.gasGeometry,b->coupledInterval.data());
        if(!b->surfaceTransferFault.check(cudaGetLastError(),"advance resident coupled wall clock")){
            b->error=b->surfaceTransferFault.message;return 1;
        }
        validateCoupledGraphBindings(*b);
        if(b->particles.active()&&!b->particles.result(ugkp::thermal::prepareParticleGasInterval(
            b->particles.owner(),b->gasCaptureStream),b->error))return 1;
        return 0;
    }
    struct StageVolumes {
        const Real* geometry;const PhysicalReal* physical;
        CHMT_HD explicit operator bool()const{return geometry||physical;}
        CHMT_HD PhysicalReal operator[](int i)const{return geometry?PhysicalReal(geometry[i]):physical[i];}
    };
    static StageVolumes stageVolumes(Backend* b,bool after){const auto& v=b->storage.hostView();const auto& g=v.gasGeometry;return {g.enabled?(after?g.newVolume:g.oldVolume):nullptr,g.enabled?nullptr:v.V};}

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
        if(!archiveCoupledRecord(*b))return 1;
        const auto& model=b->storage.hostView().gasBoundaryLayerModel;
        if(model.acceptedHistory&&b->gasBoundaryLayer.count){
            ::commitAcceptedWallHistory<<<(b->gasBoundaryLayer.count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(b->deviceState);
            const auto error=cudaGetLastError();if(error!=cudaSuccess){b->error=cudaGetErrorString(error);return 1;}
        }


        b->trial.time=b->record.end;return 0;
    }

};
}}
namespace chmt {





bool archiveCoupledRecord(Backend& b){
    const int count=int(b.interfaceRecords.size());
    if(!b.historyArchive)b.historyArchive=std::make_shared<CoupledHistoryArchive>(count,b.gasBoundaryLayer.enabled,&b.historyTransfers);
    const auto frame=b.historyArchive->next(b.gasCaptureStream);auto* records=frame.surface;auto* walls=frame.wall;
    archiveMaterialDriveInterval<<<1,1,0,b.gasCaptureStream>>>(b.coupledInterval.data(),frame.interval);
    enqueueCoupledSurfaceArchive(b.interfaceRecords.data(),count,b.interfaceFaces.data(),b.particles.wallExchange(),
        {records,walls,nullptr},b.interfaceFailure.data(),int(b.accepted.solid.size()),int(b.accepted.film.size()),
        b.diagnostics.device(),b.fixedFaceBlockThreads,b.gasCaptureStream);
    if(!b.surfaceTransferFault.check(cudaGetLastError(),"archive accepted coupled exchange")){
        b.error=b.surfaceTransferFault.message;return false;
    }
    b.historyArchive->accepted(b.gasCaptureStream);
    b.record.gasTraceTime=b.stageTime;++b.historyFrameCount;
    if(b.gasBoundaryLayer.enabled){b.wallDiagnosticsReady=true;b.wallDiagnosticTime=b.stageTime;}
    return true;
}
void populateCoupledHostRecord(bool particles,bool boundaryLayer,const CoupledSurfaceDownload* records,
    const CoupledWallDownload* walls,std::size_t count,GasIntervalRecord& record){
    record.packets.clear();record.gasTrace.clear();record.gasGradient.clear();
    record.gasAcceptedWall.clear();record.gasTraction.clear();record.radiationEnergy.clear();
    record.radiationReceiver.clear();record.gasWallMatching.clear();
    for(std::size_t i=0;i<count;++i){const auto& r=records[i];
        appendCoupledHostPackets(record.packets,r,particles);
        record.gasTrace.push_back(r.bulk);record.gasGradient.push_back(r.gradient);
        record.gasAcceptedWall.push_back(r.acceptedWall);
        record.gasTraction.push_back(r.traction);
        record.radiationEnergy.push_back(r.radiationEnergy);record.radiationReceiver.push_back(r.radiationReceiver);
        if(boundaryLayer)record.gasWallMatching.push_back(walls[i].matching);
    }
}
int prepareCoupledGeometry(Backend* b){
    const auto& base=b->trial;
    if(b->movingWindow){
        if(!b->residentGeometry->prepare(b->microDt,b->interfaceKnotTime.data(),int(b->interfaceKnotTime.size()),b->wallClock.data(),
            b->interfaceFaces.data(),b->interfaceInputs.data(),b->storage,b->gasCaptureStream,b->error))return 1;
        b->endpointGasVersion=base.gasMesh.geometryVersion+1;
        b->endpointSolidVersion=base.solidMesh.geometryVersion+1;
        b->record.gasGeometry=b->endpointGasVersion;b->record.solidGeometry=b->endpointSolidVersion;
    }else{
        if(!b->storage.clearGeometry()){b->error=b->storage.error();return 1;}
        b->record.gasGeometry=base.gasMesh.geometryVersion;b->record.solidGeometry=base.solidMesh.geometryVersion;
        if(!b->interfaceGeometryReady){
            const auto& surface=base.surface;std::vector<CoupledSurfaceInput> inputs(surface.area.size());
            for(std::size_t i=0;i<inputs.size();++i){const int f=surface.gasFace[i];auto& in=inputs[i];
                in.area=surface.area[i];in.gasDistance=surface.gasDistance[i];in.normal=toPhysical(surface.normal[i]);
                const PhysicalVec3 support=surface.baseVelocity.empty()?PhysicalVec3{}:surface.baseVelocity[i];
                const PhysicalVec3 motion{};in.particleMaterialVelocity=support+in.normal*dot(motion-support,in.normal);
                in.identity.face=surface.persistentId[i];in.identity.stage=1;in.identity.gasCell=base.gasMesh.owner[f];
                in.identity.solidCell=surface.solidCell[i];in.identity.filmFace=int(i);
            }
            if(!b->storage.write(b->interfaceInputs.data(),inputs)){b->error=b->storage.error();return 1;}
            b->interfaceGeometryReady=true;
        }
    }
    b->gasGeometry=b->storage.hostView().gasGeometry;return 0;
}
int finishCoupledGeometry(Backend* b){
    if(!b->movingWindow)return 0;
    auto& geometry=*b->residentGeometry;
    if(!b->particles.mapGeometry(geometry.particleGeometry(b->storage,false),geometry.particleGeometry(b->storage,true),
        b->gasCaptureStream,b->error)||!geometry.finish(b->storage,b->gasCaptureStream,b->error))return 1;
    b->trial.gasMesh.geometryVersion=b->endpointGasVersion;b->trial.solidMesh.geometryVersion=b->endpointSolidVersion;
    bindSharedGasStorage(*b);
    if(!b->particles.refreshTrackingGeometry(b->storage,b->gasCaptureStream,b->error))return 1;
    return 0;
}
int advanceCoupledParticles(Backend* b,Real dt){
    if(!b->particles.active())return 0;
    const int count=b->interfaceFaces.size();
    if(count){const auto surface=b->particles.surface();
        publishParticleMaterialSurfaceKernel<<<(count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(count,
            b->interfaceFaces.data(),b->interfaceInputs.data(),b->interfaceWallProgram.data(),GasWallProgramPosition{},
            const_cast<GpuReal*>(surface.temperature),const_cast<GpuReal*>(surface.velocityX),
            const_cast<GpuReal*>(surface.velocityY),const_cast<GpuReal*>(surface.velocityZ),const_cast<GpuReal*>(surface.effusivity),b->coupledInterval.data());
        if(!b->surfaceTransferFault.check(cudaGetLastError(),"publish resident particle material surface")){
            b->error=b->surfaceTransferFault.message;return 1;}
    }
    return b->particles.result(ugkp::thermal::advanceBoundParticlePhase(b->particles.owner(),dt,
        windowAbsolute(b->program.interval.begin,b->elapsed+dt),b->gasCaptureStream),b->error)?0:1;
}
int prepareResidentSurface(Backend* b,Real dt,bool transportReady){
    const int count=int(b->interfaceFaces.size());if(!count)return 0;
    b->stageWallPosition={};
    const bool reactive=b->model.physics.material.enableSurfaceReactions||b->model.physics.material.enablePoreOutflow;
    if(reactive)boundSurfaceProgressKernel<<<(b->interfaceCells.size()+ugkp::gpu::workPlan().cellBlockThreads-1)/ugkp::gpu::workPlan().cellBlockThreads,ugkp::gpu::workPlan().cellBlockThreads,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceCells.data(),int(b->interfaceCells.size()),b->interfaceFaceSlots.data(),b->interfaceInputs.data(),b->interfaceWallProgram.data(),count,
        b->stageWallPosition,dt,transportReady,b->interfaceProgress.data(),b->coupledInterval.data(),b->gasStageIndex);
    prepareResidentSurfaceKernel<<<(count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceFaces.data(),count,b->interfaceInputs.data(),b->interfaceWallProgram.data(),
        b->stageWallPosition,reactive?b->interfaceProgress.data():nullptr,b->interfaceStageWall.data(),b->interfaceWallChanged.data(),b->coupledInterval.data(),b->gasStageIndex);
    if(!b->surfaceTransferFault.check(cudaGetLastError(),"bound resident surface reaction progress")){
        b->error=b->surfaceTransferFault.message;return 1;}
    return 0;
}
int evaluateResidentSurfaceLayer(Backend* b,Real dt,Real stageTime,bool changedOnly=false){
    const auto& model=b->storage.hostView().gasBoundaryLayerModel;
    const auto error=transport::launchGasBoundaryLayerPool(b->deviceState,model,
        b->gasBoundaryLayer.count,b->gasCaptureStream,changedOnly?b->interfaceWallChanged.data():nullptr);
if(error!=cudaSuccess){b->error=std::string("boundaryLayer CUDA launch: ")+cudaGetErrorString(error);return 1;}
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


    const bool reactive=b->model.physics.material.enableSurfaceReactions||b->model.physics.material.enablePoreOutflow;
    const bool staged=reactive||b->gasBoundaryLayer.enabled;
    if(staged&&prepareResidentSurface(b,dt,true))return 1;
    if(!staged)b->stageWallPosition={};
    if(b->gasBoundaryLayer.enabled){
        if((reactive||!b->preparedWallReady||b->preparedWallTime!=stageTime||b->preparedWallDt!=dt)
            &&evaluateResidentSurfaceLayer(b,dt,stageTime,b->preparedWallReady
                &&b->preparedWallTime==stageTime&&b->preparedWallDt==dt))return 1;
        transport::refreshCoupledWallFluxKernel<<<(b->gasBoundaryLayer.count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(b->deviceState);
        const auto code=cudaGetLastError();
        if(code!=cudaSuccess){b->error=std::string("refresh current wall face flux: ")+cudaGetErrorString(code);return 1;}
    }
    b->stageTime=windowAbsolute(b->program.interval.begin,coupledStageTime(*b));b->stagePacketDt=dt;
    applyCoupledSurfaceKernel<<<(count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(b->deviceState,
        b->interfaceFaces.data(),count,b->interfaceInputs.data(),staged?b->interfaceStageWall.data():b->interfaceWallProgram.data(),
        staged?GasWallProgramPosition{}:b->stageWallPosition,b->interfacePhysics.data(),dt,b->gasStageIndex==0,b->record.microSequence,b->record.gasGeometry,b->interfaceRecords.data(),b->coupledInterval.data(),b->gasStageIndex,staged);
    if(!b->surfaceTransferFault.check(cudaGetLastError(),"evaluate resident coupled interface")){
        b->error=b->surfaceTransferFault.message;return 1;}
    return 0;
}
int accountCoupledFaces(Backend* b,Real ledgerDt){
    const int count=int(b->interfaceFaces.size());
    if(count){
        accountCoupledSurfaceKernel<<<(count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b->gasCaptureStream>>>(b->deviceState,
            b->interfaceFaces.data(),count,b->interfaceInputs.data(),b->interfaceWallProgram.data(),b->stageWallPosition,
            b->interfacePhysics.data(),b->stagePacketDt,ledgerDt,b->interfaceRecords.data(),b->coupledInterval.data(),b->gasStageIndex);
        if(!b->surfaceTransferFault.check(cudaGetLastError(),"accumulate resident coupled exchange")){
            b->error=b->surfaceTransferFault.message;return 1;}
    }
    ++b->gasStageIndex;return 0;
}


__global__ void rebaseAcceptedGasInventory(SharedGasDeviceView* state,const Real* oldVolume,const Real* newVolume){
    auto& s=*state;const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=s.nCells)return;
    using Field=typename std::remove_reference<decltype(s.rho[c])>::type;
    using Energy=typename std::remove_reference<decltype(s.rhoE[c])>::type;
    const Field oldV=Field(oldVolume[c]),newV=Field(newVolume[c]);
    s.rho[c]=(s.rho[c]*oldV)/newV;
    s.rhoUx[c]=(s.rhoUx[c]*oldV)/newV;s.rhoUy[c]=(s.rhoUy[c]*oldV)/newV;s.rhoUz[c]=(s.rhoUz[c]*oldV)/newV;
    s.rhoE[c]=(s.rhoE[c]*Energy(oldVolume[c]))/Energy(newVolume[c]);
    if(s.gasSpecies.mode!=ugkwp::GasMode::SingleLegacy)
        for(int j=0;j<Ns;++j){const int i=j*s.nCells+c;s.gasSpecies.rho[i]=(s.gasSpecies.rho[i]*oldV)/newV;}
    if(s.sstConfigured){s.rhoK[c]=(s.rhoK[c]*Energy(oldVolume[c]))/Energy(newVolume[c]);s.rhoOmega[c]=(s.rhoOmega[c]*oldV)/newV;}
}
bool loadGasWindowState(Backend& b,const HostState& snapshot,std::string& error){
    if(b.windowBatchPending){error="gas window still owns active device buffers";return false;}
    for(auto& graph:b.windowGraph)graph.storage=nullptr;
    b.windowTransport=nullptr;b.windowRadiationMaterial=nullptr;b.windowMaterial=nullptr;
    b.windowBatchPrepared=false;b.windowBatchRetired=false;b.windowMaterialHistoryReady=false;
    b.storage.beginOperation();
    b.wallDiagnosticsReady=false;b.historyFrameCount=0;b.historyPublishedFrames=0;b.historyArchive.reset();


    b.microGeometryPrepared=false;b.preparedWallReady=false;ugkwp::invalidateGasBoundaryLayerPreparation(b);
    const bool reuse=b.residentAcceptedEndpoint;
    b.residentAcceptedEndpoint=false;b.gasEndpointPublished=false;
    if(reuse){


        if((b.residentTrajectoryAdvanced&&!b.storage.uploadGeometry(snapshot.gasMesh,false))
            ||!b.storage.clearGeometry()||!b.storage.clearTrialStatus()){
            error=b.storage.error();return false;
        }
        if(b.residentTrajectoryAdvanced||b.acceptedGeometryChanged){
            bindSharedGasStorage(b);
            if(!b.particles.retainAcceptedGeometry(b.storage,b.gasCaptureStream,error))return false;
        }
        b.trial=snapshot;error.clear();return true;
    }
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
    if(options.courantUpdateInterval<1||!std::isfinite(options.maximumTimeStepGrowth)||options.maximumTimeStepGrowth<1){
        error="invalid Courant update interval or time-step growth factor";return nullptr;
    }
    if(mag(model.physics.gravity)!=0){error="coupled gas gravity source/work ledger is not enabled";return nullptr;}
    if(options.turbulenceModel!=(model.physics.enableSst?3:0))
        std::fprintf(stderr,"CHMT requested physics SST=%d execution turbulence=%d; effective SST=%d\n",
            int(model.physics.enableSst),options.turbulenceModel,int(options.turbulenceModel==3));
    ugkwp::GasCapabilityRequest request;request.mode=gas.mode;request.fluxScheme=options.fluxScheme;request.reconstruction=options.reconstruction;request.limiter=options.limiter;request.timeIntegrator=options.timeIntegrator;request.turbulenceModel=options.turbulenceModel;request.sstWallTreatment=int(model.physics.wallModel.family);request.boundaryLayerModel=int(model.physics.wallModel.model);request.movingGeometry=model.physics.meshMotion.policy!=MeshMotionPolicy::Static;request.particleCoupling=model.physics.enableParticles;
    request.pairedMixtureParticleCoupling=true;
    request.residentParticleGeometry=true;request.residentParticleWallCoupling=true;
    const auto supported=ugkwp::validateGasCapabilities(request);if(!supported){error=supported.message;return nullptr;}
    if(gas.mode!=ugkwp::GasMode::SingleLegacy
        && (!ugkwp::mixtureGasFluxImplemented(options.fluxScheme)||(options.reconstruction!=0&&options.reconstruction!=1))) {
        error="selected mixture gas operator is unavailable";return nullptr;
    }



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
    std::unique_ptr<Backend> b(new Backend);b->model=model;b->options=options;b->accepted=b->trial=state;b->accepted.ledger.clear();b->trial.ledger.clear();
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
        ||!b->interfaceWallDiagnostics.resize(b->storage.hostView().gasBoundaryLayer.enabled?state.surface.gasFace.size():0,b->surfaceTransferFault)
        ||!b->interfaceFailure.resize(1,b->surfaceTransferFault)
        ||!b->interfacePhysics.resize(1,b->surfaceTransferFault)
        ||!b->coupledInterval.resize(1,b->surfaceTransferFault)){
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
    if(model.physics.meshMotion.policy!=MeshMotionPolicy::Static){
        b->residentGeometry.reset(new ResidentCoupledGeometry);
        if(!b->residentGeometry->initialize(state,b->model.physics.tolerances,b->storage,error))return nullptr;
    }
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
    if(program.knots.size()<2){error="wall program sample outside interval";return false;}


    b.trial=b.accepted;b.program=program;
    b.movingWindow=false;
    if(b.model.physics.meshMotion.policy!=MeshMotionPolicy::Static){
        const auto changed=[](const std::vector<Vec3>& a,const std::vector<Vec3>& initial){
            if(a.size()!=initial.size())return true;
            for(std::size_t i=0;i<a.size();++i)
                if(a[i].x!=initial[i].x||a[i].y!=initial[i].y||a[i].z!=initial[i].z)return true;
            return false;
        };
        for(const auto& knot:program.knots)b.movingWindow=b.movingWindow
            ||changed(knot.gasPoints,b.accepted.gasMesh.points)
            ||changed(knot.solidPoints,b.accepted.solidMesh.points);
    }

    std::vector<WallFaceSample> faces;
    for(const auto& knot:program.knots)faces.insert(faces.end(),knot.faces.begin(),knot.faces.end());
    if(!b.interfaceWallProgram.resize(faces.size(),b.surfaceTransferFault)
        ||!b.storage.write(b.interfaceWallProgram.data(),faces)){
        error=b.surfaceTransferFault.message.empty()?b.storage.error():b.surfaceTransferFault.message;return false;
    }
    std::vector<ResidentWallKnotTime> times;times.reserve(program.knots.size());
    for(const auto& knot:program.knots)times.push_back({residentWindowTime(knotOffset(program.interval,knot)),knot.time});
    ResidentWallClock clock;clock.origin=program.interval.begin;
    if(!b.interfaceKnotTime.resize(times.size(),b.surfaceTransferFault)||!b.wallClock.resize(1,b.surfaceTransferFault)
        ||!b.storage.write(b.interfaceKnotTime.data(),times)
        ||!b.storage.write(b.wallClock.data(),std::vector<ResidentWallClock>{clock})){
        error=b.surfaceTransferFault.message.empty()?b.storage.error():b.surfaceTransferFault.message;return false;
    }
    b.interfaceGeometryReady=false;
    if(!b.history.begin(program.interval,error))return false;
    if(program.donorPlan.knots.size()==1){error="material donor interpolation needs two supplied knots";return false;}
    if(!loadGasWindowState(b,b.accepted,error))return false;
    if(b.movingWindow&&!b.residentGeometry->install(b.accepted,program,b.storage,b.interfaceInputs.data(),error))return false;
    b.elapsed=WindowTime{};b.nextGasSeed=b.options.adjustTimeStep
        ?(b.accepted.nextDt>0?b.accepted.nextDt:b.accepted.lastAcceptedDt)
        :Real(b.options.fixedDeltaT);return true;
}

bool beginGasWindow(Backend& b,const WallProgram& program,std::string& error){
    ugkp::diagnostics::beginOperation(b.hostDiagnostics,program.interval.identity.sequence,b.accepted.acceptedSteps,program.interval.begin);
    const bool success=beginGasWindowImpl(b,program,error);
    const bool healthy=dispatchHostDiagnostics(b);
    if(!healthy&&error.empty())error="fatal host gas-window diagnostic";
    return success&&healthy;
}
struct CoupledCourantMaximum {
    __host__ __device__ PhysicalReal operator()(PhysicalReal a,PhysicalReal b)const{return a!=a?a:b!=b?b:fmaxf(a,b);}
};
struct CoupledCourantValue {PhysicalReal value;bool unavailable;unsigned char initializedTail[7]{};};
struct CoupledCourantRead {
    const PhysicalReal* field;const int* status;const ugkp::diagnostics::DeviceContext* diagnostics;
    __host__ __device__ CoupledCourantValue operator()(int cell)const{


        if(ugkp::diagnostics::hasFatal(diagnostics))return {-std::numeric_limits<PhysicalReal>::infinity(),false};
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
        PhysicalReal* gasFluxPositivityScale;
    } view{b.nCells,b.nFaces,b.nInternalFaces,b.deviceState,v.gasFluxPositivityScale};
    const auto maximum=[&](const PhysicalReal* field,int count,cudaStream_t stream,double& value){
        CoupledCourantValue reduced{};
        try {
            const auto first=thrust::make_counting_iterator(0);
            reduced=thrust::transform_reduce(thrust::cuda::par(b.courantReductionStorage).on(stream),first,first+count,
                CoupledCourantRead{field,v.gasSpecies.cellStatus,b.diagnostics.device()},
                CoupledCourantValue{-std::numeric_limits<PhysicalReal>::infinity(),false},CoupledCourantReduce{});
        }
        catch(const std::exception& e){transport::setLastErrorText(e.what());return 1;}


        if(!b.storage.observeDiagnostics())return 1;
        if(reduced.unavailable){transport::setLastErrorText("current Co consumed input unavailable");return 1;}
        value=reduced.value;
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


    ugkp::diagnostics::beginOperation(b.hostDiagnostics,diagnosticSequence,b.trial.acceptedSteps,b.microTime);
    ugkp::diagnostics::beginOperationKernel<<<1,1,0,b.gasCaptureStream>>>(b.diagnostics.device(),diagnosticSequence,b.trial.acceptedSteps,b.microTime);
    const auto metadataLaunch=cudaGetLastError();
    if(metadataLaunch!=cudaSuccess){error=std::string("gas diagnostic metadata launch: ")+cudaGetErrorString(metadataLaunch);return false;}
    const Real remaining=boundedWindowDuration(nextGasWallKnotLocal(b.program,b.elapsed),b.elapsed);
    if(!finite(cap)||!finite(b.model.physics.maxDt)||!finite(remaining)||!finite(b.nextGasSeed)||b.nextGasSeed<0
        ||!finite(b.trial.lastAcceptedDt)||b.trial.lastAcceptedDt<0){error="invalid gas timestep input or seed";return false;}
    const Real proposedDt=requestedGasMicrostep(b.options.adjustTimeStep,Real(b.options.fixedDeltaT),
        b.nextGasSeed,cap,b.model.physics.maxDt);
    const Real referenceDt=minValue(proposedDt,remaining);
    if(!finite(referenceDt)||!(referenceDt>0)||b.elapsed+referenceDt==b.elapsed){error="gas reference timestep is nonfinite, nonpositive or nonadvancing";return false;}
    b.microDt=referenceDt;b.error.clear();transport::lastError.clear();
    if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}
    const bool courantRefreshed=b.options.adjustTimeStep
        &&(!b.courantReady||b.gasSteps%std::uint64_t(b.options.courantUpdateInterval)==0);
    if(courantRefreshed){
        transport::GasCourantMaxima co;
        if(!currentGasCourant(b,referenceDt,co)){
            error=b.error.empty()?transport::lastError:b.error;return false;
        }
        b.cachedCourantRate=co.maximum/referenceDt;b.courantReady=true;
    }
    const Real stable=b.cachedCourantRate>0?b.model.physics.cfl/b.cachedCourantRate:proposedDt;
    const auto schedule=selectGasMicrostep(b.options.adjustTimeStep,proposedDt,remaining,stable,
        Real(b.options.maximumTimeStepGrowth),minValue(cap,b.model.physics.maxDt),courantRefreshed);
    const Real selectedDt=schedule.accepted;


    if(!finite(selectedDt)||!(selectedDt>0)||b.elapsed+selectedDt==b.elapsed){error="gas CFL-selected timestep is nonfinite, nonpositive or nonadvancing";return false;}
    const Real dt=selectedDt;
    if(dt!=referenceDt){b.microDt=dt;if(prepareCoupledGeometry(&b)!=0){error=b.error;return false;}}
    b.microGeometryPrepared=true;
    if(transport::advanceGasTrial<transport::CoupledTrialPolicy>(&b,dt,b.microTime)!=0){
        error=b.error.empty()?transport::lastError:b.error;return false;
    }
    if(!b.history.appendAccepted(b.record,error))return false;
    b.elapsed=b.record.localEnd;
    ++b.gasSteps;b.nextGasSeed=schedule.next;b.trial.nextDt=b.nextGasSeed;b.trial.lastAcceptedDt=dt;
    report.acceptedDt=dt;report.nextGasDt=b.nextGasSeed;report.time=b.trial.time;report.localTime=b.elapsed;report.microSequence=b.record.microSequence;error.clear();return true;
}
bool advanceGasMicrostep(Backend& b,Real cap,GasMicroReport& report,std::string& error){
    const bool success=advanceGasMicrostepImpl(b,cap,report,error);
    const bool hostHealthy=dispatchHostDiagnostics(b);
    if(!hostHealthy && error.empty())error="fatal host gas diagnostic";
    return success&&hostHealthy;
}
const IntervalHistory& gasWindowHistory(const Backend& b){return b.history;}
std::shared_ptr<CoupledHistoryArchive> gasWindowHistoryArchive(const Backend& b){return b.historyArchive;}
ResidentMaterialDriveHistory residentGasWindowDriveHistory(const Backend& b){
    return b.historyArchive?b.historyArchive->residentView(b.gasCaptureStream):ResidentMaterialDriveHistory{};
}
IntervalHistory releaseGasWindowHistory(Backend& b){
    if(b.historyArchive){
        b.historyArchive->releaseProducer(b.gasCaptureStream);
        b.historyArchive.reset();
    }
    IntervalHistory published=std::move(b.history);
    b.history=IntervalHistory{};
    return published;
}
static bool publishWindowGeometry(Backend& b,std::string& error){
    if(!b.movingWindow||!b.residentGeometry->dirty())return true;
    if(!b.residentGeometry->queuePublication(b.storage,error))return false;
    if(!b.storage.observeDiagnostics()){error=b.storage.error();return false;}
    b.residentGeometry->publish(b.trial);return true;
}
bool gasWindowStageGeometry(Backend& b,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error){
    if(!b.movingWindow){
        const auto snapshot=[&](const HostMesh& mesh,std::array<HostStageGeometry,2>& stages){
            auto& stage=stages[0];stage.interval=b.microDt;stage.geometryVersion=mesh.geometryVersion;stage.topologyHash=mesh.topologyHash;
            stage.oldVolume=stage.newVolume=stage.evaluationVolume=mesh.volumes;
            stage.sweptVolume.assign(mesh.owner.size(),0);stage.areaVector=mesh.areaVectors;
            stage.cellCentre=mesh.cellCentres;stage.faceCentre=mesh.faceCentres;stage.oldPoints=stage.newPoints=mesh.points;
            stages[1]=stage;
        };
        snapshot(b.trial.gasMesh,gas);snapshot(b.trial.solidMesh,solid);
    }else{
        if(!publishWindowGeometry(b,error))return false;
        gas=b.trial.gasStages;solid=b.trial.solidStages;
    }
    error.clear();return true;
}
__global__ void packGasWindowCells(const SharedGasDeviceView* state,const int* cells,int count,int activeSpecies,HostGasPublication* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto& s=*state;const int c=cells ? cells[i] : i;HostGasPublication value{};auto& q=value.gas;
    q.mass=s.rho[c];q.momentum={s.rhoUx[c],s.rhoUy[c],s.rhoUz[c]};q.energy=s.rhoE[c];
    for(int j=0;j<Ns;++j)q.species[j]=s.gasSpecies.mode==ugkwp::GasMode::SingleLegacy?(j==activeSpecies?q.mass:GpuReal(0)):s.gasSpecies.rho[j*s.nCells+c];
    if(s.sstConfigured){value.sst.rhoK=s.rhoK[c];value.sst.rhoOmega=s.rhoOmega[c];}
    output[i]=value;
}
#include "gpu/AcceptedStateSnapshot.cuh"
static bool publishMaterialGasState(Backend& b,HostState& state,std::string& error){
    std::vector<int> cells;
    for(int face:state.surface.gasFace)if(face>=0)cells.push_back(state.gasMesh.owner[face]);
    if(b.model.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer){
        if(!b.hostWallGeometry)b.hostWallGeometry.reset(new WallClosureHost);
        auto config=b.model.physics.wallModel;config.enableSst=b.model.physics.enableSst;
        if(!b.hostWallGeometry->prepareGeometry(state.gasMesh,state.surface.gasFace,error,config,b.model.physics.diagnostics))return false;
        for(const auto& descriptor:b.hostWallGeometry->descriptors()){
            cells.push_back(descriptor.ownerCell);
            cells.insert(cells.end(),descriptor.matchingCells.begin(),descriptor.matchingCells.end());
        }
    }
    std::sort(cells.begin(),cells.end());cells.erase(std::unique(cells.begin(),cells.end()),cells.end());
    if(cells!=b.hostGasCellAddressing){
        if(!b.hostGasCells.resize(cells.size(),b.surfaceTransferFault)||!b.storage.write(b.hostGasCells.data(),cells)){
            error=b.surfaceTransferFault.message.empty()?b.storage.error():b.surfaceTransferFault.message;return false;
        }
        b.hostGasCellAddressing=std::move(cells);
    }
    const int count=int(b.hostGasCellAddressing.size());
    if(!b.hostGasValues.resize(count,b.surfaceTransferFault)){error=b.surfaceTransferFault.message;return false;}
    if(count)packGasWindowCells<<<(count+ugkp::gpu::workPlan().cellBlockThreads-1)/ugkp::gpu::workPlan().cellBlockThreads,ugkp::gpu::workPlan().cellBlockThreads,0,b.gasCaptureStream>>>(
        b.storage.deviceView(),b.hostGasCells.data(),count,b.model.physics.singleGasSpecies,b.hostGasValues.data());
    if(!b.surfaceTransferFault.check(cudaGetLastError(),"pack gas-window material cells")){error=b.surfaceTransferFault.message;return false;}
    const auto& view=b.storage.hostView();
    if(!b.storage.enqueueRead(b.hostGasValues.data(),count,b.hostGasStaging)
        ||(view.gasVolumeFraction&&!b.storage.enqueueRead(view.gasVolumeFraction,b.nCells,b.hostGasFraction))
        ||!b.storage.observeDiagnostics()){error=b.storage.error();return false;}
    if(view.gasVolumeFraction)state.gasVoidFraction=b.hostGasFraction;
    if(view.sstConfigured)state.sst.resize(state.gas.size());
    for(int i=0;i<count;++i){
        const int c=b.hostGasCellAddressing[i];const GpuReal volume=GpuReal(gasPhaseVolume(state,c));const auto& value=b.hostGasStaging[i];
        state.gas[c]=value.gas*volume;
        if(view.sstConfigured)state.sst[c]={value.sst.rhoK*volume,value.sst.rhoOmega*volume};
    }
    return true;
}
static bool publishGasWindowImpl(Backend& b,HostState& out,std::string& error,bool full){
    const bool publishHistory=b.historyPublishedFrames!=b.historyFrameCount;
    const auto faces=b.interfaceRecords.size();
    if(publishHistory&&b.historyArchive)b.historyArchive->seal(b.gasCaptureStream);
    if(b.wallDiagnosticsReady){
        const int count=int(b.interfaceWallDiagnostics.size());
        if(count){
            packCoupledWallDiagnosticsKernel<<<(count+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,b.gasCaptureStream>>>(
                b.interfaceRecords.data(),count,b.interfaceWallDiagnostics.data());
            if(!b.surfaceTransferFault.check(cudaGetLastError(),"pack gas-window wall diagnostics")){
                error=b.surfaceTransferFault.message;return false;
            }
        }


        if(!b.storage.enqueueRead(b.interfaceWallDiagnostics.data(),count,b.wallDiagnosticStaging)){
            error=b.storage.error();return false;
        }
    }
    if(b.movingWindow&&!b.residentGeometry->queuePublication(b.storage,error))return false;
    if(full){
        if(!b.storage.downloadState(b.trial,error,[&]{if(b.movingWindow)b.residentGeometry->publish(b.trial);})
            ||!b.particles.download(b.trial,error))return false;
    }else{
        if(b.movingWindow){
            if(!b.storage.observeDiagnostics()){error=b.storage.error();return false;}
            b.residentGeometry->publish(b.trial);
        }
        if(!publishMaterialGasState(b,b.trial,error))return false;
    }
    b.hostEndpointCurrent=full;
    if(publishHistory){
        const auto archive=b.historyArchive;const bool particles=b.particles.active(),matching=b.gasBoundaryLayer.enabled;
        b.history.setPayloadReader([archive,faces,particles,matching](std::size_t frame,GasIntervalRecord& record){
            const auto chunk=archive->chunk(frame);const auto offset=(frame-chunk->first)*faces;
            populateCoupledHostRecord(particles,matching,faces?chunk->surface.data()+offset:nullptr,
                faces&&matching?chunk->wall.data()+offset:nullptr,faces,record);
        },[archive,particles](const AcceptedLedger::ChunkVisitor& visit){
            std::vector<ExchangePacket> packets;
            archive->forEachSurfaceFrame([&](const CoupledSurfaceDownload* surface,std::size_t count){
                packets.clear();
                for(std::size_t i=0;i<count;++i)appendCoupledHostPackets(packets,surface[i],particles);
                for(auto& packet:packets)packet.consumerMask=requiredConsumers(packet.kind);
                if(!packets.empty())visit(packets.data(),packets.size());
            });
        });
        b.historyPublishedFrames=b.historyFrameCount;
    }
    if(b.wallDiagnosticsReady){
        b.trial.gasWallDiagnostics.resize(b.wallDiagnosticStaging.size());
        for(std::size_t i=0;i<b.wallDiagnosticStaging.size();++i)
            static_cast<ugkwp::gaswall::WallExchangeOutput<EnergyReal,Ns>&>(b.trial.gasWallDiagnostics[i])=b.wallDiagnosticStaging[i];
        b.trial.gasWallDiagnosticTime=b.wallDiagnosticTime;
    }
    out=b.trial;
    const bool published=gasWindowStageGeometry(b,out.gasStages,out.solidStages,error);
    b.gasEndpointPublished=published;return published;
}
#include "gpu/MaterialWindowExchange.inl"
bool downloadGasWindow(Backend& b,HostState& out,std::string& error){return publishGasWindowImpl(b,out,error,true);}
bool publishGasWindow(Backend& b,HostState& out,std::string& error){return publishGasWindowImpl(b,out,error,false);}
static bool commitGasWindowImpl(Backend& b,HostState& candidate,std::string& error,bool preservePublishedEndpoint,
    const MaterialWindowCommit* material=nullptr,Real externalHeat=0,Real nextDt=0){
    if(b.poisoned){error="gas backend is unusable after a failed device operation";return false;}



    if(!publishWindowGeometry(b,error))return false;







    bool corrected=b.trial.gasMesh.points.size()!=candidate.gasMesh.points.size();
    for(std::size_t i=0;!corrected&&i<candidate.gasMesh.points.size();++i){
        const auto a=b.trial.gasMesh.points[i],c=candidate.gasMesh.points[i];
        corrected=a.x!=c.x||a.y!=c.y||a.z!=c.z;
    }
    if(corrected){
        if(b.particles.active()&&!b.particles.mapGeometry(b.trial.gasMesh,candidate.gasMesh,b.storage,b.gasCaptureStream,error))return false;
        if(!b.storage.uploadGeometry(candidate.gasMesh,false)||!b.storage.refreshView()){
            error=b.storage.error();return false;
        }
        bindSharedGasStorage(b);
        if(b.particles.active()&&(!b.particles.refreshTrackingGeometry(b.storage,b.gasCaptureStream,error)
            ||!b.particles.result(ugkp::thermal::relocateMobileParticles(b.particles.owner(),b.gasCaptureStream),error)))return false;
        if(!b.storage.observeDiagnostics()){error=b.storage.error();return false;}
        if(b.particles.active()&&b.hostEndpointCurrent&&!b.particles.download(candidate,error))return false;
    }

    b.residentAcceptedEndpoint=preservePublishedEndpoint&&b.gasEndpointPublished
        &&candidate.gasMesh.topologyHash==b.trial.gasMesh.topologyHash
        &&candidate.gas.size()==b.trial.gas.size()
        &&candidate.gasMesh.owner.size()==b.trial.gasMesh.owner.size();
    if(!b.residentAcceptedEndpoint&&!b.hostEndpointCurrent){
        HostState endpoint=b.trial;
        if(!b.storage.downloadState(endpoint,error)||!b.particles.download(endpoint,error))return false;
        candidate.gas=std::move(endpoint.gas);candidate.sst=std::move(endpoint.sst);
        candidate.gasTemperature=std::move(endpoint.gasTemperature);candidate.gasVoidFraction=std::move(endpoint.gasVoidFraction);
        candidate.particles=std::move(endpoint.particles);candidate.particleSourceFaces=std::move(endpoint.particleSourceFaces);
        candidate.particleSourceResidualMass=std::move(endpoint.particleSourceResidualMass);
        candidate.gasMesh.boundaryPrimitive=std::move(endpoint.gasMesh.boundaryPrimitive);b.hostEndpointCurrent=true;
    }
    if(b.residentAcceptedEndpoint&&corrected){
        std::vector<Real> oldVolume(candidate.gas.size()),newVolume(candidate.gas.size());
        for(std::size_t c=0;c<oldVolume.size();++c){
            oldVolume[c]=gasPhaseVolume(b.trial,c);newVolume[c]=gasPhaseVolume(candidate,c);
        }
        if(!b.acceptedOldPhaseVolume.resize(oldVolume.size(),b.surfaceTransferFault)
            ||!b.acceptedNewPhaseVolume.resize(newVolume.size(),b.surfaceTransferFault)
            ||!b.storage.write(b.acceptedOldPhaseVolume.data(),oldVolume)
            ||!b.storage.write(b.acceptedNewPhaseVolume.data(),newVolume)){
            error=b.surfaceTransferFault.message.empty()?b.storage.error():b.surfaceTransferFault.message;return false;
        }
        if(!oldVolume.empty())rebaseAcceptedGasInventory<<<(oldVolume.size()+ugkp::gpu::workPlan().cellBlockThreads-1)/ugkp::gpu::workPlan().cellBlockThreads,ugkp::gpu::workPlan().cellBlockThreads,0,b.gasCaptureStream>>>(
            b.storage.deviceView(),b.acceptedOldPhaseVolume.data(),b.acceptedNewPhaseVolume.data());
        if(!b.surfaceTransferFault.check(cudaGetLastError(),"rebase accepted gas inventories")){
            error=b.surfaceTransferFault.message;return false;
        }
    }
    if(!b.hostEndpointCurrent&&b.residentAcceptedEndpoint&&corrected){
        if(!publishMaterialGasState(b,candidate,error))return false;
    }
    if(material){
        if(!b.coupledSlot.resize(1,b.surfaceTransferFault)||!b.coupledReceipt.resize(1,b.surfaceTransferFault)){
            error=b.surfaceTransferFault.message;return false;
        }
        WindowReceipt base{},endpoint{},completed{};
        base.window={b.accepted.time,b.accepted.commitSequence,b.accepted.acceptedSteps,true};
        endpoint.window=*material;
        if(!material->committed){endpoint.window.time=candidate.time;endpoint.window.sequence=b.program.interval.identity.sequence;}
        if(endpoint.window.time!=b.program.interval.end||b.elapsed!=intervalDuration(b.program.interval)){
            error="material and gas did not reach their common window endpoint";return false;
        }
        endpoint.gasGeometry=candidate.gasMesh.geometryVersion;endpoint.solidGeometry=candidate.solidMesh.geometryVersion;
        endpoint.nextDt=nextDt;endpoint.radiationTime=candidate.radiationTime;
        closeCoupledWindow<<<1,1,0,b.gasCaptureStream>>>(b.coupledSlot.data(),b.coupledReceipt.data(),
            base,endpoint,candidate.budget,externalHeat,b.hasCoupledReceipt,b.diagnostics.device());
        if(!b.surfaceTransferFault.check(cudaGetLastError(),"commit coupled geometry and heat ledger")||
           !b.surfaceTransferFault.check(cudaMemcpyAsync(&completed,b.coupledReceipt.data(),sizeof(completed),
                cudaMemcpyDeviceToHost,b.gasCaptureStream),"read coupled endpoint receipt")||
           !b.surfaceTransferFault.check(cudaStreamSynchronize(b.gasCaptureStream),"collect coupled endpoint receipt")){
            error=b.surfaceTransferFault.message;return false;
        }
        if(!completed.window.committed){b.storage.observeDiagnostics();error="coupled endpoint could not be committed";return false;}
        b.hasCoupledReceipt=true;
        candidate.time=completed.window.time;candidate.commitSequence=completed.window.sequence;
        candidate.acceptedSteps=completed.window.acceptedStep;candidate.budget.radiation=completed.radiation;
        candidate.lastAcceptedDt=completed.dt;candidate.nextDt=completed.nextDt;
    }else b.hasCoupledReceipt=false;
    HostState newAccepted=candidate,newTrial=candidate;
    newAccepted.ledger.clear();newTrial.ledger.clear();
    b.residentTrajectoryAdvanced=b.movingWindow&&!corrected;
    b.acceptedGeometryChanged=b.movingWindow||corrected;
    b.accepted=std::move(newAccepted);b.trial=std::move(newTrial);error.clear();return true;
}
bool commitGasWindow(Backend& b,HostState& candidate,std::string& error,bool preservePublishedEndpoint){
    const bool success=commitGasWindowImpl(b,candidate,error,preservePublishedEndpoint);
    const bool healthy=dispatchHostDiagnostics(b);
    if(!healthy&&error.empty())error="fatal host gas-window commit diagnostic";
    return success&&healthy;
}
bool acceptCoupledWindow(Backend& b,HostState& candidate,const MaterialWindowCommit& material,
    Real externalHeat,Real nextDt,std::string& error){
    const bool success=commitGasWindowImpl(b,candidate,error,true,&material,externalHeat,nextDt);
    const bool healthy=dispatchHostDiagnostics(b);
    if(!success||!healthy)b.poisoned=true;
    return success&&healthy;
}
bool publishAcceptedState(Backend& b,HostState& state,std::string& error){
    if(!b.hostEndpointCurrent){
        if(!b.storage.downloadState(state,error)||!b.particles.download(state,error))return false;
        b.accepted=state;b.accepted.ledger.clear();b.trial=b.accepted;b.hostEndpointCurrent=true;
    }
    error.clear();return true;
}
bool downloadState(const Backend& b,HostState& out,std::string& error){out=b.accepted;return publishAcceptedState(const_cast<Backend&>(b),out,error);}
}

namespace chmt {
bool borrowRadiationState(Backend& b,ParticleRadiationState& out,std::string& error){
    if(!b.particles.active())return downloadRadiationState(b,out,error);
    if(!b.particles.result(ugkp::thermal::orderParticlePhaseAfterGas(b.particles.owner(),b.gasCaptureStream),error))return false;
    if(!b.particles.result(ugkwpGpuResidentStrictBorrowRadiationParticleState(b.particles.owner(),&out.resident),error)
        ||!b.particles.result(ugkwpGpuResidentStrictParticleRadiationResponse(b.particles.owner(),&out.response),error))return false;


    out.resident.boundaryTemperature=nullptr;
    return true;
}
bool applyResidentRadiationEnergy(Backend& b,const ugkp::gpu::RadiationMaterialDeviceView& heat,ugkp::gpu::RadiationAcceptedEnergyDeviceView& actual,HostState& endpoint,std::string& error){
    actual={b.nCells,nullptr,nullptr};if(!b.particles.active())return true;
    if(!b.particles.result(ugkwpGpuResidentStrictApplyParticleRadiationDeviceEnergy(b.particles.owner(),&heat,nullptr),error)
        ||!b.particles.result(ugkwpGpuResidentStrictBorrowAcceptedRadiationEnergy(b.particles.owner(),&actual),error)
        ||!b.particles.result(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(b.particles.owner()),error)
        ||!b.particles.result(ugkp::thermal::orderGasAfterParticlePhase(b.particles.owner(),b.gasCaptureStream),error))return false;
    return !b.hostEndpointCurrent||b.particles.download(endpoint,error);
}
bool downloadRadiationState(Backend& b,ParticleRadiationState& out,std::string& error){
    out.mass.assign(b.nCells,0);out.temperature.assign(b.nCells,b.model.particlePhase.minimumTemperature);
    out.diameter.assign(b.nCells,0);out.occupiedArea.assign(b.nFaces,0);
    if(!b.particles.active())return true;
    if(!b.particles.result(ugkp::thermal::orderParticlePhaseAfterGas(b.particles.owner(),b.gasCaptureStream),error))return false;
    std::vector<double> wireMass(b.nCells),tm(b.nCells),dm(b.nCells),wireArea(b.nFaces);
    if(!b.particles.result(ugkwpGpuResidentStrictDownloadMobileParticleRadiationSums(b.particles.owner(),b.nCells,wireMass.data(),tm.data(),dm.data()),error)
        ||!b.particles.result(ugkwpGpuResidentStrictDownloadParticleWallOccupiedArea(b.particles.owner(),b.nFaces,wireArea.data()),error)
        ||!b.particles.result(ugkwpGpuResidentStrictParticleRadiationResponse(b.particles.owner(),&out.response),error))return false;
    for(int f=0;f<b.nFaces;++f)out.occupiedArea[f]=EnergyReal(wireArea[f]);
    for(int c=0;c<b.nCells;++c){out.mass[c]=EnergyReal(wireMass[c]);if(out.mass[c]>0){
        out.temperature[c]=EnergyReal(tm[c])/out.mass[c];out.diameter[c]=EnergyReal(dm[c])/out.mass[c];}}
    return true;
}
bool applyRadiationEnergy(Backend& b,const std::vector<EnergyReal>& heat,std::vector<EnergyReal>& actual,HostState& endpoint,std::string& error){
    actual.assign(b.nCells,0);if(!b.particles.active())return true;
    if(heat.size()!=std::size_t(b.nCells)){error="radiation cell layout mismatch";return false;}
    std::vector<double> wireHeat(heat.begin(),heat.end()),wireActual(b.nCells);
    if(!b.particles.result(ugkwpGpuResidentStrictApplyParticleRadiationEnergy(b.particles.owner(),b.nCells,wireHeat.data(),wireActual.data()),error)
        ||!b.particles.result(ugkwpGpuResidentStrictRefreshParticleEnthalpyPacked(b.particles.owner()),error))return false;
    actual.assign(wireActual.begin(),wireActual.end());
    return !b.hostEndpointCurrent||b.particles.download(endpoint,error);
}
void setWindowRadiationEnergy(Backend& b,const std::vector<EnergyReal>& energy){b.history.setExternalRadiationEnergy(energy);}
__global__ void mapWindowRadiation(int count,const int* faces,int boundaryOffset,
    const ugkp::gpu::RadiationReal* boundary,EnergyReal* mapped){
    for(int f=blockIdx.x*blockDim.x+threadIdx.x;f<count;f+=blockDim.x*gridDim.x)
        mapped[f]=faces[f]<0?EnergyReal(0):EnergyReal(boundary[faces[f]-boundaryOffset]);
}
bool setWindowRadiationEnergy(Backend& b,const ugkp::gpu::RadiationMaterialDeviceView& radiation,std::string& error){
    const auto count=b.interfaceFaces.size();const int boundary=b.nFaces-b.nInternalFaces;
    if(!radiation.enabled||radiation.boundaryFaces!=boundary||(boundary&&!radiation.boundaryEnergy)){
        error="DOM boundary energy data or face layout missing";return false;
    }
    for(int face:b.accepted.surface.gasFace)if(face>=0&&(face<b.nInternalFaces||face>=b.nFaces)){
        error="material radiation face is outside the gas boundary";return false;
    }
    if(!b.windowRadiation.resize(count,b.executionFault)){error=b.executionFault.message;return false;}
    cudaEvent_t ready=nullptr;
    if(!b.executionFault.check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"create DOM material dependency")){
        error=b.executionFault.message;return false;
    }
    const bool joined=b.executionFault.check(cudaEventRecord(ready,static_cast<cudaStream_t>(radiation.producerStream)),"record DOM energy dependency")
        &&b.executionFault.check(cudaStreamWaitEvent(b.executionStream.get(),ready,0),"join DOM energy dependency");
    const bool retired=b.executionFault.check(cudaEventDestroy(ready),"retire DOM energy dependency");
    if(!joined||!retired){error=b.executionFault.message;return false;}
    if(count)mapWindowRadiation<<<(count+255)/256,256,0,b.executionStream.get()>>>(int(count),b.interfaceFaces.data(),
        b.nInternalFaces,radiation.boundaryEnergy,b.windowRadiation.data());
    if(!b.executionFault.check(cudaGetLastError(),"map DOM material energy")){error=b.executionFault.message;return false;}
    b.history.setExternalRadiationDevice({b.windowRadiation.data(),count,b.executionStream.get()});
    error.clear();return true;
}

}

#include "gpu/GasWindowGraph.cuh"
