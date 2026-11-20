#include "mesh/MeshGeometryCache.H"
#include "gpu/ResidentMaterialLocalLoop.cuh"
#include "gpu/StreamedMaterialLoop.cuh"
#include "gpu/CoupledHistoryArchive.cuh"
#include "../../../common/gpu/PinnedTransferQueue.cuh"
#include "../../../common/gpu/WorkPlan.H"
#include "gpu/ResidentMaterialSlice.cuh"
#include "gpu/ResidentMaterialSource.cuh"
#include "../../../common/gpu/StableRowIndex.cuh"
#include "ablation/SurfaceDonorAllocation.H"
#include "gpu/MaterialBackend.H"
#include "gpu/HostTransferQueue.H"
#include "gpu/MaterialTransportKernels.cuh"
#include "gpu/MaterialSourceKernels.cuh"
#include "gpu/MaterialSurfaceKernels.cuh"
#include "ablation/MaterialSurfaceInput.H"
#include "gpu/ResidentMaterialEndpoint.cuh"
#include "../../../common/gpu/ScalarWorkLaunch.cuh"
#include "materials/MaterialInventoryMath.H"
#include "materials/CpuMaterialDriver.H"
#include "../../../common/GpuDiagnosticsOwner.cuh"
#include "../../../common/gasTransport/GasBoundaryLayerWorkspace.H"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <limits>
#include "../../../common/gpu/ImplicitDiffusion.H"
#include "../../../common/gpu/ExecutionGraph.cuh"
namespace chmt {
namespace {
using namespace materialSourceDetail;


template<class Current,class Volumes>
__global__ void prepareConductionKernel(int count,const SolidQ* initial,const Real* initialVolumes,
    Current currentValue,Volumes volumesValue,const Real* nativeVolumes,const PhysicsConfig* physics,
    MaterialThermalInput* input,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    const SolidQ* current=currentValue;const Real* volumes=volumesValue;
    status[c]={};MaterialThermalInput value;bool numerical=true;
    const auto code=makeMaterialConductionCaloric(current[c],*physics,value.caloric,numerical);
    if(code!=MaterialCaloricCode::Success){status[c]={1,int(code),c,numerical};return;}
    MaterialPrimitive primitive;
    if(!recoverMaterial<false>(initial[c],initialVolumes[c],*physics,primitive,nullptr,-1,&numerical)){
        status[c]={2,0,c,numerical};return;
    }
    value.inventory=current[c];value.anchor=primitive.temperature;value.target=current[c].energy;
    value.nativeVolume=nativeVolumes[c];value.materialVolume=volumes[c];input[c]=value;
}
__global__ void conductionAnchorTemperature(int n,const MaterialThermalInput* input,ConductionReal* temperature){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<n)temperature[c]=input[c].anchor;
}
__global__ void conductionCoefficientsKernel(int count,const MaterialThermalInput* input,
    const ConductionReal* temperature,ConductionReal dt,const PhysicsConfig* physics,MaterialThermalCoefficients* output,
    ConductionReal* capacity,ConductionReal* source,ConductionReal* conductivity){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<count){const auto value=materialThermalCoefficients(input[c],temperature[c],dt,*physics);
        output[c]=value;capacity[c]=value.capacity;source[c]=value.source;conductivity[c]=value.conductivity;}
}


__global__ void inspectConductionStartupKernel(int count,const MaterialThermalCoefficients* coefficients,
    const PhysicsConfig* physics,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    const auto value=coefficients[c];
    status[c]=value.error?MaterialStepResult{1,value.error,c,true}:MaterialStepResult{};
    if(!value.error&&value.rawPorosity!=value.porosity){auto phi=value.rawPorosity;clipPhysicsMassFraction(*physics,phi,3824,c);}
}
template<class Current,class Volumes,class Next>
__global__ void finishConductionKernel(int count,Current currentValue,Volumes volumesValue,
    const MaterialThermalInput* input,const ConductionReal* temperature,const PhysicsConfig* physics,
    Next nextValue,MaterialStepResult* status){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count)return;
    const SolidQ* current=currentValue;const Real* volumes=volumesValue;
    SolidQ* next=nextValue;
    status[c]={};SolidQ value=current[c];value.energy=input[c].caloric.energy<false>(temperature[c]);
    MaterialPrimitive primitive;bool numerical=false;
    if(!recoverMaterial<false>(value,volumes[c],*physics,primitive,nullptr,-1,&numerical)){
        status[c]={3,0,c,numerical};return;
    }
    value.porosity=primitive.porosity;next[c]=value;
}
__global__ void selectConductionFailure(const MaterialStepResult* status,int count,MaterialStepResult* result){
    MaterialStepResult value{};
    for(int c=threadIdx.x;c<count;c+=blockDim.x)if(status[c].phase){value=status[c];break;}
    reduceMaterialFailure(value,result,false);
}

__device__ void mergeConductionAudit(MaterialConductionAudit& a,const MaterialConductionAudit& b){
    a.residualEnergy.merge(b.residualEnergy);a.inventory.merge(b.inventory);a.stableChange.merge(b.stableChange);a.boundaryEnergy.merge(b.boundaryEnergy);
    a.change=fmaxf(a.change,b.change);if(!a.coefficientError)a.coefficientError=b.coefficientError;
    if(b.invalidCell>=0&&(a.invalidCell<0||b.invalidCell<a.invalidCell)){a.invalidCell=b.invalidCell;a.invalidTemperature=b.invalidTemperature;}
}
__global__ void conductionAuditParts(int cells,int boundary,const MaterialThermalInput* input,const MaterialThermalCoefficients* coefficients,
    const ConductionReal* temperature,const ConductionReal* previous,const ConductionReal* residual,const ConductionReal* flux,const int* external,ConductionReal dt,MaterialConductionAudit* parts){
    extern __shared__ __align__(16) unsigned char storage[];
    auto* values=reinterpret_cast<MaterialConductionAudit*>(storage);MaterialConductionAudit a;
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c<cells){const auto value=input[c];const ConductionReal T=temperature[c];a.coefficientError=coefficients[c].error;
        a.inventory.add(fabsf(value.target));a.inventory.add(fabsf(value.caloric.energy<false>(T)));
        if(residual)a.residualEnergy.product(dt,fabsf(residual[c]));
        if(previous){if(!finite(T)||T<=0){a.invalidCell=c;a.invalidTemperature=T;}
            a.stableChange.product(value.caloric.secant<false>(T,value.anchor),T-value.anchor);
            a.stableChange.add(-value.target);a.stableChange.add(value.caloric.energy<false>(value.anchor));
            a.change=fabsf(T-previous[c])/fmaxf(1.0f,fmaxf(fabsf(T),fabsf(previous[c])));}}
    if(c<boundary&&flux&&external[c])a.boundaryEnergy.product(dt,flux[c]);
    values[threadIdx.x]=a;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){const int step=(active+1)/2;if(threadIdx.x<active-step)mergeConductionAudit(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x)parts[blockIdx.x]=values[0];
}
__global__ void conductionAuditFinish(int count,const MaterialConductionAudit* parts,MaterialConductionAudit* output){
    extern __shared__ __align__(16) unsigned char storage[];
    auto* values=reinterpret_cast<MaterialConductionAudit*>(storage);MaterialConductionAudit a;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)mergeConductionAudit(a,parts[i]);values[threadIdx.x]=a;__syncthreads();
    for(int active=blockDim.x;active>1;active=(active+1)/2){const int step=(active+1)/2;if(threadIdx.x<active-step)mergeConductionAudit(values[threadIdx.x],values[threadIdx.x+step]);__syncthreads();}
    if(!threadIdx.x)*output=values[0];
}

struct MaterialConductionIteration {
    int continuationStage=0;
    int preconditionerFallbackBlocks=0;
    int iteration=0,failure=0;bool valid=true,converged=false,stalled=false;
    ConductionReal residual=0,energyTolerance=0;
    MaterialConductionAudit audit;
    ugkp::gpu::ConjugateGradientControls linear;
};
__global__ void beginConductionLoop(MaterialConductionIteration* report,
    ugkp::gpu::ConjugateGradientControls linear,ugkp::diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;if(report->continuationStage==0){*report={};report->linear=linear;report->valid=!ugkp::diagnostics::hasFatal(context);report->continuationStage=1;}
    cudaGraphSetConditional(handle,report->valid);
}
__global__ void conductionSelect(MaterialConductionIteration* report,int stage,const int* pending,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,report->valid&&report->continuationStage==stage&&(!pending||!*pending));
}
__global__ void conductionPrepared(MaterialConductionIteration* report){if(!threadIdx.x)report->continuationStage=2;}
__global__ void conductionContinue(MaterialConductionIteration* report,const int* pending,CpuMaterialControls controls,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;
    if(pending&&*pending){cudaGraphSetConditional(handle,0);return;}
    report->continuationStage=1;
    cudaGraphSetConditional(handle,report->valid&&!report->converged&&!report->stalled&&report->iteration<controls.nonlinearMaxIterations);
}
__global__ void prepareConductionLinearControls(MaterialConductionIteration* report,const MaterialConductionAudit* audit,
    CpuMaterialControls controls,cudaGraphConditionalHandle permitted){
    if(threadIdx.x)return;++report->iteration;report->audit=*audit;
    if(audit->coefficientError){report->valid=false;report->failure=1;cudaGraphSetConditional(permitted,0);return;}
    using ugkp::gpu::ScaledThermalSum;const auto initial=audit->residualEnergy;
    ScaledThermalSum scale;scale.add(1);if(scale.lessEqual(initial))scale=initial;
    auto tolerance=scale.multiplied(ConductionReal(controls.energyRelativeTolerance));tolerance.add(ConductionReal(controls.energyAbsoluteTolerance));
    tolerance.merge(audit->inventory.multiplied(ConductionReal(64)*std::numeric_limits<ConductionReal>::epsilon()));
    const ConductionReal relative=initial.populated?fminf(1.0f,tolerance.ratioTo(initial)*ConductionReal(controls.linearEnergyBudgetFraction)):1.0f;
    report->linear.tolerance=std::numeric_limits<ConductionReal>::min();report->linear.relativeTolerance=relative;
    cudaGraphSetConditional(permitted,1);
}
__global__ void finishConductionIteration(MaterialConductionIteration* report,const MaterialConductionAudit* audit,
    const ugkp::gpu::ConjugateGradientReport* linear,CpuMaterialControls controls,
    ugkp::diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;using namespace ugkp::diagnostics;
    if(!report->valid){cudaGraphSetConditional(handle,0);return;}
    report->audit=*audit;
    if(linear->preconditionerFallbackBlocks>0)
        recordWarning(context,makeRecord(Code::MethodFallback,36010,4,linear->iterations,
            EntityKind::None,-1,-1,linear->preconditionerFallbackBlocks,0,0,Quantity::LinearSystem));
    if(!std::isfinite(linear->finalResidual)){report->valid=false;report->failure=2;}
    else if(audit->invalidCell>=0){
        report->valid=false;report->failure=3;
        const Real temperature=audit->invalidTemperature;
        recordFatal(context,makeRecord(!finite(temperature)?Code::NonFinite:Code::NegativeTemperature,
            4103,4,0,EntityKind::Cell,audit->invalidCell,-1,temperature,0,0,Quantity::Temperature));
    }
    else if(!std::isfinite(audit->change)){
        report->valid=false;report->failure=4;
        recordFatal(context,makeRecord(Code::NonFinite,4104,4,0,EntityKind::None,-1,-1,audit->change));
    }
    if(!report->valid){cudaGraphSetConditional(handle,0);return;}
    if(!linear->converged||linear->singular){
        const int site=linear->budgetExhausted?34025:(linear->stagnated?34026:(linear->singular?34027:34020));
        recordWarning(context,makeRecord(Code::AccuracyResidual,site,4,linear->iterations,
            EntityKind::None,-1,-1,linear->finalResidual,linear->targetResidual,0,Quantity::LinearSystem));
    }
    auto energy=audit->stableChange;energy.merge(audit->boundaryEnergy);report->residual=energy.value;
    auto scale=audit->stableChange.magnitude();const auto boundary=audit->boundaryEnergy.magnitude();if(scale.lessEqual(boundary))scale=boundary;
    ugkp::gpu::ScaledThermalSum unity;unity.add(1);if(scale.lessEqual(unity))scale=unity;
    auto tolerance=scale.multiplied(ConductionReal(controls.energyRelativeTolerance));tolerance.add(ConductionReal(controls.energyAbsoluteTolerance));
    tolerance.merge(audit->inventory.multiplied(ConductionReal(64)*std::numeric_limits<ConductionReal>::epsilon()));report->energyTolerance=tolerance.value;
    const bool energyConverged=finite(report->residual)&&fabsf(report->residual)<=report->energyTolerance;
    report->stalled=audit->change==0&&linear->iterations==0;
    report->converged=audit->change<=controls.nonlinearRelativeTolerance&&energyConverged;
    const bool again=!report->converged&&!report->stalled&&report->iteration<controls.nonlinearMaxIterations;
    cudaGraphSetConditional(handle,again);
    if(!again){
        if(report->stalled&&!energyConverged)recordWarning(context,makeRecord(Code::AccuracyResidual,34021,4,report->iteration,
            EntityKind::None,-1,-1,report->residual,report->energyTolerance,0,Quantity::Accuracy));
        if(audit->change>controls.nonlinearRelativeTolerance)recordWarning(context,makeRecord(Code::AccuracyResidual,34022,4,report->iteration,
            EntityKind::None,-1,-1,audit->change,controls.nonlinearRelativeTolerance,0,Quantity::Accuracy));
        if(!energyConverged)recordWarning(context,makeRecord(Code::ConservationResidual,34023,4,report->iteration,
            EntityKind::None,-1,-1,report->residual,report->energyTolerance,0,Quantity::Conservation));
    }
}
}
namespace {
__global__ void gatherMaterialObservation(int count,const int* cells,const SolidQ* state,SolidQ* observed){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<count)observed[row]=state[cells[row]];
}
}
struct MaterialRowStorage {
    Buffer<int> keys,sortedKeys,inputIndices,counts;
};
struct PacketSolidRow {
    const ExchangePacket* packets;
    __device__ int operator()(int i)const{
        return (requiredConsumers(packets[i].kind)&ConsumeSolid)?packets[i].solidCell:-1;
    }
};
struct RadiationSolidRow {
    const unsigned* receivers;const int* cells;
    __device__ int operator()(int i)const{return receivers[i]==ConsumeSolid?cells[i]:-1;}
};
__global__ void gatherMaterialSliceReceivers(ResidentMaterialDriveHistory history,std::size_t frame,unsigned* receivers){
    const auto f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(f<history.faces)receivers[f]=history.surface[frame*history.faces+f].radiationReceiver;
}
struct MaterialBackend {
    CudaFault fault;
    DeviceStream stream;
    HostTransferQueue transfers{fault};
    ugkp::diagnostics::GpuOwner diagnostics;
    PhysicsConfig physics{};
    std::uint64_t sequence=0,step=0;Real time=0;
    Buffer<PhysicsConfig> devicePhysics,quietSurfacePhysics;
    Buffer<SolidQ> current,next,initial,gasChange,candidateGasChange,donor;



    Buffer<int> observationCells;Buffer<SolidQ> observation;
    std::vector<int> cachedObservationCells;int observedInventorySize=-1;
    std::vector<SolidQ> publishedCurrent;
    bool publishedCurrentValid=false;
    bool retains(const std::vector<SolidQ>& input)const{
        if(!publishedCurrentValid||input.size()!=publishedCurrent.size())return false;
        for(std::size_t c=0;c<input.size();++c){
            const auto& a=input[c];const auto& v=publishedCurrent[c];
            for(int k=0;k<Nc;++k)if(a.condensed[k]!=v.condensed[k])return false;
            for(int k=0;k<Ns;++k)if(a.pore[k]!=v.pore[k])return false;
            for(int k=0;k<Nr;++k)if(a.progress[k]!=v.progress[k])return false;
            if(a.energy!=v.energy||a.porosity!=v.porosity)return false;
        }
        return true;
    }
    Buffer<Real> initialVolumes,conductionVolumes;
    Buffer<ConductionReal> conductionTemperature;
    Buffer<ConductionReal> conductionCapacity,conductionSource,conductionConductivity;
    Buffer<MaterialThermalInput> conductionInput;
    Buffer<MaterialThermalCoefficients> conductionCoefficients;
    Buffer<MaterialConductionIteration> conductionReport;
    struct ConductionGraph{
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        ugkp::gpu::ExecutionGraphSignature signature;
        void clear(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);executable=nullptr;graph=nullptr;}
        ~ConductionGraph(){clear();}
    } conductionGraph;
    ConductionGraph localGraph;
    MaterialLocalBindings localBindings{};
    Buffer<ResidentMaterialState> localState;Buffer<ResidentMaterialStep> localStep;
    Buffer<MaterialWindowCommit> localCommit;
    Buffer<MaterialLocalProgress> localProgress;Buffer<ResidentWallClock> localClock;
    Buffer<ugkp::CompensatedTime> localTotal;Buffer<ResidentWallKnotTime> localKnots;
    Buffer<ugkwp::GasDriveControls> localDriveControls;Buffer<ugkwp::GasDriveSelection> localDrive;
    Buffer<int> localDriveCandidates,localGasFaces,localSolidFaces;
    Buffer<FilmAux> localAuxiliary,localCandidateAuxiliary;Buffer<Real> localRecession;
    Buffer<MaterialRecessionStatus> localRecessionStatus;
    Buffer<MaterialLocalAudit> localAudit;Buffer<SolidQ> localAuditState,localAuditRates;
    struct StreamedChunkStorage {
        Buffer<CoupledSurfaceDownload> surface;Buffer<CoupledWallDownload> wall;
        Buffer<ResidentMaterialDriveInterval> interval;
        MaterialRowStorage rows;Buffer<int> offsets,indices;
    } streamedChunks[2];
    StreamedMaterialChunk hostChunks[2]{};
    std::uint64_t chunkUse[2]{},chunkSequence=0;
    Buffer<StreamedMaterialChunk> streamedViews;
    Buffer<StreamedMaterialProgress> streamedProgress;
    Buffer<ResidentMaterialDriveInterval> streamedIntervals;
    Buffer<CoupledSurfaceDownload> streamedHeldSurface;Buffer<CoupledWallDownload> streamedHeldWall;
    Buffer<MaterialPacketAccumulator> streamedSourcePackets;Buffer<SolidQ> streamedTransported;
    StreamedMaterialBindings streamedBindings{};
    std::shared_ptr<CoupledHistoryArchive> localHistory;
    struct PendingPreparation {HostState state;WallProgram program;CpuMaterialControls controls;};
    std::shared_ptr<const PendingPreparation> pendingPreparation;
    bool localStreamed=false,localConduction=false,localHistoryPending=false;
    Buffer<int> localHistoryPrepared;
    const std::remove_pointer<decltype(MaterialSliceAccumulators::radiation)>::type* localExternalSource=nullptr;
    Buffer<int> localStage;
    ugkp::gpu::ImplicitDiffusion* localDiffusion=nullptr;
    ugkp::gpu::SparseConjugateGradient* localSparse=nullptr;
    Real localConductionDt=0,localNormalizationSmall=0,localSingularSmall=0;
    std::size_t localExternalFaces=0;
    std::shared_future<void> localCompletion;cudaEvent_t localReady=nullptr;
    bool localPrepared=false,localPending=false;Real localWindowEnd=0;
    Buffer<MaterialConductionAudit> conductionAuditParts,conductionAuditResult;Buffer<int> conductionExternal;std::vector<int> cachedConductionExternal;
    Buffer<Real> volumes,nextVolumes,sweeps,cellDt;
    Buffer<PhysicalReal> gradientSensitivity,stabilityRate,lossLeft,lossRight;
    Buffer<Vec3> centres,areas,faceCentres;
    Buffer<PhysicalVec3> pressureGradient,poreVelocity;
    Buffer<int> owner,neighbour,periodic,cellOffsets,cellFaces,cellSigns;
    Buffer<BoundaryKind> boundary;
    Buffer<int> diffusionOffsets,diffusionFaces,diffusionSides;
    Buffer<MaterialPrimitive> primitives;
    Buffer<SolidQ> faceFlux,rates;
    Buffer<int> cellErrors,faceErrors;
    Buffer<MaterialTransportDeviceResult> transportResult;
    Buffer<MaterialStepResult> sourceStatus,stepResult;
    Buffer<ExchangePacket> packets;
    Buffer<MaterialPacketAccumulator> streamedPacketRows;
    ugkp::gpu::PinnedTransferQueue streamedPacketTransfers;
    ResidentMaterialDriveHistory borrowedHistory{};
    ugkp::IntervalSlicePlan pendingSlicePlan{};
    Buffer<IntervalSliceRecordPart> sliceParts;
    Real sliceExternalFraction=0;bool sliceExternalLast=false;
    bool sliceReady=false,slicePending=false,deviceRadiation=false,externalSlicePending=false;
    Buffer<ugkp::IntervalSliceCursor> sliceCursor;
    Buffer<ugkp::IntervalSlicePlan> slicePlan;
    Buffer<ExchangePacket> sliceUsedPackets,sliceCandidatePackets;
    Buffer<EnergyReal> sliceUsedRadiation,sliceCandidateRadiation,sliceExternal;
    Buffer<EnergyReal> sliceUsedExternal,sliceCandidateExternal;
    MaterialSliceAccumulators currentSlice()const{return {sliceUsedPackets.data(),sliceUsedRadiation.data(),sliceUsedExternal.data()};}
    MaterialSliceAccumulators candidateSlice()const{return {sliceCandidatePackets.data(),sliceCandidateRadiation.data(),sliceCandidateExternal.data()};}
    ResidentMaterialSlice sourceSlice()const{
        ResidentMaterialSlice slice{borrowedHistory,slicePlan.data(),currentSlice(),true};
        slice.selectedParts=sliceParts.data();slice.externalFraction=sliceExternalFraction;
        slice.externalLast=sliceExternalLast;slice.hostSelected=true;return slice;
    }

    Buffer<int> packetOffsets,packetIndices,radiationOffsets,radiationFaces,radiationCells;
    MaterialRowStorage packetRows,radiationRows;Buffer<unsigned char> rowScratch;
    std::vector<int> radiationCellMap;
    Buffer<EnergyReal> radiation;
    Buffer<unsigned> radiationReceiver;
    MaterialSurfaceView surfacePrototype{};
    bool surfacePrototypeValid=false,surfaceCooperative=false,localEndpoint=false;int surfaceDepth=1;
    MaterialEndpointBindings endpointBindings{};ugkp::gpu::ScalarWorkLaunch endpointLaunch{};
    Buffer<SurfacePhysicsInput> endpointInput;Buffer<MaterialSurfaceView> endpointOperation;
    Buffer<SurfacePhysicsResult> endpointPhysics;Buffer<WallFaceSample> endpointWall;
    Buffer<MaterialSurfaceStatus> endpointStatus;Buffer<int> endpointComplete,endpointGasCells;
    Buffer<Real> endpointGasVolume;Buffer<GpuReal> endpointDonorShare,endpointHostFraction;
    Buffer<CoupledSurfaceDownload> endpointHistorySurface;Buffer<CoupledWallDownload> endpointHistoryWall;
    Buffer<SurfacePhysicsInput> surfaceInput;
    Buffer<int> surfaceCells;
    Buffer<unsigned char> surfaceActive,wallWorkspace;
    Buffer<SurfacePhysicsResult> surfacePhysics;
    Buffer<WallFaceSample> surfaceWall;
    Buffer<MaterialSurfaceStatus> surfaceStatus;
    Buffer<ugkwp::gaswall::WallOutput<PhysicalReal,Ns>> surfaceTeamOutput;
    bool surfaceProbePlanReady=false;int surfaceProbeResident2=0,surfaceProbeResident3=0;
    Buffer<PhysicalReal> quadratureDistance,quadratureWeight,mechanismBasis;
    Buffer<ugkwp::GasReactionData<PhysicalReal>> mechanismReactions;
    Buffer<ugkwp::GasStoichTerm<PhysicalReal>> mechanismReactants,mechanismProducts;
    Buffer<ugkwp::GasColliderEfficiency<PhysicalReal>> mechanismEfficiencies;
    ugkwp::GasMechanismView<PhysicalReal,Ns> deviceMechanism{};
    bool mechanismReady=false;
    MeshGeometryCache meshGeometry;
    int cells=0,faces=0;
    cudaEvent_t conductionReady=nullptr;
    ~MaterialBackend(){if(localCompletion.valid()){try{localCompletion.get();}catch(...){}}stream.sync();if(localReady)cudaEventDestroy(localReady);if(conductionReady)cudaEventDestroy(conductionReady);}
    bool waitConduction(void* producer){
        if(!conductionReady&&!fault.check(cudaEventCreateWithFlags(&conductionReady,cudaEventDisableTiming),"create conduction dependency"))return false;
        return fault.check(cudaEventRecord(conductionReady,static_cast<cudaStream_t>(producer)),"record conduction dependency")
            &&fault.check(cudaStreamWaitEvent(stream.get(),conductionReady,0),"wait conduction dependency");
    }
    template<class T> bool allocate(Buffer<T>& buffer,std::size_t size){return buffer.resize(size,fault);}
    template<class T> bool uploadRaw(Buffer<T>& buffer,const T* data,std::size_t size){
        return allocate(buffer,size)&&transfers.upload(buffer.data(),data,size);
    }
    template<class T> bool upload(Buffer<T>& buffer,const std::vector<T>& data){return uploadRaw(buffer,data.data(),data.size());}
    template<class T> bool read(const Buffer<T>& buffer,std::vector<T>& data){
        data.resize(buffer.size());return transfers.download(data.data(),buffer.data(),buffer.size());
    }
    template<class Row> bool buildRows(int items,int rows,Row row,MaterialRowStorage& storage,
        Buffer<int>& offsets,Buffer<int>& indices){
        const auto reserve=[&](Buffer<int>& buffer,std::size_t count){
            return buffer.size()>=count||allocate(buffer,count);
        };
        if(!reserve(storage.keys,items)||!reserve(storage.sortedKeys,items)
            ||!reserve(storage.inputIndices,items)||!reserve(storage.counts,rows+1)
            ||!reserve(offsets,rows+1)||!reserve(indices,items))return false;
        ugkp::gpu::StableRowIndexView view{storage.keys.data(),storage.sortedKeys.data(),
            storage.inputIndices.data(),indices.data(),storage.counts.data(),offsets.data()};
        std::size_t bytes=0;
        if(!fault.check(ugkp::gpu::stableRowIndexWorkspace(items,rows,view,bytes,stream.get()),
            "size stable material row index")||(rowScratch.size()<bytes&&!allocate(rowScratch,bytes)))return false;
        return fault.check(ugkp::gpu::buildStableRowIndex(items,rows,row,view,rowScratch.data(),bytes,stream.get()),
            "build stable material row index");
    }
    bool failure(std::string& error){transfers.drain();if(!fault.message.empty())error=fault.message;return false;}
    bool finish(std::string& error){
        ugkp::diagnostics::DeviceContext snapshot{};
        if(!transfers.download(&snapshot,diagnostics.device(),1)||!transfers.finish())return failure(error);
        diagnostics.consumeCompletedSnapshot(snapshot);
        if(diagnostics.dispatchPending()){error="fatal material device diagnostic";return false;}
        return true;
    }
    void operation(Real t){
        time=t;diagnostics.setOperationContext(sequence,step,time);
        ugkp::diagnostics::beginOperationKernel<<<1,1,0,stream.get()>>>(diagnostics.device(),sequence,step,time);
    }
    MaterialMeshView mesh()const{
        MaterialMeshView m;m.cells=cells;m.faces=faces;m.volumes=volumes.data();m.cellCentres=centres.data();
        m.owner=owner.data();m.neighbour=neighbour.data();m.areaVectors=areas.data();m.faceCentres=faceCentres.data();
        m.boundaryKind=boundary.data();m.periodicPartner=periodic.data();m.cellFaceOffsets=cellOffsets.data();
        m.cellFaces=cellFaces.data();m.cellFaceSigns=cellSigns.data();m.diffusionOffsets=diffusionOffsets.data();
        m.diffusionFaces=diffusionFaces.data();m.diffusionSides=diffusionSides.data();return m;
    }
    MaterialTransportWorkspace workspace()const{
        MaterialTransportWorkspace w;w.primitive=primitives.data();w.pressureGradient=pressureGradient.data();
        w.poreVelocity=poreVelocity.data();w.gradientSensitivity=gradientSensitivity.data();w.stabilityRate=stabilityRate.data();
        w.faceDiffusionLossLeft=lossLeft.data();w.faceDiffusionLossRight=lossRight.data();w.cellDtLimit=cellDt.data();
        w.faceFlux=faceFlux.data();w.rates=rates.data();w.cellErrors=cellErrors.data();w.faceErrors=faceErrors.data();return w;
    }
    bool configureMesh(const HostMesh& m,std::string& error){
        if(current.size()==0){cells=faces=0;meshGeometry.clear();return true;}
        if(m.volumes.size()==current.size()&&meshGeometry.matches(m))return true;
        if(!validateMaterialTransportMesh(m,current.size(),error))return false;
        cells=int(m.volumes.size());faces=int(m.owner.size());
        if(!upload(volumes,m.volumes)||!upload(centres,m.cellCentres)||!upload(areas,m.areaVectors)
            ||!upload(faceCentres,m.faceCentres))return failure(error);
        if(!meshGeometry.topologyMatches(m)){
            if(!upload(owner,m.owner)||!upload(neighbour,m.neighbour)||!upload(periodic,m.periodicPartner)
                ||!upload(boundary,m.boundaryKind)||!upload(cellOffsets,m.cellFaceOffsets)
                ||!upload(cellFaces,m.cellFaces)||!upload(cellSigns,m.cellFaceSigns))return failure(error);
            std::vector<std::vector<std::pair<int,int>>> rows(cells);
            for(int f=0;f<faces;++f){
                if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
                int right=m.neighbour[f];
                if(m.boundaryKind[f]==BoundaryKind::Periodic){
                    if(f>m.periodicPartner[f])continue;
                    right=m.owner[m.periodicPartner[f]];
                }
                if(right<0)continue;
                rows[m.owner[f]].push_back({f,0});rows[right].push_back({f,1});
            }
            std::vector<int> offsets(cells+1),indices,sides;
            for(int c=0;c<cells;++c){for(const auto& pair:rows[c]){indices.push_back(pair.first);sides.push_back(pair.second);}offsets[c+1]=int(indices.size());}
            if(!upload(diffusionOffsets,offsets)||!upload(diffusionFaces,indices)||!upload(diffusionSides,sides))return failure(error);
        }
        if(!allocate(primitives,cells)||!allocate(pressureGradient,cells)||!allocate(poreVelocity,cells)
            ||!allocate(gradientSensitivity,cells)||!allocate(stabilityRate,cells)||!allocate(cellDt,cells)
            ||!allocate(lossLeft,faces)||!allocate(lossRight,faces)||!allocate(faceFlux,faces)||!allocate(rates,cells)
            ||!allocate(cellErrors,cells)||!allocate(faceErrors,faces)||!allocate(transportResult,1))return failure(error);
        meshGeometry.remember(m);return true;
    }
    bool configureMechanism(const ugkwp::GasMechanismView<PhysicalReal,Ns>& m,std::string& error){
        if(mechanismReady&&deviceMechanism.mechanismHash==m.mechanismHash
            &&deviceMechanism.speciesOrderHash==m.speciesOrderHash)return true;
        if(!uploadRaw(mechanismReactions,m.reactions,m.reactionCount)
            ||!uploadRaw(mechanismReactants,m.reactants,m.reactantTermCount)
            ||!uploadRaw(mechanismProducts,m.products,m.productTermCount)
            ||!uploadRaw(mechanismEfficiencies,m.efficiencies,m.efficiencyCount)
            ||!uploadRaw(mechanismBasis,m.stoichiometricBasis,Ns*m.independentRank))return failure(error);
        deviceMechanism=m;deviceMechanism.reactions=mechanismReactions.data();deviceMechanism.reactants=mechanismReactants.data();
        deviceMechanism.products=mechanismProducts.data();deviceMechanism.efficiencies=mechanismEfficiencies.data();
        deviceMechanism.stoichiometricBasis=mechanismBasis.data();mechanismReady=true;return true;
    }
};

MaterialBackend* createMaterialBackend(std::string& error){
    std::unique_ptr<MaterialBackend> b(new MaterialBackend);
    if(!b->stream.create(b->fault)){error=b->fault.message;return nullptr;}
    b->transfers.bindStream(b->stream.get());
    if(!b->fault.check(b->diagnostics.initialise(b->stream.get()),"initialize material diagnostics")){
        error=b->fault.message;return nullptr;
    }
    error.clear();return b.release();
}
void destroyMaterialBackend(MaterialBackend* b)noexcept{delete b;}
void setMaterialDiagnosticsSink(MaterialBackend& b,const ugkp::diagnostics::HostSink& sink){b.diagnostics.setSink(sink);}
bool beginMaterialWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,std::string& error,bool collectDiagnostics){
    if(b.localPending){error="material local window still owns its buffers";return false;}
    b.physics=physics;
    auto quietPhysics=physics;quietPhysics.diagnostics=nullptr;quietPhysics.tolerances.diagnostics=nullptr;
    auto devicePhysics=physics;devicePhysics.diagnostics=b.diagnostics.device();
    devicePhysics.tolerances.diagnostics=b.diagnostics.device();
    b.sequence=state.commitSequence+1;b.step=state.acceptedSteps;b.time=state.time;
    if(!b.retains(state.solid)){
        b.publishedCurrentValid=false;
        if(!b.upload(b.current,state.solid))return b.failure(error);
        b.publishedCurrent=state.solid;b.publishedCurrentValid=true;
    }
    if(!b.configureMesh(state.solidMesh,error))return false;
    if(state.solid.empty()){
        if(!b.allocate(b.initialVolumes,0)||!b.allocate(b.nextVolumes,0))return b.failure(error);
    }else if(!b.initialVolumes.copy(b.volumes,b.fault,b.stream.get())
        ||!b.nextVolumes.copy(b.volumes,b.fault,b.stream.get()))return b.failure(error);
    if(!b.initial.copy(b.current,b.fault,b.stream.get())
        ||!b.allocate(b.next,state.solid.size())
        ||!b.allocate(b.gasChange,state.solid.size())||!b.gasChange.zero(b.stream.get())
        ||!b.allocate(b.candidateGasChange,state.solid.size())
        ||!b.allocate(b.sourceStatus,state.solid.size())||!b.allocate(b.stepResult,1)
        ||!b.uploadRaw(b.devicePhysics,&devicePhysics,1)||!b.uploadRaw(b.quietSurfacePhysics,&quietPhysics,1))return b.failure(error);
    b.operation(state.time);
    return collectDiagnostics?b.finish(error):true;
}
bool prepareResidentMaterialConduction(MaterialBackend& b,const HostMesh& mesh,
    const std::vector<Real>& nativeVolumes,std::vector<MaterialThermalInput>& input,
    std::string& error,bool& numerical,bool publishInput){
    numerical=true;
    if(!b.configureMesh(mesh,error))return false;
    if(!b.upload(b.conductionVolumes,nativeVolumes)||!b.allocate(b.conductionInput,b.current.size())
        ||!b.allocate(b.conductionCoefficients,b.current.size())||!b.allocate(b.conductionCapacity,b.current.size())
        ||!b.allocate(b.conductionSource,b.current.size())||!b.allocate(b.conductionConductivity,b.current.size()))return b.failure(error);
    b.operation(b.time);
    if(b.cells)prepareConductionKernel<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.cells,
        b.initial.data(),b.initialVolumes.data(),b.current.data(),b.volumes.data(),b.conductionVolumes.data(),
        b.devicePhysics.data(),b.conductionInput.data(),b.sourceStatus.data());
    selectConductionFailure<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialStepResult),b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
    MaterialStepResult result;
    if(!b.fault.check(cudaGetLastError(),"prepare resident material conduction")
        ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.phase){numerical=result.numerical;error=result.phase==1
        ?materialCaloricError(MaterialCaloricCode(result.code)):"CPU conduction old EOS recovery failed";return false;}
    if(!b.allocate(b.conductionTemperature,b.current.size()))return b.failure(error);
    if(b.cells)conductionAnchorTemperature<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.cells,b.conductionInput.data(),b.conductionTemperature.data());
    if(publishInput){if(!b.read(b.conductionInput,input)||!b.finish(error))return b.failure(error);}
    else input.clear();
    error.clear();return true;
}
bool publishResidentMaterialConductionInput(MaterialBackend& b,std::vector<MaterialThermalInput>& input,std::string& error){
    if(!b.read(b.conductionInput,input)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool evaluateResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    Real dt,std::vector<MaterialThermalCoefficients>& coefficients,std::string& error,
    const ConductionReal* deviceTemperature,void* temperatureStream,bool download,bool startupDiagnostics){
    if(deviceTemperature){if(!b.waitConduction(temperatureStream))return b.failure(error);}
    else {const std::vector<ConductionReal> encoded(temperature.begin(),temperature.end());
        if(!b.upload(b.conductionTemperature,encoded))return b.failure(error);}
    b.operation(b.time);
    if(b.cells)conductionCoefficientsKernel<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.cells,
        b.conductionInput.data(),deviceTemperature?deviceTemperature:b.conductionTemperature.data(),dt,b.devicePhysics.data(),b.conductionCoefficients.data(),
        b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data());
    if(!b.fault.check(cudaGetLastError(),"evaluate resident material conduction"))return b.failure(error);
    if(startupDiagnostics){
        if(b.cells)inspectConductionStartupKernel<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(
            b.cells,b.conductionCoefficients.data(),b.devicePhysics.data(),b.sourceStatus.data());
        selectConductionFailure<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialStepResult),b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
        MaterialStepResult result;
        if(!b.fault.check(cudaGetLastError(),"inspect resident startup coefficients")
            ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
        if(result.phase){error=result.code==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve";return false;}
    }
    if(download){if(!b.read(b.conductionCoefficients,coefficients)||!b.finish(error))return b.failure(error);}
    else coefficients.clear();
    error.clear();return true;
}
MaterialConductionDeviceFields residentMaterialConductionFields(MaterialBackend& b){
    return {b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data(),b.stream.get(),b.conductionTemperature.data(),b.current.size()};
}
namespace {
const MaterialConductionAudit* enqueueConductionAudit(MaterialBackend& b,const ConductionReal* temperature,const ConductionReal* previous,
    const ConductionReal* residual,const ConductionReal* flux,int boundary,Real dt,cudaStream_t stream){
    const int blocks=(std::max(b.cells,boundary)+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads;
    if(blocks)conductionAuditParts<<<blocks,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialConductionAudit),stream>>>(b.cells,boundary,b.conductionInput.data(),b.conductionCoefficients.data(),
        temperature,previous,residual,flux,b.conductionExternal.data(),dt,b.conductionAuditParts.data());
    conductionAuditFinish<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialConductionAudit),stream>>>(blocks,b.conductionAuditParts.data(),b.conductionAuditResult.data());
    return b.conductionAuditResult.data();
}
}
static ugkp::gpu::ExecutionGraphNode appendMaterialConductionLoop(MaterialBackend& b,
    ugkp::gpu::ImplicitDiffusion& diffusion,ugkp::gpu::SparseConjugateGradient& sparse,
    ugkp::gpu::ExecutionGraph parent,ugkp::gpu::ExecutionGraphNode dependency,
    std::size_t boundary,Real dt,const CpuMaterialControls& controls,Real normalizationSmall,Real singularSmall){
    using namespace ugkp::gpu;using executionGraphDetail::appendCaptured;
    auto& report=b.conductionReport;
        cudaGraphConditionalHandle handle,permitted,assembleAllowed,finishAllowed;
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&handle,parent,0,cudaGraphCondAssignDefault));
        ConjugateGradientControls linear;linear.maximumIterations=controls.linearMaxIterations;linear.normalizationSmall=normalizationSmall;linear.singularSmall=std::max(ConductionReal(singularSmall),std::numeric_limits<ConductionReal>::min());
        const auto initialized=appendCaptured(parent,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},b.stream.get(),[&]{
            beginConductionLoop<<<1,1,0,b.stream.get()>>>(report.data(),linear,b.diagnostics.device(),handle);});
        cudaGraph_t body=nullptr;const auto done=executionGraphDetail::addWhile(parent,handle,{initialized},body);
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&permitted,body,0,cudaGraphCondAssignDefault));
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&assembleAllowed,body,0,cudaGraphCondAssignDefault));
        const auto assemblySelected=appendCaptured(body,{},b.stream.get(),[&]{conductionSelect<<<1,1,0,b.stream.get()>>>(report.data(),1,sparse.capacityPendingDevice(),assembleAllowed);});
        cudaGraph_t assemblyBody=nullptr;const auto assemblyDone=executionGraphDetail::addIf(body,assembleAllowed,{assemblySelected},assemblyBody);
        const auto assembled=appendCaptured(assemblyBody,{},b.stream.get(),[&]{
            diffusion.enqueueResidentSnapshot(b.stream.get());
            if(b.cells)conductionCoefficientsKernel<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.cells,b.conductionInput.data(),
                diffusion.deviceTemperature(),dt,b.devicePhysics.data(),b.conductionCoefficients.data(),b.conductionCapacity.data(),b.conductionSource.data(),b.conductionConductivity.data());
            diffusion.enqueueResidentAssembly(b.stream.get());
        });
        const auto residual=sparse.appendResidentLocalResidual(assemblyBody,assembled,diffusion.deviceSolution(),diffusion.deviceResidual());
        const auto budget=appendCaptured(assemblyBody,{residual},b.stream.get(),[&]{
            const auto audit=enqueueConductionAudit(b,diffusion.deviceTemperature(),nullptr,diffusion.deviceResidual(),nullptr,boundary,dt,b.stream.get());
            prepareConductionLinearControls<<<1,1,0,b.stream.get()>>>(report.data(),audit,controls,permitted);
            conductionPrepared<<<1,1,0,b.stream.get()>>>(report.data());
        });
        const auto solveSelected=appendCaptured(body,{assemblyDone},b.stream.get(),[&]{conductionSelect<<<1,1,0,b.stream.get()>>>(report.data(),2,sparse.capacityPendingDevice(),permitted);});
        cudaGraph_t solveBody=nullptr;const auto solve=executionGraphDetail::addIf(body,permitted,{solveSelected},solveBody);
        const auto solved=sparse.appendResidentLocalLdu(solveBody,nullptr,diffusion.deviceSolution(),&report.data()->linear);
        executionGraphDetail::checked(cudaGraphConditionalHandleCreate(&finishAllowed,solveBody,0,cudaGraphCondAssignDefault));
        const auto finishSelected=appendCaptured(solveBody,{solved},b.stream.get(),[&]{conductionSelect<<<1,1,0,b.stream.get()>>>(report.data(),2,sparse.capacityPendingDevice(),finishAllowed);});
        cudaGraph_t finishBody=nullptr;executionGraphDetail::addIf(solveBody,finishAllowed,{finishSelected},finishBody);
        appendCaptured(finishBody,{},b.stream.get(),[&]{
            diffusion.enqueueResidentBoundaryTemperature(b.stream.get());diffusion.enqueueResidentBoundaryFlux(b.stream.get());
            enqueueConductionAudit(b,diffusion.deviceTemperature(),diffusion.devicePreviousSolution(),nullptr,diffusion.deviceBoundaryFlux(),boundary,dt,b.stream.get());
            finishConductionIteration<<<1,1,0,b.stream.get()>>>(report.data(),b.conductionAuditResult.data(),sparse.deviceReport(),controls,b.diagnostics.device(),handle);
        });
        appendCaptured(body,{solve},b.stream.get(),[&]{
            conductionContinue<<<1,1,0,b.stream.get()>>>(report.data(),sparse.capacityPendingDevice(),controls,handle);
        });
    return done;
}
bool solveResidentMaterialConduction(MaterialBackend& b,ugkp::gpu::ImplicitDiffusion& diffusion,
    ugkp::gpu::SparseConjugateGradient& sparse,const std::vector<int>& external,Real dt,const CpuMaterialControls& controls,
    Real normalizationSmall,Real singularSmall,std::string& error){
    if(!b.waitConduction(diffusion.deviceValues().readyStream))return b.failure(error);
    if(b.cachedConductionExternal!=external){if(!b.upload(b.conductionExternal,external))return b.failure(error);b.cachedConductionExternal=external;}
    const int blocks=(std::max(b.cells,static_cast<int>(external.size()))+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads;
    auto& report=b.conductionReport;
    if(!b.allocate(b.conductionAuditParts,blocks)||!b.allocate(b.conductionAuditResult,1)||!b.allocate(report,1))return b.failure(error);



    auto& graph=b.conductionGraph;
    auto signature=diffusion.residentExecutionSignature();
    signature.append(sparse.residentExecutionSignature());
    signature.add(ugkp::gpu::workPlan().materialBlockThreads);signature.add(b.cells);signature.add(external.size());signature.add(dt);
    signature.add(normalizationSmall);signature.add(singularSmall);
    signature.add(controls.nonlinearMaxIterations);signature.add(controls.nonlinearRelativeTolerance);
    signature.add(controls.energyAbsoluteTolerance);signature.add(controls.energyRelativeTolerance);
    signature.add(controls.linearMaxIterations);signature.add(controls.linearEnergyBudgetFraction);
    for(const void* pointer:{static_cast<const void*>(report.data()),static_cast<const void*>(b.diagnostics.device()),
        static_cast<const void*>(b.conductionInput.data()),static_cast<const void*>(b.devicePhysics.data()),
        static_cast<const void*>(b.conductionCoefficients.data()),static_cast<const void*>(b.conductionCapacity.data()),
        static_cast<const void*>(b.conductionSource.data()),static_cast<const void*>(b.conductionConductivity.data()),
        static_cast<const void*>(b.conductionAuditParts.data()),static_cast<const void*>(b.conductionAuditResult.data()),
        static_cast<const void*>(b.conductionExternal.data())})signature.add(pointer);
    if(!(graph.signature==signature)){graph.clear();graph.signature=std::move(signature);}
    using namespace ugkp::gpu;using executionGraphDetail::appendCaptured;
    try{
        if(!graph.executable){
        executionGraphDetail::checked(cudaGraphCreate(&graph.graph,0));
        appendMaterialConductionLoop(b,diffusion,sparse,graph.graph,nullptr,external.size(),dt,
            controls,normalizationSmall,singularSmall);
        executionGraphDetail::checked(cudaGraphInstantiateWithFlags(&graph.executable,graph.graph,0));
        }
        executionGraphDetail::checked(cudaMemsetAsync(&report.data()->continuationStage,0,sizeof(int),b.stream.get()));
        MaterialConductionIteration result;
        do{
            executionGraphDetail::checked(cudaGraphLaunch(graph.executable,b.stream.get()));
            if(!b.transfers.download(&result,report.data(),1)||!b.finish(error))return b.failure(error);
        }while(sparse.servicePendingCapacity());
        if(!result.valid){
            error=result.failure==1?(result.audit.coefficientError==1?"nonpositive implicit caloric secant":"material mass/geometry incompatible during implicit solve")
                :result.failure==2?"nonfinite OpenFOAM implicit material sparse residual"
                :result.failure==3?"nonfinite or nonpositive implicit material temperature":"nonfinite implicit material residual";
            return false;
        }
        error.clear();return true;
    }catch(const std::exception& exception){graph.clear();error=exception.what();return false;}
}
bool auditResidentMaterialConduction(MaterialBackend& b,const ConductionReal* temperature,const ConductionReal* previous,
    const ConductionReal* residual,const ConductionReal* flux,const std::vector<int>& external,Real dt,void* stream,
    MaterialConductionAudit& audit,std::string& error){
    if(!b.waitConduction(stream))return b.failure(error);
    if(b.cachedConductionExternal!=external){if(!b.upload(b.conductionExternal,external))return b.failure(error);b.cachedConductionExternal=external;}
    const int count=std::max(b.cells,static_cast<int>(external.size())),blocks=(count+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads;
    if(!b.allocate(b.conductionAuditParts,blocks)||!b.allocate(b.conductionAuditResult,1))return b.failure(error);
    if(blocks)conductionAuditParts<<<blocks,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialConductionAudit),b.stream.get()>>>(b.cells,external.size(),b.conductionInput.data(),b.conductionCoefficients.data(),
        temperature,previous,residual,flux,b.conductionExternal.data(),dt,b.conductionAuditParts.data());
    conductionAuditFinish<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialConductionAudit),b.stream.get()>>>(blocks,b.conductionAuditParts.data(),b.conductionAuditResult.data());
    if(!b.fault.check(cudaGetLastError(),"reduce resident material conduction")
        ||!b.transfers.download(&audit,b.conductionAuditResult.data(),1)||!b.finish(error))return b.failure(error);
    error.clear();return true;
}
bool finishResidentMaterialConduction(MaterialBackend& b,const std::vector<Real>& temperature,
    std::vector<SolidQ>& hostMirror,std::string& error,bool& numerical,const ConductionReal* deviceTemperature,void* temperatureStream){
    numerical=true;
    if(deviceTemperature){if(!b.waitConduction(temperatureStream))return b.failure(error);}
    else {const std::vector<ConductionReal> encoded(temperature.begin(),temperature.end());
        if(!b.upload(b.conductionTemperature,encoded))return b.failure(error);}
    b.operation(b.time);
    if(b.cells)finishConductionKernel<<<(b.cells+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.cells,b.current.data(),
        b.volumes.data(),b.conductionInput.data(),deviceTemperature?deviceTemperature:b.conductionTemperature.data(),b.devicePhysics.data(),
        b.next.data(),b.sourceStatus.data());
    selectConductionFailure<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialStepResult),b.stream.get()>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
    MaterialStepResult result;
    if(!b.fault.check(cudaGetLastError(),"finish resident material conduction")
        ||!b.transfers.download(&result,b.stepResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.phase){numerical=result.numerical;error="implicit material endpoint EOS invalid";return false;}
    std::vector<SolidQ> candidate;
    if(!b.read(b.next,candidate)||!b.finish(error))return b.failure(error);
    b.current.swap(b.next);hostMirror.swap(candidate);
    b.publishedCurrent=hostMirror;b.publishedCurrentValid=true;
    error.clear();return true;
}
bool evaluateResidentMaterialTransport(MaterialBackend& b,const HostMesh& mesh,
    const std::vector<Real>& swept,Real interval,PhysicalReal cfl,Real& dtLimit,std::string& error,bool& numerical){
    numerical=true;
    if(!(interval>0)||!finite(interval)||!finite(cfl)||!(cfl>0)||(!swept.empty()&&swept.size()!=mesh.owner.size())){
        error="invalid material transport interval/control";return false;
    }
    if(b.current.size()==0){dtLimit=std::numeric_limits<Real>::max();error.clear();return true;}
    if(!b.configureMesh(mesh,error))return false;
    if(!b.upload(b.sweeps,swept))return b.failure(error);
    b.operation(b.time);
    if(!b.fault.check(launchMaterialTransport(b.mesh(),b.current.data(),b.devicePhysics.data(),
        swept.empty()?nullptr:b.sweeps.data(),interval,cfl,b.workspace(),b.transportResult.data(),b.stream.get()),
        "launch resident material transport"))return b.failure(error);
    MaterialTransportDeviceResult result;
    if(!b.transfers.download(&result,b.transportResult.data(),1)||!b.finish(error))return b.failure(error);
    if(result.errorCode){numerical=result.numerical;error=materialTransport::errorMessage(MaterialTransportError(result.errorCode));
        if(result.errorCode==int(MaterialTransportError::CellEos))error+=std::to_string(result.entity);
        return false;}
    numerical=result.numerical;dtLimit=result.dtLimit;error.clear();return true;
}
static bool initializeMaterialSliceStorage(MaterialBackend& b,std::size_t faces,std::string& error){
    const ugkp::IntervalSliceCursor initialCursor{};
    if(!b.uploadRaw(b.sliceCursor,&initialCursor,1)||!b.allocate(b.slicePlan,1)
        ||!b.allocate(b.sliceUsedPackets,3*faces)||!b.allocate(b.sliceCandidatePackets,3*faces)
        ||!b.allocate(b.sliceUsedRadiation,faces)||!b.allocate(b.sliceCandidateRadiation,faces)
        ||!b.sliceUsedPackets.zero(b.stream.get())||!b.sliceUsedRadiation.zero(b.stream.get()))return b.failure(error);
    b.sliceReady=true;return true;
}
bool beginResidentMaterialHistory(MaterialBackend& b,const ResidentMaterialDriveHistory* supplied,
    const IntervalHistory& host,std::string& error){
    b.sliceReady=false;b.slicePending=false;b.borrowedHistory={};
    b.deviceRadiation=false;b.externalSlicePending=false;
    const auto external=host.externalRadiationDevice();
    if(external.faces){
        if(!external.energy){error="missing device boundary radiation energy";return false;}
        if(!b.waitConduction(external.producerStream)||!b.allocate(b.sliceExternal,external.faces)
            ||!b.fault.check(cudaMemcpyAsync(b.sliceExternal.data(),external.energy,external.faces*sizeof(EnergyReal),
                cudaMemcpyDeviceToDevice,b.stream.get()),"borrow device boundary radiation"))return b.failure(error);
        b.deviceRadiation=true;
    }else if(!b.upload(b.sliceExternal,host.externalRadiationEnergy()))return b.failure(error);
    if(!b.allocate(b.sliceUsedExternal,b.sliceExternal.size())||!b.allocate(b.sliceCandidateExternal,b.sliceExternal.size())
        ||!b.sliceUsedExternal.zero(b.stream.get()))return b.failure(error);



    if(!supplied||!supplied->surface||!supplied->interval||!supplied->faces
        ||supplied->frames!=host.records().size())return true;
    for(std::size_t i=0;i<host.records().size();++i){const auto record=host.readRecord(i);
        if(record.radiationEnergy.size()!=supplied->faces||record.radiationReceiver.size()!=supplied->faces)return true;}
    b.borrowedHistory=*supplied;b.pendingSlicePlan={};
    if(!b.waitConduction(supplied->producerStream))return b.failure(error);
    return initializeMaterialSliceStorage(b,supplied->faces,error);
}

namespace {
struct MaterialRadiationCellRow {
    const int* cells=nullptr;
    __host__ __device__ int operator()(int i)const{return cells[i];}
};
}
static bool prepareMaterialLocalWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,
    const IntervalHistory& history,const ResidentMaterialDriveHistory& resident,const WallProgram& program,
    const CpuMaterialControls& controls,std::string& error,bool streamed,bool pending=false){
    if(b.localPending){error="material local window still owns its buffers";return false;}
    b.localPrepared=false;b.localCompletion={};b.localConduction=false;b.localEndpoint=false;
    b.localHistoryPending=pending;b.localExternalSource=nullptr;
    b.localDiffusion=nullptr;b.localSparse=nullptr;
    const std::size_t nf=state.surface.area.size();
    if(physics.meshMotion.policy!=MeshMotionPolicy::Static){error="material local window lacks moving-geometry stages";return false;}
    if(controls.retainDonorHistory){error="material local window lacks requested external donor-history storage";return false;}
    if((!pending&&(!history.complete()||resident.frames!=history.records().size()))||!resident.interval
        ||resident.faces!=nf||(nf&&!streamed&&!resident.surface)){
        error="material local window needs its complete resident driving history";return false;
    }
    if(state.surface.gasFace.size()!=nf||state.surface.solidFace.size()!=nf
        ||state.surface.solidCell.size()!=nf||state.filmAux.size()!=nf){
        error="material local window face or auxiliary data missing";return false;
    }
    if(nf&&(streamed?CoupledHistoryArchive::slabFrames:resident.frames)>std::size_t(INT_MAX)/nf/3){error="material local window packet storage is not addressable";return false;}
    for(std::size_t f=0;f<nf;++f){
        const int c=state.surface.solidCell[f],sf=state.surface.solidFace[f];
        if(c<-1||c>=int(state.solid.size())||sf<-1||sf>=int(state.solidMesh.owner.size())){
            error="material local window has an invalid material face owner";return false;
        }
    }
    if(!beginMaterialWindow(b,state,physics,error,false)
        ||!beginResidentMaterialHistory(b,streamed?nullptr:&resident,history,error))return false;
    if(streamed||pending){if(!initializeMaterialSliceStorage(b,nf,error))return false;b.borrowedHistory=resident;}
    if(pending){const int initial=0;if(!b.uploadRaw(b.localHistoryPrepared,&initial,1))return b.failure(error);
        b.localExternalSource=history.externalRadiationDevice().energy;}
    if(!b.sliceReady){error="material local window history lacks source-slice data";return false;}
    const auto total=residentWindowTime(intervalDuration(history.interval()));
    const ResidentWallClock clock{{},history.interval().begin,history.interval().identity.sequence};
    auto mesh=b.mesh(),candidateMesh=mesh;candidateMesh.volumes=b.nextVolumes.data();
    const ResidentMaterialState material{b.current.data(),b.next.data(),b.gasChange.data(),
        b.candidateGasChange.data(),b.initial.data(),mesh,candidateMesh};
    std::vector<ResidentWallKnotTime> knots;
    for(const auto& knot:program.knots)knots.push_back({residentWindowTime(knotOffset(history.interval(),knot)),knot.time});

    std::vector<MaterialLocalAudit> observations;
    const auto sampleCount=std::min(resident.frames?resident.frames-1:0,
        controls.driveSourceDiagnosticSamples>0?std::size_t(controls.driveSourceDiagnosticSamples):std::size_t(0));
    if(resident.frames)observations.push_back({0,{}});
    for(std::size_t i=1;i<=sampleCount;++i){
        const auto record=(resident.frames/(sampleCount+1))*i+((resident.frames%(sampleCount+1))*i)/(sampleCount+1);
        observations.push_back({record,pending?ugkp::CompensatedTime{}:residentWindowTime(history.records()[record].localBegin)});
    }
    if(b.current.size()&&observations.size()>std::numeric_limits<std::size_t>::max()/b.current.size()){
        error="material diagnostic snapshot extent cannot be represented";return false;
    }
    if(!b.upload(b.localAudit,observations)
        ||!b.allocate(b.localAuditState,observations.size()*b.current.size())
        ||!b.allocate(b.localAuditRates,observations.size()*b.current.size()))return b.failure(error);
    const ResidentMaterialStep step{};const MaterialLocalProgress progress{};const MaterialWindowCommit commit{};
    if(!b.uploadRaw(b.localState,&material,1)||!b.uploadRaw(b.localStep,&step,1)
        ||!b.uploadRaw(b.localTotal,&total,1)||!b.uploadRaw(b.localClock,&clock,1)
        ||!b.uploadRaw(b.localCommit,&commit,1)||!b.uploadRaw(b.localProgress,&progress,1)||!b.upload(b.localKnots,knots)
        ||!b.allocate(b.localDriveControls,1)||!b.allocate(b.localDrive,1)||!b.allocate(b.localDriveCandidates,resident.frames)
        ||!b.upload(b.localAuxiliary,state.filmAux)||!b.allocate(b.localCandidateAuxiliary,nf)
        ||!b.allocate(b.localRecession,nf)||!b.allocate(b.localRecessionStatus,nf)
        ||!b.upload(b.localGasFaces,state.surface.gasFace)||!b.upload(b.localSolidFaces,state.surface.solidFace)
        ||!b.upload(b.radiationCells,state.surface.solidCell)||!b.allocate(b.radiation,nf)
        ||!b.allocate(b.radiationReceiver,nf))return b.failure(error);
    MaterialLocalBindings binding{};
    binding.material=b.localState.data();binding.step=b.localStep.data();binding.total=b.localTotal.data();
    binding.clock=b.localClock.data();binding.progress=b.localProgress.data();binding.cursor=b.sliceCursor.data();
    binding.driveControls=b.localDriveControls.data();binding.drive=b.localDrive.data();binding.driveCandidates=b.localDriveCandidates.data();
    binding.slice={resident,b.slicePlan.data(),b.currentSlice(),true};binding.slicePlan=b.slicePlan.data();
    binding.used=b.currentSlice();binding.candidateUsed=b.candidateSlice();
    binding.knots=b.localKnots.data();binding.knotCount=knots.size();binding.physics=b.devicePhysics.data();
    binding.diagnostics=b.diagnostics.device();binding.controls=controls;
    binding.transport=b.workspace();binding.transportResult=b.transportResult.data();
    binding.sourceStatus=b.sourceStatus.data();binding.sourceResult=b.stepResult.data();
    binding.radiation=b.radiation.data();binding.externalRadiation=b.sliceExternal.data();binding.externalFaces=b.sliceExternal.size();
    binding.receivers=b.radiationReceiver.data();binding.gasFaces=b.localGasFaces.data();binding.solidFaces=b.localSolidFaces.data();
    binding.auxiliary=b.localAuxiliary.data();binding.candidateAuxiliary=b.localCandidateAuxiliary.data();
    binding.recessionTarget=b.localRecession.data();binding.recessionStatus=b.localRecessionStatus.data();
    binding.cells=b.cells;binding.faces=b.faces;
    binding.audit=b.localAudit.data();binding.auditCount=int(observations.size());
    binding.auditState=b.localAuditState.data();binding.auditRates=b.localAuditRates.data();
    if(!streamed){
        material_slice_detail::plan<<<1,1,0,b.stream.get()>>>(resident,b.sliceCursor.data(),b.localTotal.data(),b.sliceExternal.size(),b.slicePlan.data());
        if(!b.fault.check(cudaGetLastError(),"plan full material packet rows")
            ||!b.buildRows(int(resident.frames*nf*3),b.cells,ResidentMaterialSliceRow{binding.slice},
                b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
    }
    if(!b.buildRows(int(nf),b.cells,MaterialRadiationCellRow{b.radiationCells.data()},
            b.radiationRows,b.radiationOffsets,b.radiationFaces))return b.failure(error);
    binding.packetOffsets=b.packetOffsets.data();binding.packetIndices=b.packetIndices.data();
    binding.radiationOffsets=b.radiationOffsets.data();binding.radiationFaces=b.radiationFaces.data();
    b.localBindings=binding;b.localWindowEnd=history.interval().end;
    b.localPrepared=true;error.clear();return true;
}
bool preparePendingResidentMaterialLocalWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,
    const ResidentMaterialDriveHistory& resident,const IntervalRadiationDeviceView& radiation,const WallProgram& program,
    const CpuMaterialControls& controls,std::string& error){
    if(!resident.deviceFrames||!resident.frames||!resident.surface){error="pending material history count or storage missing";return false;}
    IntervalHistory descriptor;if(!descriptor.begin(program.interval,error))return false;
    descriptor.setExternalRadiationDevice(radiation);
    if(!prepareMaterialLocalWindow(b,state,physics,descriptor,resident,program,controls,error,false,true))return false;
    b.pendingPreparation=std::make_shared<MaterialBackend::PendingPreparation>(MaterialBackend::PendingPreparation{state,program,controls});
    b.localStreamed=false;b.localHistory.reset();return true;
}
bool prepareResidentMaterialLocalWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,
    const IntervalHistory& history,const ResidentMaterialDriveHistory& resident,const WallProgram& program,
    const CpuMaterialControls& controls,std::string& error){
    if(!prepareMaterialLocalWindow(b,state,physics,history,resident,program,controls,error,false))return false;
    b.localStreamed=false;b.localHistory.reset();return true;
}
bool prepareStreamedMaterialLocalWindow(MaterialBackend& b,const HostState& state,const PhysicsConfig& physics,
    const IntervalHistory& history,const std::shared_ptr<CoupledHistoryArchive>& archive,const WallProgram& program,
    const CpuMaterialControls& controls,std::string& error){
    if(b.localPending){error="material local window still owns its buffers";return false;}
    b.localPrepared=false;
    if(!archive||archive->frames()!=history.records().size()){error="material history payload or interval metadata missing";return false;}
    const auto faces=state.surface.area.size();
    if(!faces||faces>std::size_t(INT_MAX)/CoupledHistoryArchive::slabFrames/3){error="material history chunk extent is not addressable";return false;}
    std::vector<ResidentMaterialDriveInterval> metadata;metadata.reserve(history.records().size());
    for(const auto& record:history.records())metadata.push_back({residentWindowTime(record.localBegin),residentWindowTime(record.localEnd),record.actualDt});
    if(!b.upload(b.streamedIntervals,metadata))return b.failure(error);
    const ResidentMaterialDriveHistory complete{nullptr,nullptr,b.streamedIntervals.data(),metadata.size(),faces,b.stream.get()};
    if(!prepareMaterialLocalWindow(b,state,physics,history,complete,program,controls,error,true))return false;
    b.localPrepared=false;
    const bool matching=physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer;
    for(auto& slot:b.streamedChunks){
        if(!b.allocate(slot.surface,CoupledHistoryArchive::slabFrames*faces)
            ||!b.allocate(slot.wall,matching?CoupledHistoryArchive::slabFrames*faces:0)
            ||!b.allocate(slot.interval,CoupledHistoryArchive::slabFrames))return b.failure(error);
    }
    b.hostChunks[0]={};b.hostChunks[1]={};b.chunkUse[0]=b.chunkUse[1]=b.chunkSequence=0;
    const StreamedMaterialProgress progress{};
    if(!b.uploadRaw(b.streamedViews,b.hostChunks,2)||!b.uploadRaw(b.streamedProgress,&progress,1)
        ||!b.allocate(b.streamedHeldSurface,faces)||!b.allocate(b.streamedHeldWall,matching?faces:0)
        ||!b.allocate(b.streamedSourcePackets,b.cells)||!b.allocate(b.streamedTransported,b.cells))return b.failure(error);
    b.streamedBindings={b.localBindings,b.streamedProgress.data(),b.streamedViews.data(),
        b.streamedHeldSurface.data(),b.streamedHeldWall.data(),matching,b.streamedSourcePackets.data(),b.streamedTransported.data()};
    b.localStreamed=true;b.localHistory=archive;b.localPrepared=true;error.clear();return true;
}
namespace {
struct StreamedMaterialPacketRow {
    const CoupledSurfaceDownload* surface=nullptr;
    __host__ __device__ int operator()(int i)const{
        const auto& raw=surface[i/3];const unsigned kind=i%3;
        if(kind&&!hasPacketTransfer(materialHistoryPacket(raw,kind)))return -1;
        const auto packet=materialHistoryPacket(raw,kind);
        return (requiredConsumers(packet.kind)&ConsumeSolid)?packet.solidCell:-1;
    }
};
}
bool supplyResidentMaterialHistory(MaterialBackend& b,std::size_t frame,std::string& error){
    if(!b.localStreamed||!b.localHistory||frame>=b.localHistory->frames()){
        error="requested material history frame is unavailable";return false;
    }
    const auto chunk=b.localHistory->chunk(frame);const auto faces=b.localBindings.slice.history.faces;
    const bool matching=b.streamedBindings.matching;
    if(chunk->frames>CoupledHistoryArchive::slabFrames||chunk->surface.size()!=chunk->frames*faces
        ||chunk->interval.size()!=chunk->frames||(matching&&chunk->wall.size()!=chunk->frames*faces)){
        error="material history chunk is missing consumed fields";return false;
    }
    const int index=b.chunkUse[0]<=b.chunkUse[1]?0:1;auto& slot=b.streamedChunks[index];
    if(!b.transfers.upload(slot.surface.data(),chunk->surface.data(),chunk->surface.size())
        ||(matching&&!b.transfers.upload(slot.wall.data(),chunk->wall.data(),chunk->wall.size()))
        ||!b.transfers.upload(slot.interval.data(),chunk->interval.data(),chunk->interval.size())
        ||!b.buildRows(int(chunk->frames*faces*3),b.cells,StreamedMaterialPacketRow{slot.surface.data()},
            slot.rows,slot.offsets,slot.indices))return b.failure(error);
    b.hostChunks[index]={{slot.surface.data(),matching?slot.wall.data():nullptr,slot.interval.data(),chunk->frames,faces,b.stream.get()},
        chunk->first,slot.offsets.data(),slot.indices.data()};
    b.chunkUse[index]=++b.chunkSequence;
    if(!b.transfers.upload(b.streamedViews.data()+index,&b.hostChunks[index],1))return b.failure(error);
    error.clear();return true;
}
namespace {
__global__ void preparePendingHistory(MaterialLocalBindings b,int* prepared,cudaGraphConditionalHandle allowed){
    if(threadIdx.x||blockIdx.x)return;
    bool run=!*prepared&&!b.progress->failurePhase&&!ugkp::diagnostics::hasFatal(b.diagnostics);
    if(run){
        const auto h=b.slice.history.resolved();
        if(!h.frames||h.frames>b.slice.history.frames||!(h.interval[h.frames-1].end==*b.total)){
            material_local_detail::fail(b,2,int(ugkp::IntervalSliceError::ExceedsHistory),-1);run=false;
        }
    }
    cudaGraphSetConditional(allowed,run);if(run)*prepared=1;
}
__global__ void initializePendingAudit(MaterialLocalBindings b){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=b.auditCount)return;
    const auto h=b.slice.history.resolved();
    const auto samples=h.frames<std::size_t(b.auditCount)?h.frames:std::size_t(b.auditCount);
    if(std::size_t(i)>=samples){b.audit[i]={};b.audit[i].record=std::numeric_limits<std::size_t>::max();b.audit[i].time=*b.total;b.audit[i].ready=true;return;}
    const auto record=(h.frames/samples)*i+((h.frames%samples)*i)/samples;
    b.audit[i]={record,h.interval[record].begin};
}
}
static ugkp::gpu::ExecutionGraphNode appendPendingMaterialHistory(MaterialBackend& b,
    ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.localHistoryPending)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    const auto local=b.localBindings;const auto stream=b.stream.get();
    cudaGraphConditionalHandle permitted;checked(cudaGraphConditionalHandleCreate(&permitted,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        preparePendingHistory<<<1,1,0,stream>>>(local,b.localHistoryPrepared.data(),permitted);
    });
    cudaGraph_t body=nullptr;const auto done=addIf(graph,permitted,{selected},body);
    appendRecorded(body,{},stream,[&]{
        material_slice_detail::plan<<<1,1,0,stream>>>(local.slice.history,local.cursor,local.total,local.externalFaces,local.slicePlan);
        if(!b.buildRows(int(local.slice.history.frames*local.slice.history.faces*3),b.cells,ResidentMaterialSliceRow{local.slice},
            b.packetRows,b.packetOffsets,b.packetIndices))throw std::runtime_error(b.fault.message);
        if(b.localExternalSource&&local.externalFaces)checked(cudaMemcpyAsync(b.sliceExternal.data(),b.localExternalSource,
            local.externalFaces*sizeof(*local.externalRadiation),cudaMemcpyDeviceToDevice,stream));
        const int threads=workPlan().materialBlockThreads;
        initializePendingAudit<<<std::max(1,(local.auditCount+threads-1)/threads),threads,0,stream>>>(local);
    });
    return done;
}
namespace {
__global__ void localConductionChoose(MaterialLocalBindings local,const int* stage,int expected,
    cudaGraphConditionalHandle permitted){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(permitted,*stage==expected
        &&!local.progress->failurePhase&&!ugkp::diagnostics::hasFatal(local.diagnostics));
}
__global__ void localSourcesDone(MaterialLocalBindings local,int* stage){
    if(!threadIdx.x&&!blockIdx.x&&local.progress->complete&&!local.progress->failurePhase){
        *stage=1;
        ugkp::diagnostics::beginOperation(*local.diagnostics,local.clock->sequence,local.diagnostics->step,
            ugkp::absoluteTime(local.clock->origin,*local.total));
    }
}
__global__ void localConductionStatus(MaterialLocalBindings local,int* stage,int next){
    if(threadIdx.x||blockIdx.x)return;
    const auto result=*local.sourceResult;
    if(result.phase){material_local_detail::fail(local,10+result.phase,result.code,result.entity);return;}
    *stage=next;
}
__global__ void localConductionSolved(MaterialLocalBindings local,int* stage,
    const MaterialConductionIteration* report,const int* capacity){
    if(threadIdx.x||blockIdx.x||(capacity&&*capacity))return;
    if(!report->valid){material_local_detail::fail(local,14,report->failure,report->audit.invalidCell);return;}
    *stage=3;
}
__global__ void localConductionCommit(MaterialLocalBindings local,int* stage){
    if(threadIdx.x||blockIdx.x)return;
    const auto result=*local.sourceResult;
    if(result.phase){material_local_detail::fail(local,15,result.code,result.entity);return;}
    if(ugkp::diagnostics::hasFatal(local.diagnostics))return;
    auto& state=*local.material;auto* old=state.current;state.current=state.candidate;state.candidate=old;
    *stage=4;
}
}
bool prepareMaterialLocalConduction(MaterialBackend& b,ugkp::gpu::ImplicitDiffusion& diffusion,
    ugkp::gpu::SparseConjugateGradient& sparse,const std::vector<int>& external,Real dt,
    const CpuMaterialControls& controls,Real normalizationSmall,Real singularSmall,std::string& error){
    if(!b.localPrepared||b.localPending){error="prepare material sources before binding conduction";return false;}
    if(!diffusion.residentIteration()||!finite(dt)||!(dt>0)){error="material conduction equation or interval missing";return false;}
    if(!b.waitConduction(diffusion.deviceValues().readyStream))return b.failure(error);
    if(b.cachedConductionExternal!=external){if(!b.upload(b.conductionExternal,external))return b.failure(error);b.cachedConductionExternal=external;}
    const int threads=ugkp::gpu::workPlan().materialBlockThreads;
    const int blocks=(std::max(b.cells,int(external.size()))+threads-1)/threads;
    const int initial=0;
    if(!b.allocate(b.conductionAuditParts,blocks)||!b.allocate(b.conductionAuditResult,1)
        ||!b.allocate(b.conductionReport,1)||!b.uploadRaw(b.localStage,&initial,1))return b.failure(error);
    b.localDiffusion=&diffusion;b.localSparse=&sparse;b.localExternalFaces=external.size();
    b.localConductionDt=dt;b.localNormalizationSmall=normalizationSmall;b.localSingularSmall=singularSmall;
    b.localConduction=true;error.clear();return true;
}
bool prepareResidentMaterialEndpointSurface(MaterialBackend& b,const HostState& geometry,std::string& error){
    b.localEndpoint=false;

    if(!b.localPrepared||(!b.localConduction&&b.cells))return true;
    const std::size_t faces=geometry.surface.area.size();
    if(!b.surfacePrototypeValid||b.surfacePrototype.count!=int(faces)||b.surfaceInput.size()!=faces){
        error="material endpoint surface inputs were not prepared by the window predictor";return false;
    }
    std::vector<GpuReal> shares;std::vector<Real> volumes(faces);std::vector<int> cells(faces,-1);
    if(!surfaceGasDonorShares(geometry,shares,error))return false;
    for(std::size_t f=0;f<faces;++f){const int face=geometry.surface.gasFace[f];if(face<0)continue;
        if(std::size_t(face)>=geometry.gasMesh.owner.size()){error="material endpoint gas face is unavailable";return false;}
        const int cell=geometry.gasMesh.owner[face];
        if(cell<0||std::size_t(cell)>=geometry.gasMesh.volumes.size()
            ||(!geometry.gasVoidFraction.empty()&&std::size_t(cell)>=geometry.gasVoidFraction.size())){
            error="material endpoint gas phase volume is unavailable";return false;
        }
        cells[f]=cell;volumes[f]=geometry.gasMesh.volumes[cell];
    }
    const int zero=0;
    if(!b.endpointInput.copy(b.surfaceInput,b.fault,b.stream.get())||!b.allocate(b.endpointPhysics,faces)
        ||!b.allocate(b.endpointWall,faces)||!b.allocate(b.endpointStatus,faces)||!b.allocate(b.endpointOperation,1)
        ||!b.uploadRaw(b.endpointComplete,&zero,1)||!b.upload(b.endpointGasCells,cells)
        ||!b.upload(b.endpointGasVolume,volumes)||!b.upload(b.endpointDonorShare,shares)
        ||!b.upload(b.endpointHostFraction,geometry.gasVoidFraction))return b.failure(error);
    auto history=b.localBindings.slice.history;
    if(b.localStreamed){
        const auto chunk=b.localHistory->chunk(b.localHistory->frames()-1);
        const auto offset=(b.localHistory->frames()-1-chunk->first)*faces;
        if(!b.allocate(b.endpointHistorySurface,faces)
            ||!b.transfers.upload(b.endpointHistorySurface.data(),chunk->surface.data()+offset,faces))return b.failure(error);
        if(!chunk->wall.empty()&&(!b.allocate(b.endpointHistoryWall,faces)
            ||!b.transfers.upload(b.endpointHistoryWall.data(),chunk->wall.data()+offset,faces)))return b.failure(error);
        history={b.endpointHistorySurface.data(),chunk->wall.empty()?nullptr:b.endpointHistoryWall.data(),nullptr,1,faces,b.stream.get()};
    }
    auto view=b.surfacePrototype;view.inputs=b.endpointInput.data();view.physicsOut=b.endpointPhysics.data();
    view.wallOut=b.endpointWall.data();view.status=b.endpointStatus.data();
    using DynamicSurface=ugkp::gpu::DeviceValue<MaterialSurfaceView>;
    if(!b.surfaceCooperative&&faces){
        if(!b.fault.check(ugkp::gpu::scalarWorkLaunch(evaluateMaterialSurfaceKernel<DynamicSurface>,view.workers,b.endpointLaunch),
            "schedule material endpoint workers"))return b.failure(error);
        view.scalarWorkersPerBlock=b.endpointLaunch.workersPerBlock;
    }
    b.endpointBindings={b.localBindings,history,view,b.endpointOperation.data(),b.endpointInput.data(),
        b.endpointGasCells.data(),b.endpointGasVolume.data(),b.endpointDonorShare.data(),
        geometry.gasVoidFraction.empty()?nullptr:b.endpointHostFraction.data(),
        b.localConduction?b.localStage.data():nullptr,b.endpointComplete.data(),b.localWindowEnd-b.time,
        b.physics.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer};
    b.localEndpoint=true;error.clear();return true;
}
bool bindResidentMaterialEndpointGasFraction(MaterialBackend& b,const GpuReal* fraction,void* producer,std::string& error){
    if(!b.localEndpoint)return true;
    if(b.localPending){error="material endpoint is already submitted";return false;}
    if(!b.waitConduction(producer))return b.failure(error);
    b.endpointBindings.gasFraction=fraction;error.clear();return true;
}
static ugkp::gpu::ExecutionGraphNode appendMaterialEndpointSurface(MaterialBackend& b,
    ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.localEndpoint)return dependency;
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    const auto binding=b.endpointBindings;const auto stream=b.stream.get();
    const int threads=workPlan().materialBlockThreads,blocks=std::max(1,(binding.prototype.count+threads-1)/threads);
    cudaGraphConditionalHandle condition,valid;
    checked(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
    const auto selected=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        material_endpoint_detail::select<<<1,1,0,stream>>>(binding,condition,false);
    });
    cudaGraph_t body=nullptr;const auto completed=addIf(graph,condition,{selected},body);
    checked(cudaGraphConditionalHandleCreate(&valid,body,0,cudaGraphCondAssignDefault));
    const auto ready=appendRecorded(body,{},stream,[&]{
        material_endpoint_detail::prepare<<<blocks,threads,0,stream>>>(binding);
        material_endpoint_detail::select<<<1,1,0,stream>>>(binding,valid,true);
    });
    cudaGraph_t closure=nullptr;addIf(body,valid,{ready},closure);
    appendRecorded(closure,{},stream,[&]{
        using DynamicSurface=DeviceValue<MaterialSurfaceView>;
        const DynamicSurface input{binding.operation};const auto& view=binding.prototype;
        if(view.count&&b.surfaceCooperative){
            if(b.surfaceDepth==3){using Team=MaterialSurfaceRootTeam<3>;
                evaluateMaterialSurfaceTeamKernel<DynamicSurface,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),stream>>>(input,b.devicePhysics.data());
            }else if(b.surfaceDepth==2){using Team=MaterialSurfaceRootTeam<2>;
                evaluateMaterialSurfaceTeamKernel<DynamicSurface,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),stream>>>(input,b.devicePhysics.data());
            }else evaluateMaterialSurfaceTeamKernel<DynamicSurface><<<view.workers,ugkwp::gaswall::detail::wallWarpWidth,0,stream>>>(input,b.devicePhysics.data());
        }else if(view.count)evaluateMaterialSurfaceKernel<DynamicSurface><<<b.endpointLaunch.blocks,b.endpointLaunch.threads,0,stream>>>(input,b.devicePhysics.data());
        material_endpoint_detail::inspect<<<blocks,threads,0,stream>>>(binding);
        material_endpoint_detail::finish<<<1,1,0,stream>>>(binding);
    });
    return completed;
}
bool collectResidentMaterialEndpointSurface(MaterialBackend& b,CpuSurfaceResult& output,bool& published,std::string& error){
    published=false;if(!b.localEndpoint)return true;
    if(b.localPending){error="collect material window before publishing its endpoint surface";return false;}
    int done=0;
    if(!b.transfers.download(&done,b.endpointComplete.data(),1)||!b.read(b.endpointPhysics,output.physics)
        ||!b.read(b.endpointWall,output.wall)||!b.finish(error))return b.failure(error);
    if(!done){error="material endpoint surface did not finish";return false;}
    published=true;error.clear();return true;
}

namespace {
__global__ void commitMaterialWindow(MaterialLocalBindings local,const int* conduction,const int* endpoint,
    MaterialWindowCommit* commit){
    if(threadIdx.x||blockIdx.x||commit->committed||ugkp::diagnostics::hasFatal(local.diagnostics)
        ||!local.progress->complete||local.progress->failurePhase
        ||(conduction&&*conduction!=4)||(endpoint&&!*endpoint))return;
    commit->time=ugkp::absoluteTime(local.clock->origin,*local.total);
    commit->sequence=local.clock->sequence;commit->acceptedStep=std::uint64_t(local.diagnostics->step)+1;
    commit->committed=true;
}
}
static ugkp::gpu::ExecutionGraphNode appendMaterialWindowCommit(MaterialBackend& b,
    ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency){


    if((b.cells&&!b.localConduction)||!b.localEndpoint)return dependency;
    return ugkp::gpu::executionGraphDetail::appendRecorded(graph,{dependency},b.stream.get(),[&]{
        commitMaterialWindow<<<1,1,0,b.stream.get()>>>(b.localBindings,b.localConduction?b.localStage.data():nullptr,
            b.endpointComplete.data(),b.localCommit.data());
    });
}
ugkp::gpu::ExecutionGraphNode appendResidentMaterialLocalWindow(MaterialBackend& b,
    ugkp::gpu::ExecutionGraph graph,ugkp::gpu::ExecutionGraphNode dependency){
    if(!b.localPrepared)throw std::logic_error("prepare material local buffers before graph construction");
    dependency=appendPendingMaterialHistory(b,graph,dependency);
    auto appendLocal=[&](ugkp::gpu::ExecutionGraph parent,ugkp::gpu::ExecutionGraphNode dep){
        return b.localStreamed?appendStreamedMaterialLoop(parent,dep,b.streamedBindings,b.stream.get())
            :appendResidentMaterialLocalLoop(parent,dep,b.localBindings,b.stream.get());
    };
    if(!b.localConduction)return appendMaterialWindowCommit(b,graph,appendMaterialEndpointSurface(b,graph,appendLocal(graph,dependency)));
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    auto& diffusion=*b.localDiffusion;auto& sparse=*b.localSparse;
    const auto local=b.localBindings;const auto stream=b.stream.get();
    const int threads=workPlan().materialBlockThreads,blocks=std::max(1,(b.cells+threads-1)/threads);
    cudaGraphNode_t previous=dependency;
    for(int phase=0;phase<4;++phase){
        cudaGraphConditionalHandle permitted;checked(cudaGraphConditionalHandleCreate(&permitted,graph,0,cudaGraphCondAssignDefault));
        const auto selected=appendRecorded(graph,previous?std::vector<cudaGraphNode_t>{previous}:std::vector<cudaGraphNode_t>{},stream,[&]{
            localConductionChoose<<<1,1,0,stream>>>(local,b.localStage.data(),phase,permitted);
        });
        cudaGraph_t body=nullptr;previous=addIf(graph,permitted,{selected},body);
        if(phase==0){
            const auto done=appendLocal(body,nullptr);
            appendRecorded(body,{done},stream,[&]{localSourcesDone<<<1,1,0,stream>>>(local,b.localStage.data());});
        }else if(phase==1){
            const auto prepared=appendRecorded(body,{},stream,[&]{
                prepareConductionKernel<<<blocks,threads,0,stream>>>(b.cells,b.initial.data(),b.initialVolumes.data(),
                    DeviceValue<SolidQ*>{&local.material->current},DeviceValue<const Real*>{&local.material->mesh.volumes},
                    b.conductionVolumes.data(),b.devicePhysics.data(),b.conductionInput.data(),b.sourceStatus.data());
                selectConductionFailure<<<1,threads,threads*sizeof(MaterialStepResult),stream>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
                localConductionStatus<<<1,1,0,stream>>>(local,b.localStage.data(),1);
            });
            cudaGraphConditionalHandle valid;checked(cudaGraphConditionalHandleCreate(&valid,graph,0,cudaGraphCondAssignDefault));
            const auto ready=appendRecorded(body,{prepared},stream,[&]{localConductionChoose<<<1,1,0,stream>>>(local,b.localStage.data(),1,valid);});
            cudaGraph_t initialized=nullptr;addIf(body,valid,{ready},initialized);
            appendRecorded(initialized,{},stream,[&]{
                conductionAnchorTemperature<<<blocks,threads,0,stream>>>(b.cells,b.conductionInput.data(),diffusion.deviceSolution());
                diffusion.enqueueResidentBoundaryTemperature(stream);
                conductionCoefficientsKernel<<<blocks,threads,0,stream>>>(b.cells,b.conductionInput.data(),diffusion.deviceTemperature(),
                    b.localConductionDt,b.devicePhysics.data(),b.conductionCoefficients.data(),b.conductionCapacity.data(),
                    b.conductionSource.data(),b.conductionConductivity.data());
                inspectConductionStartupKernel<<<blocks,threads,0,stream>>>(b.cells,b.conductionCoefficients.data(),b.devicePhysics.data(),b.sourceStatus.data());
                selectConductionFailure<<<1,threads,threads*sizeof(MaterialStepResult),stream>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
                checked(cudaMemsetAsync(&b.conductionReport.data()->continuationStage,0,sizeof(int),stream));
                localConductionStatus<<<1,1,0,stream>>>(local,b.localStage.data(),2);
            });
        }else if(phase==2){
            const auto done=appendMaterialConductionLoop(b,diffusion,sparse,body,nullptr,b.localExternalFaces,b.localConductionDt,
                local.controls,b.localNormalizationSmall,b.localSingularSmall);
            appendRecorded(body,{done},stream,[&]{localConductionSolved<<<1,1,0,stream>>>(local,b.localStage.data(),b.conductionReport.data(),sparse.capacityPendingDevice());});
        }else{
            appendRecorded(body,{},stream,[&]{
                finishConductionKernel<<<blocks,threads,0,stream>>>(b.cells,DeviceValue<SolidQ*>{&local.material->current},
                    DeviceValue<const Real*>{&local.material->mesh.volumes},b.conductionInput.data(),diffusion.deviceTemperature(),
                    b.devicePhysics.data(),DeviceValue<SolidQ*>{&local.material->candidate},b.sourceStatus.data());
                selectConductionFailure<<<1,threads,threads*sizeof(MaterialStepResult),stream>>>(b.sourceStatus.data(),b.cells,b.stepResult.data());
                localConductionCommit<<<1,1,0,stream>>>(local,b.localStage.data());
            });
        }
    }
    return appendMaterialWindowCommit(b,graph,appendMaterialEndpointSurface(b,graph,previous));
}
static bool launchMaterialLocalGraph(MaterialBackend& b,std::string& error){
    try{
        using ugkp::gpu::executionGraphDetail::checked;
        ugkp::gpu::ExecutionGraphSignature signature;signature.add(b.localBindings);signature.add(b.localStreamed);
        if(b.localStreamed)signature.add(b.streamedBindings);
        signature.add(b.localHistoryPending);signature.add(b.localHistoryPrepared.data());signature.add(b.localExternalSource);
        signature.add(b.localConduction);signature.add(b.localEndpoint);signature.add(b.localCommit.data());
        if(b.localEndpoint){signature.add(b.endpointBindings);signature.add(b.surfaceCooperative);
            signature.add(b.surfaceDepth);signature.add(b.endpointLaunch);}
        if(b.localConduction){
            signature.append(b.localDiffusion->residentExecutionSignature());
            signature.append(b.localSparse->residentExecutionSignature());
            signature.add(b.localStage.data());signature.add(b.localConductionDt);signature.add(b.localExternalFaces);
            signature.add(b.localNormalizationSmall);signature.add(b.localSingularSmall);
            for(const void* pointer:{static_cast<const void*>(b.conductionReport.data()),static_cast<const void*>(b.conductionInput.data()),
                static_cast<const void*>(b.conductionCoefficients.data()),static_cast<const void*>(b.conductionVolumes.data()),
                static_cast<const void*>(b.conductionCapacity.data()),static_cast<const void*>(b.conductionSource.data()),
                static_cast<const void*>(b.conductionConductivity.data()),static_cast<const void*>(b.conductionAuditParts.data()),
                static_cast<const void*>(b.conductionAuditResult.data()),static_cast<const void*>(b.conductionExternal.data()),
                static_cast<const void*>(b.initial.data()),static_cast<const void*>(b.initialVolumes.data())})signature.add(pointer);
        }
        if(!b.localGraph.executable||!(b.localGraph.signature==signature)){
            b.localGraph.clear();checked(cudaGraphCreate(&b.localGraph.graph,0));
            appendResidentMaterialLocalWindow(b,b.localGraph.graph,nullptr);
            checked(cudaGraphInstantiateWithFlags(&b.localGraph.executable,b.localGraph.graph,0));b.localGraph.signature=signature;
        }
        checked(cudaGraphLaunch(b.localGraph.executable,b.stream.get()));b.localPending=true;error.clear();return true;
    }catch(const std::exception& e){error=e.what();return false;}
}
bool enqueueResidentMaterialLocalWindow(MaterialBackend& b,std::string& error){
    if(!b.localPrepared||b.localPending){error="material local window is unprepared or already submitted";return false;}
    return launchMaterialLocalGraph(b,error);
}
void acceptResidentMaterialLocalCompletion(MaterialBackend& b,const std::shared_future<void>& result){
    b.localCompletion=result;b.localPending=true;
}
void acceptResidentMaterialLocalExecution(MaterialBackend& b,void* producer){
    using ugkp::gpu::executionGraphDetail::checked;
    if(!b.localReady)checked(cudaEventCreateWithFlags(&b.localReady,cudaEventDisableTiming));
    checked(cudaEventRecord(b.localReady,static_cast<cudaStream_t>(producer)));
    checked(cudaStreamWaitEvent(b.stream.get(),b.localReady,0));
    b.localPending=true;
}
MaterialLocalDeviceView residentMaterialLocalDeviceView(const MaterialBackend& b){
    return {b.localState.data(),b.localProgress.data(),b.stream.get(),b.localStreamed?b.streamedProgress.data():nullptr,
        b.localConduction?b.localStage.data():nullptr,b.diagnostics.device(),b.localCommit.data()};
}
bool serviceResidentMaterialLocalCapacity(MaterialBackend& b,std::string& error){
    StreamedMaterialProgress storage;MaterialLocalProgress local;int stage=0;
    if((b.localStreamed&&!b.transfers.download(&storage,b.streamedProgress.data(),1))
        ||(b.localConduction&&!b.transfers.download(&stage,b.localStage.data(),1))
        ||!b.transfers.download(&local,b.localProgress.data(),1)||!b.finish(error))return false;
    if(local.failurePhase){error="material parent has a failed operation";return false;}
    if(b.localConduction&&stage==2&&b.localSparse->servicePendingCapacity()){error.clear();return true;}
    if(b.localStreamed&&stage==0&&!local.complete&&storage.requestFrame!=std::numeric_limits<std::size_t>::max())
        return supplyResidentMaterialHistory(b,storage.requestFrame,error);
    error="material parent stopped without completion or a storage request";return false;
}
bool releaseUnusedMaterialLocalWindow(MaterialBackend& b,std::string& error){
    MaterialLocalProgress result;ugkp::IntervalSliceCursor cursor;
    if(!b.transfers.download(&result,b.localProgress.data(),1)||!b.transfers.download(&cursor,b.sliceCursor.data(),1)
        ||!b.finish(error))return false;
    if(result.acceptedSteps||result.failurePhase||!(cursor.time==ugkp::CompensatedTime{})){
        error="cannot replace a material window after a physical candidate has begun";return false;
    }
    b.localPending=false;b.localPrepared=false;b.localCompletion={};error.clear();return true;
}
bool completePendingMaterialHistory(MaterialBackend& b,const IntervalHistory& history,
    const std::shared_ptr<CoupledHistoryArchive>& archive,std::string& error){
    const auto pending=b.pendingPreparation;
    if(!pending||!b.localHistoryPending||!history.complete()){
        error="pending material reader is missing its completed gas history";return false;
    }
    const auto physics=b.physics;const bool conduction=b.localConduction;
    auto* diffusion=b.localDiffusion;auto* sparse=b.localSparse;
    const auto external=b.cachedConductionExternal;
    const auto dt=b.localConductionDt,normalization=b.localNormalizationSmall,singular=b.localSingularSmall;
    const auto* fraction=b.endpointBindings.gasFraction;


    if(!releaseUnusedMaterialLocalWindow(b,error)
        ||!prepareStreamedMaterialLocalWindow(b,pending->state,physics,history,archive,pending->program,pending->controls,error))return false;
    if(conduction&&!prepareMaterialLocalConduction(b,*diffusion,*sparse,external,dt,pending->controls,normalization,singular,error))return false;
    if(!prepareResidentMaterialEndpointSurface(b,pending->state,error))return false;
    if(b.localEndpoint)b.endpointBindings.gasFraction=fraction;
    b.pendingPreparation.reset();error.clear();return true;
}
bool collectResidentMaterialLocalWindow(MaterialBackend& b,MaterialLocalReport& report,std::vector<FilmAux>& auxiliary,std::string& error){
    if(!b.localPending){error="no material local window has been submitted";return false;}
    if(b.localCompletion.valid())b.localCompletion.get();
    if(b.localStreamed||b.localConduction)for(;;){
        StreamedMaterialProgress progress;MaterialLocalProgress local;int stage=0;
        if((b.localStreamed&&!b.transfers.download(&progress,b.streamedProgress.data(),1))
            ||(b.localConduction&&!b.transfers.download(&stage,b.localStage.data(),1))
            ||!b.transfers.download(&local,b.localProgress.data(),1)||!b.finish(error))return false;
        if(local.failurePhase){error="material window device operation failed";return false;}
        if(local.complete&&(!b.localConduction||stage==4))break;
        if(b.localConduction&&stage==2&&b.localSparse->servicePendingCapacity()){
            if(!launchMaterialLocalGraph(b,error))return false;continue;
        }
        if(b.localStreamed&&stage==0&&progress.requestFrame!=std::numeric_limits<std::size_t>::max()){
            if(!supplyResidentMaterialHistory(b,progress.requestFrame,error)||!launchMaterialLocalGraph(b,error))return false;
            continue;
        }
        error="material window stopped without completion or a storage request";return false;
    }
    MaterialLocalProgress result;ResidentMaterialState state;ugkp::IntervalSliceCursor cursor;MaterialWindowCommit commit;
    if(!b.transfers.download(&result,b.localProgress.data(),1)||!b.transfers.download(&state,b.localState.data(),1)
        ||!b.transfers.download(&cursor,b.sliceCursor.data(),1)||!b.transfers.download(&commit,b.localCommit.data(),1)||!b.read(b.localAuxiliary,auxiliary)
        ||!b.transfers.finish())return b.failure(error);
    b.localPending=false;b.localPrepared=false;b.pendingPreparation.reset();
    if(state.current!=b.current.data())b.current.swap(b.next);
    if(state.gasChange!=b.gasChange.data())b.gasChange.swap(b.candidateGasChange);
    if(state.mesh.volumes!=b.volumes.data())b.volumes.swap(b.nextVolumes);
    report.acceptedSteps=result.acceptedSteps;report.complete=result.complete;report.commit=commit;
    report.failurePhase=result.failurePhase;report.failureCode=result.failureCode;report.entity=result.entity;
    report.time=result.complete?b.localWindowEnd:ugkp::absoluteTime(b.time,cursor.time);
    b.time=report.time;b.publishedCurrentValid=false;b.sliceReady=false;b.slicePending=false;
    if(!b.finish(error))return false;
    if(!result.complete||result.failurePhase){error="material local window did not complete its physical interval";return false;}
    error.clear();return true;
}

bool collectResidentMaterialInventory(MaterialBackend& b,std::vector<SolidQ>& output,std::string& error){
    if(b.localPending){error="collect material completion before publishing inventory";return false;}
    if(!b.read(b.current,output)||!b.finish(error))return b.failure(error);
    b.publishedCurrent=output;b.publishedCurrentValid=true;error.clear();return true;
}
bool collectResidentMaterialLocalAudit(MaterialBackend& b,std::vector<MaterialLocalAudit>& observations,std::string& error){
    if(b.localPending){error="collect material completion before its diagnostic observations";return false;}
    if(!b.read(b.localAudit,observations)||!b.transfers.finish())return b.failure(error);
    if(b.localHistoryPending)observations.erase(std::remove_if(observations.begin(),observations.end(),
        [](const MaterialLocalAudit& sample){return sample.record==std::numeric_limits<std::size_t>::max();}),observations.end());
    for(const auto& sample:observations)if(!sample.ready){error="material diagnostic observation was not reached";return false;}
    error.clear();return true;
}

bool advanceResidentMaterialSources(MaterialBackend& b,const HostMesh& endpointMesh,
    const SurfaceMesh& surface,const std::vector<ExchangePacket>& packets,
    const std::vector<EnergyReal>& radiation,const std::vector<unsigned>& receivers,
    Real dt,const CpuMaterialControls& controls,bool finalStep,std::vector<SolidQ>& shadow,
    std::vector<SolidQ>& cumulative,std::string& error,bool& numerical,const std::vector<int>* observedCells,const IntervalSliceMetadata* selectedSlice){
    numerical=true;
    const int count=int(b.current.size());
    if(controls.retainDonorHistory&&!b.allocate(b.donor,count))return b.failure(error);


    for(const auto& packet:packets)if(requiredConsumers(packet.kind)&ConsumeSolid){
        if(packet.solidCell<0||packet.solidCell>=count){error="material packet owner invalid";return false;}
    }
    for(std::size_t f=0;f<radiation.size();++f){
        if(receivers[f]!=ConsumeSolid)continue;
        const int cell=surface.solidCell[f];
        if(cell<0||cell>=count){error="radiation has no material owner";return false;}
    }
    if(b.radiationCellMap!=surface.solidCell){
        if(!b.upload(b.radiationCells,surface.solidCell))return b.failure(error);
        b.radiationCellMap=surface.solidCell;
    }
    b.slicePending=b.sliceReady&&selectedSlice;
    constexpr std::size_t packetBatchSize=65536;
    const bool streamPackets=!b.slicePending&&packets.size()>packetBatchSize;
    if(b.slicePending){


        b.pendingSlicePlan={};b.pendingSlicePlan.begin.record=selectedSlice->beginRecord;
        b.pendingSlicePlan.past=selectedSlice->beginRecord+selectedSlice->parts.size();
        b.pendingSlicePlan.end.record=selectedSlice->endRecord;
        b.sliceExternalFraction=selectedSlice->externalFraction;b.sliceExternalLast=selectedSlice->externalLast;
    }
    if(!b.upload(b.nextVolumes,endpointMesh.volumes))return b.failure(error);
    if(b.slicePending){
        if(!b.allocate(b.radiation,radiation.size())||!b.allocate(b.radiationReceiver,receivers.size())
            ||!b.uploadRaw(b.slicePlan,&b.pendingSlicePlan,1)||!b.upload(b.sliceParts,selectedSlice->parts))return b.failure(error);
        if(!b.fault.check(prepareHostSelectedMaterialSlice(b.sourceSlice(),b.candidateSlice(),
            b.sliceExternal.data(),b.sliceExternal.size(),b.radiation.data(),b.stream.get()),
            "prepare borrowed material source slice"))return b.failure(error);
        gatherMaterialSliceReceivers<<<material_slice_detail::blocks(b.borrowedHistory.faces),ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(
            b.borrowedHistory,b.pendingSlicePlan.begin.record,b.radiationReceiver.data());
        if(!b.fault.check(cudaGetLastError(),"borrow material radiation receiver map"))return b.failure(error);
        const auto slots=(b.pendingSlicePlan.past-b.pendingSlicePlan.begin.record)*b.borrowedHistory.faces*3;
        if(!b.buildRows(int(slots),count,ResidentMaterialSliceRow{b.sourceSlice()},b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
    }else{
        if(!b.upload(b.radiation,radiation)||!b.upload(b.radiationReceiver,receivers))return b.failure(error);
        if(streamPackets){
            if(!b.allocate(b.streamedPacketRows,count))return b.failure(error);
            b.streamedPacketTransfers.bind(b.stream.get());
        }else if(!b.upload(b.packets,packets)
            ||!b.buildRows(int(packets.size()),count,PacketSolidRow{b.packets.data()},b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
    }
    b.externalSlicePending=b.deviceRadiation&&!b.slicePending;
    if(b.externalSlicePending){
        if(!selectedSlice||radiation.size()!=b.sliceExternal.size()){
            error="device radiation material slice is incomplete";return false;
        }
        material_slice_detail::prepareExternal<<<material_slice_detail::blocks(b.sliceExternal.size()),
            ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(b.sliceExternal.size(),b.sliceExternal.data(),
            b.sliceUsedExternal.data(),b.sliceCandidateExternal.data(),selectedSlice->externalFraction,
            selectedSlice->externalLast,b.radiation.data());
        if(!b.fault.check(cudaGetLastError(),"slice device DOM boundary energy"))return b.failure(error);
    }
    if(!b.buildRows(int(radiation.size()),count,RadiationSolidRow{b.radiationReceiver.data(),b.radiationCells.data()},
        b.radiationRows,b.radiationOffsets,b.radiationFaces))return b.failure(error);
    if(!b.fault.check(cudaMemsetAsync(b.stepResult.data(),0,sizeof(MaterialStepResult),b.stream.get()),
        "clear material source result"))return b.failure(error);
    if(count){
        if(streamPackets){
            const int width=ugkp::gpu::workPlan().materialBlockThreads,blocks=(count+width-1)/width;
            beginStreamedMaterialPackets<<<blocks,width,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
                b.gasChange.data(),dt,b.next.data(),b.candidateGasChange.data(),b.streamedPacketRows.data(),b.sourceStatus.data());
            if(!b.fault.check(cudaGetLastError(),"initialize bounded material packet accumulator"))return b.failure(error);
            for(std::size_t begin=0;begin<packets.size();begin+=packetBatchSize){
                const auto size=std::min(packetBatchSize,packets.size()-begin);
                if(!b.allocate(b.packets,size))return b.failure(error);
                {ugkp::gpu::PinnedTransferQueue::Operation transfer(b.streamedPacketTransfers);
                    b.streamedPacketTransfers.upload(b.packets.data(),packets.data()+begin,size*sizeof(ExchangePacket));}
                if(!b.buildRows(int(size),count,PacketSolidRow{b.packets.data()},b.packetRows,b.packetOffsets,b.packetIndices))return b.failure(error);
                appendStreamedMaterialPackets<<<blocks,width,0,b.stream.get()>>>(count,b.packets.data(),
                    b.packetOffsets.data(),b.packetIndices.data(),b.streamedPacketRows.data(),b.candidateGasChange.data());
                if(!b.fault.check(cudaGetLastError(),"append bounded material packet batch"))return b.failure(error);
            }
            finishStreamedMaterialSources<<<blocks,width,0,b.stream.get()>>>(count,b.streamedPacketRows.data(),
                b.initial.data(),b.candidateGasChange.data(),b.radiationOffsets.data(),b.radiationFaces.data(),
                radiation.empty()?nullptr:b.radiation.data(),b.radiationReceiver.data(),b.nextVolumes.data(),
                b.devicePhysics.data(),dt,finalStep,b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,
                b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
        }else if(b.slicePending){
        applyMaterialSliceSourcesKernel<<<(count+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
            b.initial.data(),b.gasChange.data(),b.candidateGasChange.data(),b.sourceSlice(),b.packetOffsets.data(),b.packetIndices.data(),
            b.radiationOffsets.data(),b.radiationFaces.data(),radiation.empty()?nullptr:b.radiation.data(),
            b.radiationReceiver.data(),b.nextVolumes.data(),b.devicePhysics.data(),dt,finalStep,
            b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
        }else{
        applyMaterialSourcesKernel<<<(count+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(count,b.current.data(),b.rates.data(),
            b.initial.data(),b.gasChange.data(),b.candidateGasChange.data(),b.packets.data(),b.packetOffsets.data(),b.packetIndices.data(),
            b.radiationOffsets.data(),b.radiationFaces.data(),radiation.empty()?nullptr:b.radiation.data(),
            b.radiationReceiver.data(),b.nextVolumes.data(),b.devicePhysics.data(),dt,finalStep,
            b.next.data(),controls.retainDonorHistory?b.donor.data():nullptr,b.sourceStatus.data(),controls.reactionMaxSubstep,controls.reactionCfl,controls.maxSubsteps);
        }
    }
    selectSourceFailure<<<1,ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads*sizeof(MaterialStepResult),b.stream.get()>>>(b.sourceStatus.data(),count,b.stepResult.data());
    if(!b.fault.check(cudaGetLastError(),"launch resident material local sources"))return b.failure(error);
    MaterialStepResult result;std::vector<SolidQ> candidateShadow,candidateDonor;
    if(observedCells){
        if(b.cachedObservationCells!=*observedCells||b.observedInventorySize!=count){
            for(int cell:*observedCells)if(cell<0||cell>=count){error="material observation cell outside inventory";return false;}
            if(!b.upload(b.observationCells,*observedCells))return b.failure(error);
            b.cachedObservationCells=*observedCells;b.observedInventorySize=count;
        }
        if(!b.allocate(b.observation,observedCells->size()))return b.failure(error);
        const int rows=observedCells->size();
        if(rows)gatherMaterialObservation<<<(rows+ugkp::gpu::workPlan().materialBlockThreads-1)/ugkp::gpu::workPlan().materialBlockThreads,ugkp::gpu::workPlan().materialBlockThreads,0,b.stream.get()>>>(rows,b.observationCells.data(),b.next.data(),b.observation.data());
        if(!b.fault.check(cudaGetLastError(),"gather material geometry observation")
            ||!b.read(b.observation,candidateShadow))return b.failure(error);
    }else if(!b.read(b.next,candidateShadow))return b.failure(error);
    if(!b.transfers.download(&result,b.stepResult.data(),1)||(controls.retainDonorHistory&&!b.read(b.donor,candidateDonor))
        ||!b.finish(error))return b.failure(error);
    if(result.phase){
        numerical=result.numerical;
        if(result.phase==1)error=materialReactionMessage(MaterialReactionCode(result.code));
        else if(result.phase==2)error=materialInventoryMessage(MaterialInventoryCode(result.code));
        else error="material endpoint inventory/EOS invalid";
        return false;
    }
    shadow=std::move(candidateShadow);cumulative=std::move(candidateDonor);
    error.clear();return true;
}
bool commitResidentMaterialStep(MaterialBackend& b,Real time,std::string& error){
    if(b.slicePending){


        if(!b.fault.check(commitResidentMaterialSlice(b.slicePlan.data(),b.candidateSlice(),b.currentSlice(),
            b.borrowedHistory.faces,b.sliceExternal.size(),b.sliceCursor.data(),nullptr,b.stream.get()),
            "commit borrowed material source slice"))return b.failure(error);
        b.slicePending=false;
    }
    if(b.externalSlicePending){
        if(!b.fault.check(cudaMemcpyAsync(b.sliceUsedExternal.data(),b.sliceCandidateExternal.data(),
            b.sliceExternal.size()*sizeof(EnergyReal),cudaMemcpyDeviceToDevice,b.stream.get()),"commit DOM boundary slice"))return b.failure(error);
        b.externalSlicePending=false;
    }
    b.current.swap(b.next);b.volumes.swap(b.nextVolumes);b.gasChange.swap(b.candidateGasChange);b.time=time;

    b.publishedCurrentValid=false;return true;
}

void commitResidentMaterialStep(MaterialBackend& b,Real time){



    std::string error;(void)commitResidentMaterialStep(b,time,error);
}

bool evaluateResidentMaterialSurfaces(MaterialBackend& b,const HostState& h,
    const std::vector<MaterialSurfaceRequest>& requests,Real dt,bool useRates,bool candidate,
    std::vector<CpuSurfaceResult>& output,std::string& error,bool& numerical,
    const std::vector<Real>* sideVolume,bool* terminalWallFailure,int localAudit){
    numerical=true;if(terminalWallFailure)*terminalWallFailure=false;
    if(localAudit>=0&&(b.localPending||std::size_t(localAudit)>=b.localAudit.size())){
        numerical=false;error="material diagnostic snapshot is unavailable";return false;
    }
    const auto& s=h.surface;const std::size_t nf=s.area.size();
    const auto& p=b.physics;
    const bool boundaryLayer=p.wallModel.family==ugkwp::gaswall::WallFamily::BoundaryLayer;
    const std::size_t count=nf*requests.size();
    std::vector<PhysicalReal> donorShares;
    if(!surfaceGasDonorShares(h,donorShares,error))return false;
    std::vector<SurfacePhysicsInput> inputs(count);
    std::vector<int> solidCells(count,-1),quadratureOffsets(count);
    std::vector<unsigned char> active(count);
    std::vector<PhysicalReal> distances,weights;
    for(std::size_t request=0;request<requests.size();++request){
        const auto& bulk=*requests[request].bulk;
        const auto& gradients=*requests[request].gradients;
        const auto* contexts=requests[request].contexts;
        const auto* accepted=requests[request].acceptedRates;
        if(accepted&&!accepted->empty()&&accepted->size()!=nf){
            numerical=false;error="accepted surface stage rate layout mismatch";return false;}
        if(!finite(dt)||dt<=0||bulk.size()!=nf||(!gradients.empty()&&gradients.size()!=nf)
            ||s.gasFace.size()!=nf||s.normal.size()!=nf||s.gasDistance.size()!=nf
            ||(!s.solidCell.empty()&&s.solidCell.size()!=nf)||(!s.baseVelocity.empty()&&s.baseVelocity.size()!=nf)){
            numerical=bulk.size()>=nf&&(gradients.empty()||gradients.size()>=nf)
                &&s.gasFace.size()>=nf&&s.normal.size()>=nf&&s.gasDistance.size()>=nf
                &&(s.solidCell.empty()||s.solidCell.size()>=nf)&&(s.baseVelocity.empty()||s.baseVelocity.size()>=nf);
            error="CPU surface input layout/interval mismatch";return false;
        }
        if(!s.gasMapOffsets.empty()||!s.gasMapFaces.empty()||!s.gasMapWeights.empty()){
            error="CPU surface requires the supported conformal interface map";return false;
        }
        if(boundaryLayer&&(!contexts||contexts->size()!=nf)){
            numerical=contexts&&contexts->size()>=nf;
            error="CPU boundaryLayer requires current matching snapshot and certified stage geometry";return false;
        }
        if(contexts)for(std::size_t f=0;f<nf;++f)if(s.gasFace[f]>=0){
            if(!b.configureMechanism((*contexts)[f].mechanism,error))return false;
            break;
        }
        for(std::size_t f=0;f<nf;++f){
            const std::size_t index=request*nf+f;
            const int gf=s.gasFace[f];if(gf<0)continue;active[index]=1;
            if(std::size_t(gf)>=h.gasMesh.owner.size()||std::size_t(gf)>=h.gasMesh.areaVectors.size()
                ||std::size_t(gf)>=h.gasMesh.faceIds.size()){error="CPU surface gas face outside real mesh";return false;}
            const int gc=h.gasMesh.owner[gf];
            if(gc<0||std::size_t(gc)>=h.gasMesh.volumes.size()){error="CPU surface gas owner outside real mesh";return false;}
            auto& in=inputs[index];in.ablating=s.ablating.empty()||s.ablating.at(f);
            setMaterialSurfaceGasDrive(in,bulk[f],gradients.empty()?nullptr:&gradients[f],PhysicalReal(gasPhaseVolume(h,gc)),donorShares[f],
                accepted&&!accepted->empty()?&(*accepted)[f]:nullptr,p);
            if(contexts){
                setMaterialSurfaceMatching(in,(*contexts)[f]);
                in.wallContext.mechanism=b.deviceMechanism;in.wallContext.workspace=nullptr;in.wallContext.failureStatus=nullptr;
                in.wallContext.teamOutput=nullptr;in.wallContext.teamDiagnostics=nullptr;in.wallContext.quietPhysics=nullptr;
                const auto& q=(*contexts)[f].quadrature;
                quadratureOffsets[index]=int(distances.size());
                for(int j=0;j<q.count;++j){distances.push_back(q.distance[j]);weights.push_back(q.volumeWeight[j]);}
            }
            setMaterialSurfaceGasGeometry(in,PhysicalReal(s.area[f]),toPhysical(s.normal[f]),toPhysical(h.gasMesh.areaVectors[gf]),PhysicalReal(s.gasDistance[f]),
                s.baseVelocity.empty()?PhysicalVec3{}:s.baseVelocity[f],dt);
            const int sc=s.solidCell.empty()?-1:s.solidCell[f];solidCells[index]=sc;in.hasSolid=sc>=0;
            if(in.hasSolid){
                if(std::size_t(sc)>=b.current.size()||std::size_t(sc)>=h.solidMesh.volumes.size()
                    ||s.solidFace.size()!=nf||s.solidDistance.size()!=nf){
                    numerical=s.solidFace.size()>=nf&&s.solidDistance.size()>=nf;
                    error="CPU interface material owner invalid";return false;
                }
                const int sf=s.solidFace[f];
                if(sf<0||std::size_t(sf)>=h.solidMesh.areaVectors.size()){error="CPU interface solid face invalid";return false;}
                setMaterialSurfaceSolidGeometry(in,PhysicalReal(s.solidDistance[f]),toPhysical(h.solidMesh.areaVectors[sf]));
            }
            if(sideVolume){
                if(sideVolume->size()!=nf){numerical=sideVolume->size()>=nf;error="surface side sweep layout mismatch";return false;}
                in.sideMeshVolumeRate=(*sideVolume)[f];
            }
        }
    }
    if(!b.upload(b.quadratureDistance,distances)||!b.upload(b.quadratureWeight,weights))return b.failure(error);
    for(std::size_t f=0;f<count;++f){
        auto& q=inputs[f].wallContext.quadrature;
        q.distance=q.count?b.quadratureDistance.data()+quadratureOffsets[f]:nullptr;
        q.volumeWeight=q.count?b.quadratureWeight.data()+quadratureOffsets[f]:nullptr;
    }
    MaterialSurfaceView view;view.count=int(count);view.workers=int(count);
    int surfaceProbeDepth=1,surfaceProbes=1;
    const bool cooperativeWall=boundaryLayer&&p.wallModel.model==ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
    if(boundaryLayer&&p.wallModel.model!=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport&&count){
        int device=0,sms=0;
        if(!b.fault.check(cudaGetDevice(&device),"query material wall device")
            ||!b.fault.check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device),"query material wall multiprocessors"))return b.failure(error);
        view.workspaceCapacity=ugkwp::gasBoundaryLayerWorkspaceCapacity(p.wallModel.nodes);
        view.workspaceStride=ugkwp::gasBoundaryLayerWorkspaceBytes<PhysicalReal,Ns>(p.wallModel.nodes);
        auto sizing=ugkwp::resolveBoundaryLayerWorkspace(int(count),p.wallWorkspaceSlots,sms,0,view.workspaceStride);
        if(!sizing.slots){error="boundaryLayer workspace has no compiled capacity or its byte count is not representable";return false;}
        if(cooperativeWall){
            if(!b.surfaceProbePlanReady){
                using Two=MaterialSurfaceRootTeam<2>;using Three=MaterialSurfaceRootTeam<3>;
                if(!b.fault.check(ugkp::gpu::residentProbeBlocks(
                        evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Two>,Two::width,
                        sizeof(Two::Storage<SurfacePhysicsResult>),sms,b.surfaceProbeResident2),"query two-round root residency")
                    ||!b.fault.check(ugkp::gpu::residentProbeBlocks(
                        evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Three>,Three::width,
                        sizeof(Three::Storage<SurfacePhysicsResult>),sms,b.surfaceProbeResident3),"query three-round root residency"))return b.failure(error);
                b.surfaceProbePlanReady=true;
            }


            if(sizing.slots<=b.surfaceProbeResident3)surfaceProbeDepth=3;
            else if(sizing.slots<=b.surfaceProbeResident2)surfaceProbeDepth=2;
            surfaceProbes=(1<<surfaceProbeDepth)-1;
            if(view.workspaceStride>std::numeric_limits<std::size_t>::max()/std::size_t(surfaceProbes)){
                error="boundaryLayer probe workspace byte count is not representable";return false;
            }
            sizing=ugkwp::resolveBoundaryLayerWorkspace(int(count),p.wallWorkspaceSlots,sms,0,view.workspaceStride*surfaceProbes);
            if(!sizing.slots){error="boundaryLayer probe workspace byte count is not representable";return false;}
        }
        if(!b.allocate(b.wallWorkspace,sizing.bytes))return b.failure(error);
        view.workers=sizing.slots;view.wallWorkspaces=b.wallWorkspace.data();
    }
    if(cooperativeWall){
        if(!b.allocate(b.surfaceTeamOutput,std::size_t(view.workers)*surfaceProbes))return b.failure(error);
        view.wallTeamOutput=b.surfaceTeamOutput.data();view.quietPhysics=b.quietSurfacePhysics.data();
    }
    if(!b.upload(b.surfaceInput,inputs)||!b.upload(b.surfaceCells,solidCells)||!b.upload(b.surfaceActive,active)
        ||!b.allocate(b.surfacePhysics,count)||!b.allocate(b.surfaceWall,count)||!b.allocate(b.surfaceStatus,count))return b.failure(error);
    view.inputs=b.surfaceInput.data();view.solidCells=b.surfaceCells.data();view.solid=candidate?b.next.data():b.current.data();
    view.volumes=candidate?b.nextVolumes.data():b.volumes.data();view.materialRate=useRates?b.rates.data():nullptr;
    if(localAudit>=0){
        view.solid=b.localAuditState.data()+std::size_t(localAudit)*b.cells;
        view.materialRate=useRates?b.localAuditRates.data()+std::size_t(localAudit)*b.cells:nullptr;

        view.volumes=b.initialVolumes.data();
    }
    view.active=b.surfaceActive.data();view.physicsOut=b.surfacePhysics.data();view.wallOut=b.surfaceWall.data();view.status=b.surfaceStatus.data();
    b.operation(h.time);
    if(count&&cooperativeWall){
        if(surfaceProbeDepth==3){
            using Team=MaterialSurfaceRootTeam<3>;
            evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),b.stream.get()>>>(view,b.devicePhysics.data());
        }else if(surfaceProbeDepth==2){
            using Team=MaterialSurfaceRootTeam<2>;
            evaluateMaterialSurfaceTeamKernel<MaterialSurfaceView,Team><<<view.workers,Team::width,sizeof(Team::Storage<SurfacePhysicsResult>),b.stream.get()>>>(view,b.devicePhysics.data());
        }else evaluateMaterialSurfaceTeamKernel<<<view.workers,ugkwp::gaswall::detail::wallWarpWidth,0,b.stream.get()>>>(view,b.devicePhysics.data());
    }else if(count){
        ugkp::gpu::ScalarWorkLaunch launch;
        if(!b.fault.check(ugkp::gpu::scalarWorkLaunch(evaluateMaterialSurfaceKernel<MaterialSurfaceView>,view.workers,launch),
            "schedule independent material surface workers"))return b.failure(error);
        view.scalarWorkersPerBlock=launch.workersPerBlock;
        evaluateMaterialSurfaceKernel<<<launch.blocks,launch.threads,0,b.stream.get()>>>(view,b.devicePhysics.data());
    }
    if(!b.fault.check(cudaGetLastError(),"evaluate resident material surface"))return b.failure(error);
    CpuSurfaceResult result;std::vector<MaterialSurfaceStatus> statuses;
    if(!b.read(b.surfacePhysics,result.physics)||!b.read(b.surfaceWall,result.wall)||!b.read(b.surfaceStatus,statuses)
        ||!b.finish(error))return b.failure(error);
    for(std::size_t f=0;f<count;++f)if(!statuses[f].success){
        const auto& status=statuses[f];numerical=status.numericalFailure;
        error="CPU surface constitutive solve rejected face "+std::to_string(f%nf)+" code "+std::to_string(status.device.code);
        if(boundaryLayer&&status.wall.code!=ugkwp::gaswall::WallCode::Success){
            if(terminalWallFailure)*terminalWallFailure=true;
            error+="; stationary wall profile code "+std::to_string(int(status.wall.code))+" node "+std::to_string(status.wall.node)
                +" iteration "+std::to_string(status.wall.iteration)+" residual "+(std::isfinite(status.wall.residual)?std::to_string(status.wall.residual):"NOT_AVAILABLE");
        }
        return false;
    }
    std::vector<CpuSurfaceResult> completed(requests.size());
    for(std::size_t i=0;i<requests.size();++i){
        completed[i].physics.assign(result.physics.begin()+i*nf,result.physics.begin()+(i+1)*nf);
        completed[i].wall.assign(result.wall.begin()+i*nf,result.wall.begin()+(i+1)*nf);
    }
    if(requests.size()==1&&localAudit<0&&!candidate){
        b.surfacePrototype=view;b.surfacePrototypeValid=true;b.surfaceCooperative=cooperativeWall;b.surfaceDepth=surfaceProbeDepth;
    }
    output=std::move(completed);error.clear();return true;
}
bool evaluateResidentMaterialSurface(MaterialBackend& b,const HostState& h,
    const std::vector<GasPrimitive>& bulk,const std::vector<GasGradient>& gradients,
    Real dt,bool useRates,bool candidate,CpuSurfaceResult& output,std::string& error,
    bool& numerical,const std::vector<Real>* sideVolume,
    const std::vector<GasWallClosureContext>* contexts,bool* terminalWallFailure,
    const std::vector<WallFaceSample>* acceptedRates,int localAudit){
    std::vector<CpuSurfaceResult> result;
    if(!evaluateResidentMaterialSurfaces(b,h,{{&bulk,&gradients,contexts,acceptedRates}},dt,useRates,candidate,
        result,error,numerical,sideVolume,terminalWallFailure,localAudit))return false;
    output=std::move(result.front());return true;
}
}
