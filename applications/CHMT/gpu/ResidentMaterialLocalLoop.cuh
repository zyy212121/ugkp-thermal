#pragma once
#include "gpu/ResidentMaterialSource.cuh"
#include "gpu/ResidentMaterialRecession.cuh"
#include "gpu/ResidentMaterialStep.cuh"
#include "gpu/ResidentMaterialSlice.cuh"
#include "gpu/MaterialLocalAudit.H"
#include "gpu/MaterialLocalState.H"
#include "gpu/MaterialTransportKernels.cuh"
#include "../../../common/gasTransport/GasDriveSelection.cuh"
#include "../../../common/gpu/ExecutionGraph.cuh"
#include "../../../common/Diagnostics.H"
#include <climits>

namespace chmt {



struct MaterialLocalBindings {
    ResidentMaterialState* material=nullptr;
    ResidentMaterialStep* step=nullptr;
    const ugkp::CompensatedTime* total=nullptr;
    ugkp::IntervalSliceCursor* cursor=nullptr;
    ugkwp::GasDriveControls* driveControls=nullptr;
    ugkwp::GasDriveSelection* drive=nullptr;
    int* driveCandidates=nullptr;
    ResidentMaterialSlice slice{};
    ugkp::IntervalSlicePlan* slicePlan=nullptr;
    MaterialSliceAccumulators used{},candidateUsed{};
    const ResidentWallKnotTime* knots=nullptr;std::size_t knotCount=0;
    const PhysicsConfig* physics=nullptr;
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
    CpuMaterialControls controls{};
    MaterialTransportWorkspace transport{};
    MaterialTransportDeviceResult* transportResult=nullptr;
    materialSourceDetail::MaterialStepResult* sourceStatus=nullptr;
    materialSourceDetail::MaterialStepResult* sourceResult=nullptr;


    const int *packetOffsets=nullptr,*packetIndices=nullptr;
    const int *radiationOffsets=nullptr,*radiationFaces=nullptr;
    decltype(MaterialSliceAccumulators::radiation) radiation=nullptr;
    decltype(MaterialSliceAccumulators::radiation) externalRadiation=nullptr;
    std::size_t externalFaces=0;
    unsigned* receivers=nullptr;
    const int *gasFaces=nullptr,*solidFaces=nullptr;
    FilmAux *auxiliary=nullptr,*candidateAuxiliary=nullptr;
    Real* recessionTarget=nullptr;
    MaterialRecessionStatus* recessionStatus=nullptr;
    MaterialLocalProgress* progress=nullptr;
    const ResidentWallClock* clock=nullptr;
    MaterialLocalAudit* audit=nullptr;
    SolidQ *auditState=nullptr,*auditRates=nullptr;
    int auditCount=0;
    int cells=0,faces=0;
};

namespace material_local_detail {
__device__ inline void fail(MaterialLocalBindings b,int phase,int code,int entity){
    if(b.progress->failurePhase)return;
    b.progress->failurePhase=phase;b.progress->failureCode=code;b.progress->entity=entity;
    auto record=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::InvalidPhysicalState,
        4105,4,phase,ugkp::diagnostics::EntityKind::Cell,entity,-1,code);
    ugkp::diagnostics::recordFatal(b.diagnostics,record);
}
__global__ void begin(MaterialLocalBindings b,cudaGraphConditionalHandle loop){
    if(threadIdx.x||blockIdx.x)return;
    if(b.cursor->time==ugkp::CompensatedTime{})*b.progress={};
    b.progress->complete=b.cursor->time==*b.total;
    cudaGraphSetConditional(loop,!b.progress->complete&&!b.progress->failurePhase
        &&!ugkp::diagnostics::hasFatal(b.diagnostics));
}
__global__ void checkStep(MaterialLocalBindings b,cudaGraphConditionalHandle permitted){
    if(threadIdx.x||blockIdx.x)return;
    if(b.step->error!=MaterialStepSelectionError::None)fail(b,1,int(b.step->error),-1);
    cudaGraphSetConditional(permitted,b.step->active&&!b.progress->failurePhase
        &&!ugkp::diagnostics::hasFatal(b.diagnostics));
}
__global__ void prepareOperation(MaterialLocalBindings b){
    if(threadIdx.x||blockIdx.x)return;
    b.progress->accepted=0;
    ugkp::diagnostics::beginOperation(*b.diagnostics,b.clock->sequence,b.diagnostics->step,
        ugkp::absoluteTime(b.clock->origin,b.cursor->time));
}
__global__ void checkSlice(MaterialLocalBindings b,cudaGraphConditionalHandle permitted){
    if(threadIdx.x||blockIdx.x)return;
    if(b.slicePlan->error!=ugkp::IntervalSliceError::None)fail(b,2,int(b.slicePlan->error),-1);
    cudaGraphSetConditional(permitted,!b.progress->failurePhase&&!ugkp::diagnostics::hasFatal(b.diagnostics));
}
__global__ void receivers(MaterialLocalBindings b){
    const auto record=b.slice.history.record(b.step->drive);
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<b.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x)
        b.receivers[f]=record.receiver(f);
}
__global__ void observe(MaterialLocalBindings b){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=0;i<b.auditCount;++i){
        const auto sample=b.audit[i];
        if(sample.time<b.step->begin||!(sample.time<b.step->end))continue;


        if(c<b.cells){
            const auto offset=std::size_t(i)*b.cells+c;
            b.auditState[offset]=b.material->current[c];
            b.auditRates[offset]=b.transport.rates[c];
        }
    }
}
__global__ void observeMetadata(MaterialLocalBindings b){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=b.auditCount)return;
    auto& sample=b.audit[i];
    if(sample.time<b.step->begin||!(sample.time<b.step->end))return;
    sample.heldBegin=b.step->begin;sample.heldEnd=b.step->end;
    sample.dt=b.step->dt;sample.ready=true;
}
__device__ inline int lower(const int* indices,int begin,int end,std::size_t value){
    while(begin<end){const int middle=begin+(end-begin)/2;
        if(std::size_t(indices[middle])<value)begin=middle+1;else end=middle;
    }
    return begin;
}
__global__ void sources(MaterialLocalBindings b){
    const auto state=*b.material;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=b.cells)return;
    b.sourceStatus[c]={};
    const auto width=b.slice.history.faces*3;
    const auto first=b.slicePlan->begin.record*width,past=b.slicePlan->past*width;
    const int start=lower(b.packetIndices,b.packetOffsets[c],b.packetOffsets[c+1],first);
    const int end=lower(b.packetIndices,start,b.packetOffsets[c+1],past);
    const auto transported=addSolid(state.current[c],b.transport.rates[c],b.step->dt);
    MaterialPacketAccumulator sum;sum.begin(transported);auto change=state.gasChange[c];
    for(int e=start;e<end;++e){
        const auto packet=b.slice.packet(std::size_t(b.packetIndices[e])-first);
        sum.append(packet);change=addSolid(change,packetDelta(packet).solid);
    }
    SolidQ value;const auto code=sum.finish(transported,value);
    if(code!=MaterialInventoryCode::Success){b.sourceStatus[c]={2,int(code),c,true};return;}
    if(!materialSourceDetail::finishMaterialSourceCell(c,value,b.radiationOffsets,b.radiationFaces,
        b.radiation,b.receivers,state.candidateMesh.volumes,b.physics,b.step->dt,b.step->finalStep,b.sourceStatus,
        b.controls.reactionMaxSubstep,b.controls.reactionCfl,b.controls.maxSubsteps))return;
    state.candidate[c]=value;state.candidateGasChange[c]=change;
}
__global__ void acceptance(MaterialLocalBindings b){
    if(threadIdx.x||blockIdx.x)return;
    b.progress->accepted=0;
    for(std::size_t f=0;f<b.slice.history.faces;++f)
        if(b.recessionStatus[f].error!=MaterialRecessionError::None){
            fail(b,3,int(b.recessionStatus[f].error),int(f));return;
        }
    const auto result=*b.sourceResult;
    if(result.phase){fail(b,4+result.phase,result.code,result.entity);return;}
    b.progress->accepted=!b.progress->failurePhase&&!ugkp::diagnostics::hasFatal(b.diagnostics);
}
__global__ void commitAuxiliary(MaterialLocalBindings b){
    if(!b.progress->accepted)return;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<b.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x)
        b.auxiliary[f]=b.candidateAuxiliary[f];
}
__global__ void advance(MaterialLocalBindings b,cudaGraphConditionalHandle loop){
    if(threadIdx.x||blockIdx.x)return;
    if(b.progress->accepted)++b.progress->acceptedSteps;
    if(b.progress->accepted&&b.progress->acceptedSteps==std::uint64_t(b.controls.maxSubsteps)){
        auto record=ugkp::diagnostics::makeRecord(ugkp::diagnostics::Code::AccuracyResidual,
            4106,4,0,ugkp::diagnostics::EntityKind::None,-1,-1,
            double(b.progress->acceptedSteps),b.controls.maxSubsteps);
        ugkp::diagnostics::recordWarning(b.diagnostics,record);
    }
    b.progress->complete=b.cursor->time==*b.total;
    cudaGraphSetConditional(loop,!b.progress->complete&&!b.progress->failurePhase
        &&!ugkp::diagnostics::hasFatal(b.diagnostics));
}
}

inline ugkp::gpu::ExecutionGraphNode appendResidentMaterialLocalLoop(ugkp::gpu::ExecutionGraph graph,
    ugkp::gpu::ExecutionGraphNode dependency,MaterialLocalBindings b,cudaStream_t stream){
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    if(!b.progress||!b.material||!b.step||!b.total||!b.cursor||!b.physics||!b.diagnostics||!b.clock)
        throw std::invalid_argument("missing material local-loop state bindings");
    if(b.slice.history.faces&&b.slice.history.frames>std::size_t(INT_MAX)/b.slice.history.faces/3)
        throw std::overflow_error("material packet row capacity exceeds addressable slots");
    b.slice.plan=b.slicePlan;b.slice.used=b.used;b.slice.hostSelected=false;b.slice.particles=true;
    const int threads=workPlan().materialBlockThreads;
    const int cells=std::max(1,(b.cells+threads-1)/threads);
    const unsigned faces=material_slice_detail::blocks(std::max(b.slice.history.faces,b.externalFaces));
    cudaGraphConditionalHandle loop,selected,transported,sliced;
    for(auto handle:{&loop,&selected,&transported,&sliced})checked(cudaGraphConditionalHandleCreate(handle,graph,0,cudaGraphCondAssignDefault));
    const auto start=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        material_local_detail::begin<<<1,1,0,stream>>>(b,loop);
    });
    cudaGraph_t body=nullptr;const auto completed=addWhile(graph,loop,{start},body);
    const auto request=appendRecorded(body,{},stream,[&]{
        material_local_detail::prepareOperation<<<1,1,0,stream>>>(b);
        requestResidentMaterialStep<<<1,1,0,stream>>>(b.cursor,b.total,b.controls,b.step,b.driveControls);
        checked(ugkwp::selectOrderedGasDriveInterval<Ns>(b.slice.history,b.driveControls,
            &b.cursor->record,b.drive,b.driveCandidates,stream));
        alignResidentMaterialStep<<<1,1,0,stream>>>(b.step,b.drive,b.knots,b.knotCount);
        material_local_detail::checkStep<<<1,1,0,stream>>>(b,selected);
    });
    cudaGraph_t transportBody=nullptr;const auto transportNode=addIf(body,selected,{request},transportBody);
    const auto transport=appendRecorded(transportBody,{},stream,[&]{
        checked(launchMaterialTransport(b.cells,b.faces,DeviceValue<MaterialMeshView>{&b.material->mesh},
            DeviceValue<SolidQ*>{&b.material->current},b.physics,static_cast<const Real*>(nullptr),
            DeviceValue<Real>{&b.step->dt},b.controls.transportCfl,b.transport,b.transportResult,stream));
        limitResidentMaterialStep<<<1,1,0,stream>>>(b.step,b.total,b.transportResult);
        material_local_detail::checkStep<<<1,1,0,stream>>>(b,transported);
    });
    cudaGraph_t sliceBody=nullptr;addIf(transportBody,transported,{transport},sliceBody);
    const auto slice=appendRecorded(sliceBody,{},stream,[&]{
        material_slice_detail::plan<<<1,1,0,stream>>>(b.slice.history,b.cursor,&b.step->end,b.externalFaces,b.slicePlan);
        material_local_detail::checkSlice<<<1,1,0,stream>>>(b,sliced);
    });
    cudaGraph_t sourceBody=nullptr;addIf(sliceBody,sliced,{slice},sourceBody);
    const auto sources=appendRecorded(sourceBody,{},stream,[&]{
        material_local_detail::observe<<<cells,threads,0,stream>>>(b);
        material_local_detail::observeMetadata<<<std::max(1,(b.auditCount+threads-1)/threads),threads,0,stream>>>(b);
        prepareResidentMaterialRecession<<<faces,threads,0,stream>>>(b.slice,b.material,b.step,b.physics,
            b.gasFaces,b.solidFaces,b.auxiliary,b.recessionTarget,b.candidateAuxiliary,b.recessionStatus);
        material_local_detail::receivers<<<faces,threads,0,stream>>>(b);
        material_slice_detail::prepare<<<faces,threads,0,stream>>>(b.slice,b.candidateUsed,
            b.externalRadiation,b.externalFaces,DeviceValue<ugkp::CompensatedTime>{b.total},b.radiation);
        checked(cudaMemsetAsync(b.sourceResult,0,sizeof(*b.sourceResult),stream));
        material_local_detail::sources<<<cells,threads,0,stream>>>(b);
        materialSourceDetail::selectSourceFailure<<<1,threads,threads*sizeof(materialSourceDetail::MaterialStepResult),stream>>>(b.sourceStatus,b.cells,b.sourceResult);
        material_local_detail::acceptance<<<1,1,0,stream>>>(b);
        checked(commitResidentMaterialSlice(b.slicePlan,b.candidateUsed,b.used,b.slice.history.faces,b.externalFaces,
            b.cursor,&b.progress->accepted,stream));
        commitResidentMaterialState<<<1,1,0,stream>>>(b.material,&b.progress->accepted);
        material_local_detail::commitAuxiliary<<<faces,threads,0,stream>>>(b);
    });
    (void)sources;
    appendRecorded(body,{transportNode},stream,[&]{material_local_detail::advance<<<1,1,0,stream>>>(b,loop);});
    return completed;
}
}
