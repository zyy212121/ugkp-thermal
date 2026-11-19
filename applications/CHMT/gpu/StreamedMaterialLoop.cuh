#pragma once
#include "gpu/ResidentMaterialLocalLoop.cuh"
#include "gpu/StreamedMaterialState.H"
namespace chmt {
struct StreamedMaterialBindings {
    MaterialLocalBindings local{};
    StreamedMaterialProgress* progress=nullptr;
    const StreamedMaterialChunk* chunks=nullptr;
    CoupledSurfaceDownload* heldSurface=nullptr;
    CoupledWallDownload* heldWall=nullptr;
    bool matching=false;
    MaterialPacketAccumulator* packets=nullptr;
    SolidQ* transported=nullptr;
};
namespace streamed_material_detail {
__device__ inline int available(StreamedMaterialBindings b,std::size_t frame){
    for(int i=0;i<2;++i)if(b.chunks[i].contains(frame))return i;
    return -1;
}
__device__ inline ResidentMaterialDriveRecord held(StreamedMaterialBindings b){
    const auto faces=b.local.slice.history.faces;
    return {b.heldSurface,b.matching?b.heldWall:nullptr,faces,b.matching?faces:0};
}
__global__ void select(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*b.progress;const auto local=b.local;
    if(local.cursor->time==*local.total){p.phase=StreamedMaterialPhase::Complete;local.progress->complete=true;return;}
    local.progress->accepted=0;
    ugkp::diagnostics::beginOperation(*local.diagnostics,local.clock->sequence,local.diagnostics->step,
        ugkp::absoluteTime(local.clock->origin,local.cursor->time));
    *local.step=requestMaterialStep(local.cursor->time,*local.total,local.controls.maxSubstep,
        local.controls.driveRelativeTolerance,local.controls.driveTractionScale,*local.driveControls);
    p.held=local.cursor->record;
    while(p.held<local.slice.history.frames&&!(local.cursor->time<local.slice.history.interval[p.held].end))++p.held;
    if(p.held>=local.slice.history.frames){material_local_detail::fail(local,1,int(ugkwp::GasDriveSelectionError::NoSample),-1);return;}
    p.search=p.held+1;p.phase=StreamedMaterialPhase::Held;
}
__global__ void requireHeld(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x)return;
    if(available(b,b.progress->held)<0)b.progress->requestFrame=b.progress->held;
}
__global__ void copyHeld(StreamedMaterialBindings b){
    const int slot=available(b,b.progress->held);if(slot<0)return;
    const auto record=b.chunks[slot].history.record(b.progress->held-b.chunks[slot].first);
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<b.local.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x){
        b.heldSurface[f]=record.surface[f];
        if(b.matching)b.heldWall[f]=record.wall[f];
    }
}
__global__ void heldReady(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x||available(b,b.progress->held)<0)return;
    b.local.drive->held=b.progress->held;b.local.drive->first=b.local.slice.history.frames;
    b.local.drive->end=b.local.driveControls->requested;b.local.drive->error=ugkwp::GasDriveSelectionError::None;
    b.progress->phase=StreamedMaterialPhase::Search;
}
__global__ void search(StreamedMaterialBindings b){
    auto& p=*b.progress;const auto local=b.local;const auto control=*local.driveControls;
    const auto history=local.slice.history;
    if(p.search>=history.frames||!(history.interval[p.search].begin<control.requested)){
        if(!threadIdx.x)p.phase=StreamedMaterialPhase::Transport;return;
    }
    const int slot=available(b,p.search);
    if(slot<0){if(!threadIdx.x)p.requestFrame=p.search;return;}
    const auto chunk=b.chunks[slot];const auto current=held(b);
    __shared__ unsigned first;__shared__ int failure;
    const std::size_t begin=p.search,past=chunk.first+chunk.history.frames;
    for(std::size_t base=begin;base<past;base+=blockDim.x){
        if(!(history.interval[base].begin<control.requested))break;
        if(!threadIdx.x){first=blockDim.x;failure=0;}__syncthreads();
        const std::size_t frame=base+threadIdx.x;
        int decision=0;
        if(frame<past&&history.interval[frame].begin<control.requested){
            const auto change=ugkwp::compareGasDrive<Ns>(current,chunk.history.record(frame-chunk.first),control.tractionScale);
            if(!change.layoutMatches)decision=2;
            else if(!::isfinite(change.change))decision=3;
            else if(change.change>control.tolerance||change.receiverChanged)decision=1;
            if(decision)atomicMin(&first,threadIdx.x);
        }
        __syncthreads();
        if(first<blockDim.x){
            if(threadIdx.x==first)failure=decision;__syncthreads();
            if(!threadIdx.x){
                const auto frame=base+first;local.drive->first=frame;
                if(failure>1){
                    local.drive->error=failure==2?ugkwp::GasDriveSelectionError::LayoutChanged:ugkwp::GasDriveSelectionError::NonfiniteChange;
                    material_local_detail::fail(local,1,int(local.drive->error),-1);
                }else{local.drive->end=history.interval[frame].begin;p.phase=StreamedMaterialPhase::Transport;}
            }
            return;
        }
        __syncthreads();
    }
    if(!threadIdx.x){
        p.search=past;
        if(past>=history.frames||!(history.interval[past].begin<control.requested))p.phase=StreamedMaterialPhase::Transport;
    }
}
__device__ inline ExchangePacket packet(StreamedMaterialBindings b,const StreamedMaterialChunk& chunk,
    std::size_t frame,std::size_t face,unsigned kind){
    const auto local=b.local;const auto& slice=local.slice;
    const auto value=materialHistoryPacket(chunk.history.surface[(frame-chunk.first)*slice.history.faces+face],kind);
    return slice.recordLast(frame)?subtractIntervalPacket(value,frame==slice.plan->begin.record?local.used.packets[3*face+kind]:ExchangePacket{})
        :scaledIntervalPacket(value,slice.recordFraction(frame));
}
__device__ inline bool active(const CoupledSurfaceDownload& raw,unsigned kind){
    return kind==0||hasPacketTransfer(materialHistoryPacket(raw,kind));
}
__global__ void beginPackets(StreamedMaterialBindings b){
    const auto local=b.local;const auto state=*local.material;
    for(int c=blockIdx.x*blockDim.x+threadIdx.x;c<local.cells;c+=blockDim.x*gridDim.x){
        b.transported[c]=addSolid(state.current[c],local.transport.rates[c],local.step->dt);
        b.packets[c].begin(b.transported[c]);state.candidateGasChange[c]=state.gasChange[c];local.sourceStatus[c]={};
    }
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<local.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x){
        local.radiation[f]=0;local.candidateUsed.radiation[f]=local.used.radiation[f];
        local.receivers[f]=b.heldSurface[f].radiationReceiver;
        local.recessionTarget[f]=0;local.recessionStatus[f]={};
        for(int k=0;k<3;++k)local.candidateUsed.packets[3*f+k]={};
    }
}
__global__ void packetRange(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x)return;
    auto& p=*b.progress;const auto plan=*b.local.slicePlan;
    if(p.packetRecord>=plan.past){p.phase=StreamedMaterialPhase::Finish;return;}
    p.packetSlot=available(b,p.packetRecord);
    if(p.packetSlot<0){p.requestFrame=p.packetRecord;return;}
    const auto chunk=b.chunks[p.packetSlot];
    p.packetPast=plan.past<chunk.first+chunk.history.frames?plan.past:chunk.first+chunk.history.frames;
}
__global__ void accumulatePackets(StreamedMaterialBindings b){
    const auto p=*b.progress;if(p.packetSlot<0||p.phase!=StreamedMaterialPhase::Packets)return;
    const auto local=b.local;const auto chunk=b.chunks[p.packetSlot];
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=local.cells)return;
    const auto width=chunk.history.faces*3;
    const auto first=(p.packetRecord-chunk.first)*width,past=(p.packetPast-chunk.first)*width;
    const int start=material_local_detail::lower(chunk.packetIndices,chunk.packetOffsets[c],chunk.packetOffsets[c+1],first);
    const int end=material_local_detail::lower(chunk.packetIndices,start,chunk.packetOffsets[c+1],past);
    auto& accumulator=b.packets[c];auto& gasChange=local.material->candidateGasChange[c];
    for(int i=start;i<end;++i){
        const auto index=std::size_t(chunk.packetIndices[i]);
        const auto value=packet(b,chunk,chunk.first+index/width,(index%width)/3,index%3);
        accumulator.append(value);gasChange=addSolid(gasChange,packetDelta(value).solid);
    }
}
__global__ void accumulateFaces(StreamedMaterialBindings b){
    const auto p=*b.progress;if(p.packetSlot<0||p.phase!=StreamedMaterialPhase::Packets)return;
    const auto local=b.local;const auto chunk=b.chunks[p.packetSlot];const auto plan=*local.slicePlan;
    using Energy=typename std::remove_pointer<decltype(local.radiation)>::type;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<local.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x){
        Energy used=local.candidateUsed.radiation[f],sum=local.radiation[f];
        Real target=local.recessionTarget[f];auto status=local.recessionStatus[f];
        for(std::size_t r=p.packetRecord;r<p.packetPast;++r){
            const auto& raw=chunk.history.surface[(r-chunk.first)*chunk.history.faces+f];
            const bool last=local.slice.recordLast(r);
            const Energy part=last?raw.radiationEnergy-used:Energy(local.slice.recordFraction(r))*raw.radiationEnergy;
            used+=part;sum+=part;if(last)used=0;
            for(unsigned k=0;k<3;++k)if(active(raw,k)){
                const auto value=packet(b,chunk,r,f,k);
                if(status.error==MaterialRecessionError::None&&value.kind==ExchangeKind::GasSolid){
                    status.error=addMaterialRecessionTarget(value.condensed,local.material->current[value.solidCell],
                        local.material->mesh.volumes[value.solidCell],*local.physics,target);
                    if(status.error!=MaterialRecessionError::None)status.packet=(r-plan.begin.record)*chunk.history.faces*3+3*f+k;
                }
                if(plan.end.record<plan.past&&r==plan.end.record){
                    auto cumulative=r==plan.begin.record?local.used.packets[3*f+k]:ExchangePacket{};
                    addIntervalPacket(cumulative,value);local.candidateUsed.packets[3*f+k]=cumulative;
                }
            }
        }
        local.candidateUsed.radiation[f]=used;local.radiation[f]=sum;
        local.recessionTarget[f]=target;local.recessionStatus[f]=status;
    }
}
__global__ void consumedChunk(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x||b.progress->packetSlot<0)return;
    b.progress->packetRecord=b.progress->packetPast;b.progress->packetSlot=-1;
    if(b.progress->packetRecord==b.local.slicePlan->past)b.progress->phase=StreamedMaterialPhase::Finish;
}
__global__ void preparationDone(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x)return;
    const auto local=b.local;
    if(local.step->error!=MaterialStepSelectionError::None){material_local_detail::fail(local,1,int(local.step->error),-1);return;}
    if(local.slicePlan->error!=ugkp::IntervalSliceError::None){material_local_detail::fail(local,2,int(local.slicePlan->error),-1);return;}
    b.progress->packetRecord=local.slicePlan->begin.record;b.progress->packetSlot=-1;
    b.progress->phase=StreamedMaterialPhase::BeginPackets;
}
__global__ void setPhase(StreamedMaterialBindings b,StreamedMaterialPhase phase){
    if(!threadIdx.x&&!blockIdx.x)b.progress->phase=phase;
}
__global__ void finishFaces(StreamedMaterialBindings b){
    const auto local=b.local;
    using Energy=typename std::remove_pointer<decltype(local.radiation)>::type;
    for(std::size_t f=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;f<local.slice.history.faces;f+=std::size_t(blockDim.x)*gridDim.x){
        local.candidateAuxiliary[f]=local.auxiliary[f];
        if(local.recessionStatus[f].error==MaterialRecessionError::None&&local.gasFaces[f]>=0){
            const int face=local.solidFaces[f];const Real area=face>=0?mag(local.material->mesh.areaVectors[face]):Real(0);
            local.candidateAuxiliary[f]=materialRecessionFront(local.auxiliary[f],local.recessionTarget[f],area,local.step->dt);
        }
        if(f<local.externalFaces){
            const Real fraction=(local.slicePlan->target-local.slicePlan->begin.time).value()/local.total->value();
            const bool last=local.slicePlan->target==*local.total;
            const Energy used=local.used.externalRadiation[f];
            const Energy part=last?local.externalRadiation[f]-used:local.externalRadiation[f]*Energy(fraction);
            local.candidateUsed.externalRadiation[f]=used+part;local.radiation[f]+=part;
        }
    }
}
__global__ void finishSources(StreamedMaterialBindings b){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;const auto local=b.local;if(c>=local.cells)return;
    SolidQ value;const auto code=b.packets[c].finish(b.transported[c],value);
    if(code!=MaterialInventoryCode::Success){local.sourceStatus[c]={2,int(code),c,true};return;}
    if(!materialSourceDetail::finishMaterialSourceCell(c,value,local.radiationOffsets,local.radiationFaces,
        local.radiation,local.receivers,local.material->candidateMesh.volumes,local.physics,local.step->dt,
        local.step->finalStep,local.sourceStatus,local.controls.reactionMaxSubstep,local.controls.reactionCfl,local.controls.maxSubsteps))return;
    local.material->candidate[c]=value;
}
__global__ void advance(StreamedMaterialBindings b){
    if(threadIdx.x||blockIdx.x)return;const auto local=b.local;
    if(local.progress->accepted){
        ++local.progress->acceptedSteps;
        if(local.progress->acceptedSteps==std::uint64_t(local.controls.maxSubsteps))
            ugkp::diagnostics::recordWarning(local.diagnostics,ugkp::diagnostics::makeRecord(
                ugkp::diagnostics::Code::AccuracyResidual,4106,4,0,ugkp::diagnostics::EntityKind::None,-1,-1,
                double(local.progress->acceptedSteps),local.controls.maxSubsteps));
    }
    local.progress->complete=local.cursor->time==*local.total;
    b.progress->phase=local.progress->complete?StreamedMaterialPhase::Complete:StreamedMaterialPhase::Select;
}
__device__ inline bool runnable(StreamedMaterialBindings b){
    return !b.local.progress->failurePhase&&!ugkp::diagnostics::hasFatal(b.local.diagnostics)
        &&b.progress->requestFrame==std::numeric_limits<std::size_t>::max();
}
__global__ void choose(StreamedMaterialBindings b,StreamedMaterialPhase phase,cudaGraphConditionalHandle permitted){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(permitted,runnable(b)&&b.progress->phase==phase);
}
__global__ void resume(StreamedMaterialBindings b,cudaGraphConditionalHandle loop){
    if(threadIdx.x||blockIdx.x)return;
    b.progress->requestFrame=std::numeric_limits<std::size_t>::max();
    cudaGraphSetConditional(loop,runnable(b)&&b.progress->phase!=StreamedMaterialPhase::Complete);
}
__global__ void continuation(StreamedMaterialBindings b,cudaGraphConditionalHandle loop){
    if(!threadIdx.x&&!blockIdx.x)cudaGraphSetConditional(loop,runnable(b)&&b.progress->phase!=StreamedMaterialPhase::Complete);
}
}

inline ugkp::gpu::ExecutionGraphNode appendStreamedMaterialLoop(ugkp::gpu::ExecutionGraph graph,
    ugkp::gpu::ExecutionGraphNode dependency,StreamedMaterialBindings b,cudaStream_t stream){
    using namespace ugkp::gpu;using namespace executionGraphDetail;
    if(!b.progress||!b.chunks||!b.local.material||!b.local.progress||!b.local.diagnostics)
        throw std::invalid_argument("missing streamed material state or history storage");
    b.local.slice.plan=b.local.slicePlan;b.local.slice.used=b.local.used;
    b.local.slice.hostSelected=false;b.local.slice.particles=true;
    const auto local=b.local;const int threads=workPlan().materialBlockThreads;
    const int cells=std::max(1,(local.cells+threads-1)/threads);
    const auto faces=material_slice_detail::blocks(std::max(local.slice.history.faces,local.externalFaces));
    cudaGraphConditionalHandle loop;checked(cudaGraphConditionalHandleCreate(&loop,graph,0,cudaGraphCondAssignDefault));
    const auto start=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
        streamed_material_detail::resume<<<1,1,0,stream>>>(b,loop);
    });
    cudaGraph_t body=nullptr;const auto completed=addWhile(graph,loop,{start},body);
    cudaGraphNode_t previous=nullptr;
    const StreamedMaterialPhase phases[]={StreamedMaterialPhase::Select,StreamedMaterialPhase::Held,StreamedMaterialPhase::Search,
        StreamedMaterialPhase::Transport,StreamedMaterialPhase::BeginPackets,StreamedMaterialPhase::Packets,
        StreamedMaterialPhase::Finish,StreamedMaterialPhase::Commit};
    for(const auto phase:phases){
        cudaGraphConditionalHandle permitted;checked(cudaGraphConditionalHandleCreate(&permitted,graph,0,cudaGraphCondAssignDefault));
        const auto selected=appendRecorded(body,previous?std::vector<cudaGraphNode_t>{previous}:std::vector<cudaGraphNode_t>{},stream,[&]{
            streamed_material_detail::choose<<<1,1,0,stream>>>(b,phase,permitted);
        });
        cudaGraph_t stage=nullptr;previous=addIf(body,permitted,{selected},stage);
        appendRecorded(stage,{},stream,[&]{
            switch(phase){
            case StreamedMaterialPhase::Select:
                streamed_material_detail::select<<<1,1,0,stream>>>(b);break;
            case StreamedMaterialPhase::Held:
                streamed_material_detail::requireHeld<<<1,1,0,stream>>>(b);
                streamed_material_detail::copyHeld<<<faces,threads,0,stream>>>(b);
                streamed_material_detail::heldReady<<<1,1,0,stream>>>(b);break;
            case StreamedMaterialPhase::Search:
                streamed_material_detail::search<<<1,threads,0,stream>>>(b);break;
            case StreamedMaterialPhase::Transport:
                alignResidentMaterialStep<<<1,1,0,stream>>>(local.step,local.drive,local.knots,local.knotCount);
                checked(launchMaterialTransport(local.cells,local.faces,DeviceValue<MaterialMeshView>{&local.material->mesh},
                    DeviceValue<SolidQ*>{&local.material->current},local.physics,static_cast<const Real*>(nullptr),
                    DeviceValue<Real>{&local.step->dt},local.controls.transportCfl,local.transport,local.transportResult,stream));
                limitResidentMaterialStep<<<1,1,0,stream>>>(local.step,local.total,local.transportResult);
                material_slice_detail::plan<<<1,1,0,stream>>>(local.slice.history,local.cursor,&local.step->end,local.externalFaces,local.slicePlan);
                streamed_material_detail::preparationDone<<<1,1,0,stream>>>(b);break;
            case StreamedMaterialPhase::BeginPackets:
                material_local_detail::observe<<<cells,threads,0,stream>>>(local);
                material_local_detail::observeMetadata<<<std::max(1,(local.auditCount+threads-1)/threads),threads,0,stream>>>(local);
                streamed_material_detail::beginPackets<<<std::max(cells,int(faces)),threads,0,stream>>>(b);
                streamed_material_detail::setPhase<<<1,1,0,stream>>>(b,StreamedMaterialPhase::Packets);break;
            case StreamedMaterialPhase::Packets:
                streamed_material_detail::packetRange<<<1,1,0,stream>>>(b);
                streamed_material_detail::accumulatePackets<<<cells,threads,0,stream>>>(b);
                streamed_material_detail::accumulateFaces<<<faces,threads,0,stream>>>(b);
                streamed_material_detail::consumedChunk<<<1,1,0,stream>>>(b);break;
            case StreamedMaterialPhase::Finish:
                streamed_material_detail::finishFaces<<<faces,threads,0,stream>>>(b);
                checked(cudaMemsetAsync(local.sourceResult,0,sizeof(*local.sourceResult),stream));
                streamed_material_detail::finishSources<<<cells,threads,0,stream>>>(b);
                materialSourceDetail::selectSourceFailure<<<1,threads,threads*sizeof(materialSourceDetail::MaterialStepResult),stream>>>(local.sourceStatus,local.cells,local.sourceResult);
                streamed_material_detail::setPhase<<<1,1,0,stream>>>(b,StreamedMaterialPhase::Commit);break;
            case StreamedMaterialPhase::Commit:
                material_local_detail::acceptance<<<1,1,0,stream>>>(local);
                checked(commitResidentMaterialSlice(local.slicePlan,local.candidateUsed,local.used,local.slice.history.faces,local.externalFaces,
                    local.cursor,&local.progress->accepted,stream));
                commitResidentMaterialState<<<1,1,0,stream>>>(local.material,&local.progress->accepted);
                material_local_detail::commitAuxiliary<<<faces,threads,0,stream>>>(local);
                streamed_material_detail::advance<<<1,1,0,stream>>>(b);break;
            default:break;
            }
        });
    }
    appendRecorded(body,{previous},stream,[&]{streamed_material_detail::continuation<<<1,1,0,stream>>>(b,loop);});
    return completed;
}
}
