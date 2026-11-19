#pragma once
#include "gpu/ResidentMaterialLocalLoop.cuh"
#include "gpu/MaterialSurfaceKernels.cuh"
#include "ablation/MaterialSurfaceInput.H"
#include "../../../common/GpuPrecisionTypes.H"
namespace chmt {
struct MaterialEndpointBindings {
    MaterialLocalBindings local{};
    ResidentMaterialDriveHistory history{};
    MaterialSurfaceView prototype{};
    MaterialSurfaceView* operation=nullptr;
    SurfacePhysicsInput* inputs=nullptr;
    const int* gasCells=nullptr;
    const Real* gasVolume=nullptr;const GpuReal* donorShare=nullptr;
    const GpuReal* gasFraction=nullptr;
    const int* conductionStage=nullptr;
    int* complete=nullptr;
    Real dt=0;bool matching=false;
};
namespace material_endpoint_detail {
__device__ inline void fail(MaterialEndpointBindings b,int code,int face,double value=0){
    using namespace ugkp::diagnostics;
    auto record=makeRecord(Code::InvalidPhysicalState,4107,4,0,EntityKind::Face,face,-1,value);
    record.metadata=std::uint32_t(code);recordFatal(b.local.diagnostics,record);
}
__global__ void select(MaterialEndpointBindings b,cudaGraphConditionalHandle condition,bool prepared){
    if(threadIdx.x||blockIdx.x)return;
    bool run=b.local.progress->complete&&!b.local.progress->failurePhase&&!(*b.complete)
        &&(!b.conductionStage||*b.conductionStage==4)&&!ugkp::diagnostics::hasFatal(b.local.diagnostics);
    if(run&&!prepared){
        ugkp::diagnostics::beginOperation(*b.local.diagnostics,b.local.clock->sequence,b.local.diagnostics->step,
            ugkp::absoluteTime(b.local.clock->origin,*b.local.total));
        const auto h=b.history.resolved();
        if(!h.frames||h.frames>b.history.frames||!h.surface||h.faces!=std::size_t(b.prototype.count)
            ||(b.matching&&!h.wall)){fail(b,1,-1,double(h.frames));run=false;}
    }
    cudaGraphSetConditional(condition,run);
}
__global__ void prepare(MaterialEndpointBindings b){
    const auto history=b.history.resolved();const auto record=history.record(history.frames-1);
    const auto state=*b.local.material;
    if(!blockIdx.x&&!threadIdx.x){
        auto view=b.prototype;view.solid=state.current;view.volumes=state.mesh.volumes;view.materialRate=nullptr;
        *b.operation=view;
    }
    for(int f=blockIdx.x*blockDim.x+threadIdx.x;f<b.prototype.count;f+=blockDim.x*gridDim.x){
        if(b.prototype.active&&!b.prototype.active[f])continue;
        auto& input=b.inputs[f];const int cell=b.gasCells[f];


        const Real phaseVolume=b.gasVolume[f]*(b.gasFraction?Real(b.gasFraction[cell]):Real(1));
        setMaterialSurfaceGasDrive(input,record.bulk(f),&record.gradient(f),phaseVolume,b.donorShare[f],
            &record.surface[f].acceptedWall,*b.local.physics);
        if(b.matching){const auto& sample=record.wall[f].matching;
            if(!finite(sample.mechanicalPressure)||sample.mechanicalPressure<=0){fail(b,2,b.local.gasFaces[f],sample.mechanicalPressure);continue;}
            auto& context=input.wallContext;context.matching=sample.state;
            context.matchingPressure=sample.pressure;context.mechanicalPressure=sample.mechanicalPressure;
            input.bulk.pressure=context.matchingPressure;
        }
        input.dt=b.dt;input.aux.pressure=input.bulk.pressure;
    }
}
__global__ void inspect(MaterialEndpointBindings b){
    for(int f=blockIdx.x*blockDim.x+threadIdx.x;f<b.prototype.count;f+=blockDim.x*gridDim.x){
        const auto status=b.prototype.status[f];
        if(!status.success)fail(b,3+int(status.device.code),b.local.gasFaces[f],status.wall.residual);
    }
}
__global__ void finish(MaterialEndpointBindings b){
    if(threadIdx.x||blockIdx.x)return;
    if(!ugkp::diagnostics::hasFatal(b.local.diagnostics))*b.complete=1;
}
}
}
