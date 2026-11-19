#pragma once
#include "../gpu/WorkPlan.H"
#include "gasTransport/GasBoundaryLayerModelState.H"


template<bool Capture=false,class GasState>
__global__ void evaluateGasBoundaryLayerKernel(GasState* state,const unsigned char* active=nullptr)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    if constexpr(ugkwp::GasBoundaryLayerModelCapability<GasState>::value)
    {
        const int worker=blockIdx.x*blockDim.x+threadIdx.x;
        const bool analytic=state->gasBoundaryLayerModel.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
        const int workers=analytic?state->gasBoundaryLayer.count:state->gasBoundaryLayerModel.workspaceCount;
        if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
        for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
            if(!active||active[slot])ugkwp::evaluateGasBoundaryLayerSlot<Capture>(*state,slot,worker);
    }
}

#if defined(__CUDACC__)
#include "gasWall/WallCudaTeam.cuh"

template<bool Capture,int Capacity,class GasState>
__device__ inline void evaluateGasBoundaryLayerTeamWorker(GasState*,const unsigned char*,
    std::false_type){}
template<bool Capture,int Capacity,class GasState>
__device__ inline void evaluateGasBoundaryLayerTeamWorker(GasState* state,const unsigned char* active,
    std::true_type)
{
    const int worker=blockIdx.x,workers=state->gasBoundaryLayerModel.workspaceCount;
    if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
    const ugkwp::gaswall::detail::CudaWallTeam team{};
    for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
        if(!active||active[slot])ugkwp::evaluateGasBoundaryLayerSlot<Capture,Capacity>(*state,slot,worker,team);
}

template<bool Capture=false,int Capacity=0,class GasState>
__global__ __launch_bounds__(ugkwp::gaswall::detail::wallWarpWidth) void evaluateGasBoundaryLayerTeamKernel(GasState* state,const unsigned char* active=nullptr)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    evaluateGasBoundaryLayerTeamWorker<Capture,Capacity>(state,active,
        typename ugkwp::GasBoundaryLayerModelCapability<GasState>::type{});
}


template<bool Capture=false,class GasState>
__global__ void evaluateGasBoundaryLayerTile8Kernel(GasState* state,const unsigned char* active=nullptr)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    if constexpr(ugkwp::GasBoundaryLayerModelCapability<GasState>::value){
        const int workersPerBlock=blockDim.x/8;
        const int worker=workersPerBlock*blockIdx.x+threadIdx.x/8;
        const int workers=state->gasBoundaryLayerModel.workspaceCount;
        if(worker>=workers||!ugkwp::gasBoundaryLayerEnabled(*state))return;
        const ugkwp::gaswall::detail::CudaWallTile8 team{};
        for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
            if(!active||active[slot])ugkwp::evaluateGasBoundaryLayerSlot<Capture,24>(*state,slot,worker,team);
    }
}




template<bool Capture=false,class GasState,class Model>
inline cudaError_t launchGasBoundaryLayerPool(GasState* state,const Model& model,
    int wallCount,cudaStream_t stream,const unsigned char* active=nullptr)
{
    const bool analytic=model.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
    const int workers=analytic?wallCount:model.workspaceCount;
    if(workers<=0)return cudaErrorInvalidValue;
    if(analytic)
        evaluateGasBoundaryLayerKernel<Capture><<<(workers+ugkp::gpu::workPlan().faceBlockThreads-1)/ugkp::gpu::workPlan().faceBlockThreads,ugkp::gpu::workPlan().faceBlockThreads,0,stream>>>(state,active);
    else if(model.config.model==ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde
        &&model.config.nodes==8&&model.workspaceCapacity==24&&!model.conservative.enabled)
    {


        const int block=ugkp::gpu::workPlan().faceBlockThreads;
        const int workersPerBlock=block/8;
        evaluateGasBoundaryLayerTile8Kernel<Capture><<<(workers+workersPerBlock-1)/workersPerBlock,block,0,stream>>>(state,active);
    }
    else
    {
#define UGKWP_LAUNCH_WALL_CAPACITY(N) case N: evaluateGasBoundaryLayerTeamKernel<Capture,N><<<workers,ugkwp::gaswall::detail::wallWarpWidth,0,stream>>>(state,active);break
        switch(model.workspaceCapacity)
        {
            UGKWP_LAUNCH_WALL_CAPACITY(24);
            UGKWP_LAUNCH_WALL_CAPACITY(48);
            UGKWP_LAUNCH_WALL_CAPACITY(128);
            default:evaluateGasBoundaryLayerTeamKernel<Capture><<<workers,ugkwp::gaswall::detail::wallWarpWidth,0,stream>>>(state,active);
        }
#undef UGKWP_LAUNCH_WALL_CAPACITY
    }
    return cudaGetLastError();
}

#endif
