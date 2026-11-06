#pragma once
#include "gasTransport/GasBoundaryLayerModelState.H"
// One sparse worker owns one bounded scratch slot and walks distinct faces.
// Expensive profile scratch scales with workspaceCount, not wall-face count.
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
// Full CTA entry/exit is uniform even when nodes<32 or an evaluation fails.
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
// Node-parallel nonlinear BVP; scalar/analytic kernel remains available above.
template<bool Capture=false,int Capacity=0,class GasState>
__global__ __launch_bounds__(ugkwp::gaswall::detail::wallWarpWidth) void evaluateGasBoundaryLayerTeamKernel(GasState* state,const unsigned char* active=nullptr)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    evaluateGasBoundaryLayerTeamWorker<Capture,Capacity>(state,active,
        typename ugkwp::GasBoundaryLayerModelCapability<GasState>::type{});
}
// Nodes 0..7 occupy a tile; each tile owns a distinct scratch slot. A partial
// final block returns entire tiles only, never individual participating lanes.
template<bool Capture=false,class GasState>
__global__ __launch_bounds__(32) void evaluateGasBoundaryLayerTile8Kernel(GasState* state,const unsigned char* active=nullptr)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    if constexpr(ugkwp::GasBoundaryLayerModelCapability<GasState>::value){
        const int worker=4*blockIdx.x+threadIdx.x/8;
        const int workers=state->gasBoundaryLayerModel.workspaceCount;
        if(worker>=workers||!ugkwp::gasBoundaryLayerEnabled(*state))return;
        const ugkwp::gaswall::detail::CudaWallTile8 team{};
        for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
            if(!active||active[slot])ugkwp::evaluateGasBoundaryLayerSlot<Capture,24>(*state,slot,worker,team);
    }
}

// All application adapters submit the same resident wall pool. The optional
// mask selects whole wall teams, preserving accepted/candidate history for
// unchanged slots without letting individual lanes skip a collective.
template<bool Capture=false,class GasState,class Model>
inline cudaError_t launchGasBoundaryLayerPool(GasState* state,const Model& model,
    int wallCount,cudaStream_t stream,const unsigned char* active=nullptr)
{
    const bool analytic=model.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
    const int workers=analytic?wallCount:model.workspaceCount;
    if(workers<=0)return cudaErrorInvalidValue;
    if(analytic)
        evaluateGasBoundaryLayerKernel<Capture><<<(workers+31)/32,32,0,stream>>>(state,active);
    else if(model.config.model==ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde
        &&model.config.nodes==8&&model.workspaceCapacity==24&&!model.conservative.enabled)
        evaluateGasBoundaryLayerTile8Kernel<Capture><<<(workers+3)/4,32,0,stream>>>(state,active);
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
