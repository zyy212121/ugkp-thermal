#pragma once
#include "gasTransport/GasBoundaryLayerModelState.H"
// One sparse worker owns one bounded scratch slot and walks distinct faces.
// Expensive profile scratch scales with workspaceCount, not wall-face count.
template<bool Capture=false,class GasState>
__global__ void evaluateGasBoundaryLayerKernel(GasState* state)
{
    if constexpr(ugkwp::GasBoundaryLayerModelCapability<GasState>::value)
    {
        const int worker=blockIdx.x*blockDim.x+threadIdx.x;
        const bool analytic=state->gasBoundaryLayerModel.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
        const int workers=analytic?state->gasBoundaryLayer.count:state->gasBoundaryLayerModel.workspaceCount;
        if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
        for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
            ugkwp::evaluateGasBoundaryLayerSlot<Capture>(*state,slot,worker);
    }
}

#if defined(__CUDACC__)
#include "gasWall/WallCudaTeam.cuh"
// Full CTA entry/exit is uniform even when nodes<32 or an evaluation fails.
template<bool Capture,int Capacity,class GasState>
__device__ inline void evaluateGasBoundaryLayerTeamWorker(GasState*,
    std::false_type){}
template<bool Capture,int Capacity,class GasState>
__device__ inline void evaluateGasBoundaryLayerTeamWorker(GasState* state,
    std::true_type)
{
    const int worker=blockIdx.x,workers=state->gasBoundaryLayerModel.workspaceCount;
    if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
    const ugkwp::gaswall::detail::CudaWallTeam team{};
    for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
        ugkwp::evaluateGasBoundaryLayerSlot<Capture,Capacity>(*state,slot,worker,team);
}
// Node-parallel nonlinear BVP; scalar/analytic kernel remains available above.
template<bool Capture=false,int Capacity=0,class GasState>
__global__ __launch_bounds__(ugkwp::gaswall::detail::wallWarpWidth) void evaluateGasBoundaryLayerTeamKernel(GasState* state)
{
    evaluateGasBoundaryLayerTeamWorker<Capture,Capacity>(state,
        typename ugkwp::GasBoundaryLayerModelCapability<GasState>::type{});
}
#endif
