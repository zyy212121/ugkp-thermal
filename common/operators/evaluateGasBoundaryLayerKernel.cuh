#pragma once
#include "gasTransport/GasBoundaryLayerModelState.H"
// One sparse worker owns one bounded scratch slot and walks distinct faces.
// Expensive profile scratch scales with workspaceCount, not wall-face count.
template<class GasState>
__global__ void evaluateGasBoundaryLayerKernel(GasState* state)
{
    if constexpr(ugkwp::GasBoundaryLayerModelCapability<GasState>::value)
    {
        const int worker=blockIdx.x*blockDim.x+threadIdx.x;
        const bool analytic=state->gasBoundaryLayerModel.config.model==ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
        const int workers=analytic?state->gasBoundaryLayer.count:state->gasBoundaryLayerModel.workspaceCount;
        if(worker>=workers || !ugkwp::gasBoundaryLayerEnabled(*state))return;
        for(int slot=worker;slot<state->gasBoundaryLayer.count;slot+=workers)
            ugkwp::evaluateGasBoundaryLayerSlot(*state,slot,worker);
    }
}
