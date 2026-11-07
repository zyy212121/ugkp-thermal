#pragma once
#include "../Diagnostics.H"
#include "WallModelTypes.H"
#include "WallGeometryTypes.H"
namespace ugkwp { namespace gaswall {
// A cell-average matching stencil has one donor per wall. Its offsets, wall
// identities and allocations are installed once by the application adapter.
// Reconstructed FV stencils use their separate projection operator.
template<class R,int Ns>
__global__ void publishCellAverageWallGeometryKernel(int count,const WallMatchingGeometry* geometry,
    const WallMatchingStatus* status,WallInput<R,Ns>* input,int* matchingCells,R* matchingWeights,
    ugkp::diagnostics::DeviceContext* diagnostics)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(!ugkp::diagnostics::blockMayProceed(diagnostics))return;
    if(row>=count||status->invalidFace>=0||status->firstInvalidRow<count)return;
    const auto& g=geometry[row];WallInput<R,Ns> next{};
    next.matchingDistance=g.matchingDistance;next.ownerDistance=g.ownerDistance;
    for(int k=0;k<3;++k)next.normal[k]=g.normal[k];
    // Production cell-average geometry does not construct optional quadrature
    // nodes, but retains its exact owner volume and first normal moment.
    next.quadrature.volume=g.volume;next.quadrature.firstMoment=g.firstMoment;
    next.diagnostics=diagnostics;
    input[row]=next;matchingCells[row]=g.donor;matchingWeights[row]=R(1);
}
// Invalidate only geometry-dependent frames and exchanges. The node history is
// physical state and survives coordinate-only changes, as in host publication.
template<class T> __device__ void clearWallGeometryValue(T* value){
    if(value){auto* bytes=reinterpret_cast<unsigned char*>(value);
        for(unsigned int k=0;k<sizeof(T);++k)bytes[k]=0;}
}
template<class State>
__global__ void resetWallGeometryFramesKernel(State* state,const WallMatchingStatus* status){
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    const int count=state->gasBoundaryLayer.count,row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=count||status->invalidFace>=0||status->firstInvalidRow<count)return;
    auto& wall=state->gasBoundaryLayer;auto& layer=state->gasBoundaryLayerModel;
    if(layer.acceptedFrame)clearWallGeometryValue(layer.acceptedFrame+row);
    if(layer.candidateFrame)clearWallGeometryValue(layer.candidateFrame+row);
    clearWallGeometryValue(wall.exchange+row);clearWallGeometryValue(wall.status+row);
    clearWallGeometryValue(layer.status+row);
}
}}
