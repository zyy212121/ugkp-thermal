#pragma once
#include "ConstantTransportWall.H"
#include "ReactingWallLayer.H"
#include "WallPrecision.H"
namespace ugkwp { namespace gaswall {
// Valid after a successful ReactingSst solve, for 0 <= node < config.nodes.
// Workspace temperature is compensated relative to the prescribed wall value.
template<class R,int N,int M> UGKWP_GAS_HD inline R reactingWallProfileTemperature(
 const WallInput<R,N>& input,const WallWorkspace<R,N,M>& workspace,int node){
    return input.temperature+workspace.state[node][2];
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool evaluateWallModel(
 const WallInput<R,N>& input,const WallModelConfig<R>& config,WallWorkspace<R,N,M>& workspace,
 WallOutput<R,N>& output,WallStatus& status){
    if(config.model==BoundaryLayerModel::ConstantTransport)return evaluateConstantTransportWall(input,config,output,status);
    if(config.model==BoundaryLayerModel::ReactingSst)return evaluateReactingWallLayer(input,config,workspace,output,status);
    return detail::fail(status,WallCode::Unsupported);
}
}}
