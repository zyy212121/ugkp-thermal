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
// Values decoded directly from the solved workspace, in mechanism species order.
template<class R,int N> struct WallProfileNode {
    R y=0,velocity[3]{},temperature=0,massFraction[N]{},k=0,omega=0;
};
template<class R,int N,int M> UGKWP_GAS_HD inline bool reactingWallProfileNode(
 const WallInput<R,N>& input,const WallModelConfig<R>& config,
 const WallWorkspace<R,N,M>& workspace,int node,const detail::LayerContext<R,N>& context,
 WallProfileNode<R,N>& result){
    detail::LayerPoint<R,N> point;
    if(!detail::pointState(input,config,context,workspace.state[node],point,true))return false;
    result.y=workspace.y[node];result.temperature=point.T;
    for(int species=0;species<N;++species)result.massFraction[species]=point.Y[species];
    for(int d=0;d<3;++d)result.velocity[d]=input.velocity[d]
        +context.tangent[0][d]*point.u[0]+context.tangent[1][d]*point.u[1]
        +input.normal[d]*(point.normalVelocity-context.wallNormal);
    result.k=point.k;
    // omega is singular at a no-injection wall (the solved inverse root is 0).
    result.omega=config.enableSst?(point.z==0?std::numeric_limits<R>::infinity():point.omega):R(0);
    return true;
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool reactingWallProfileNode(
 const WallInput<R,N>& input,const WallModelConfig<R>& config,
 const WallWorkspace<R,N,M>& workspace,int node,WallProfileNode<R,N>& result){
    detail::LayerContext<R,N> context;
    return detail::makeContext(input,config,context)
        && reactingWallProfileNode(input,config,workspace,node,context,result);
}
template<class R,int N,int M> UGKWP_GAS_HD inline bool evaluateWallModel(
 const WallInput<R,N>& input,const WallModelConfig<R>& config,WallWorkspace<R,N,M>& workspace,
 WallOutput<R,N>& output,WallStatus& status){
    if(config.model==BoundaryLayerModel::ConstantTransport)return evaluateConstantTransportWall(input,config,output,status);
    if(config.model==BoundaryLayerModel::ReactingSst)return evaluateReactingWallLayer(input,config,workspace,output,status);
    return detail::fail(status,WallCode::Unsupported);
}
}}
