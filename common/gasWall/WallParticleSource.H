#pragma once
#include "WallModelTypes.H"
#if defined(__CUDA_ARCH__)
#define UGKWP_WALL_PARTICLE_LEAF __noinline__
#else
#define UGKWP_WALL_PARTICLE_LEAF inline
#endif
namespace ugkwp { namespace gaswall { namespace detail {
// The source has already been exchanged with the macro gas inventory. Here it
// enters only the subgrid normal flux divergence, never a second volume update.
template<class R,int N,class T> UGKWP_GAS_HD inline R particleMomentumSource(
 const WallInput<R,N,T>& in,int node,const R* direction){
 R value=0;if(in.particleSource)for(int d=0;d<3;++d)value+=R(in.particleSource[4*node+d])*direction[d];return value;
}
template<class R,int N,class T,class Context> UGKWP_GAS_HD inline R particleEnergySource(
 const WallInput<R,N,T>& in,const Context& ctx,int node){
 R value=in.particleSource?R(in.particleSource[4*node+3]):R(0);
 for(int d=0;d<2;++d)value-=ctx.wallU[d]*particleMomentumSource(in,node,ctx.tangent[d]);
 return value;
}
// Piecewise uniform source within each node's geometric dual CV. This is the
// integral from the wall to y, used by the local-stress constitutive closure.
template<class R,int N,class T> UGKWP_GAS_HD UGKWP_WALL_PARTICLE_LEAF R particleMomentumPrefix(
 const WallInput<R,N,T>& in,const WallModelConfig<R>& c,R y,const R* direction){
 R value=0;if(!in.particleSource)return value;
 R previous=0,current=0;
 for(int i=0;i<c.nodes;++i){
  const R next=i+1<c.nodes?in.matchingDistance*::pow(R(i+1)/R(c.nodes-1),c.stretch):in.matchingDistance;
  const R lo=i?(previous+current)/R(2):R(0),hi=i+1<c.nodes?(current+next)/R(2):current;
  const R fraction=y>=hi?R(1):(y<=lo?R(0):(y-lo)/(hi-lo));
  value+=fraction*particleMomentumSource(in,i,direction);previous=current;current=next;
 }return value;
}
}}}

#undef UGKWP_WALL_PARTICLE_LEAF
