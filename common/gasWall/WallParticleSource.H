#pragma once
#include "WallModelTypes.H"
#include "../GpuMomentDragRelaxation.H"
#include "../gpu/GpuCouplingMath.H"
#if defined(__CUDA_ARCH__)
#define UGKWP_WALL_PARTICLE_LEAF __noinline__
#else
#define UGKWP_WALL_PARTICLE_LEAF inline
#endif
namespace ugkwp { namespace gaswall { namespace detail {
// Read-only particle moments and frozen relaxation coefficient for an ODE
// dual control volume. Production evaluation shares the macro moment operator;
// it never changes parcel or gas state.
template<class R> struct NodeParticleMomentResponse {
 R weightedMass=0,weightedParticleVelocity[3]{},parcelKineticConstant=0;
 R parcelKineticLinear[3]{},parcelKineticQuadratic=0,unresolvedEnergyChange=0;
 R heatCapacityFactor=0,particleTemperature=0,heatMass=0,heatEnthalpy=0,heatDiameter=0;
 R heatGasCapacity=0,heatSolidCapacity=0,heatRate=0,heatDt=0;
 R heatMomentum[3]{};
 R rawMass=0,rawParticleEnergy=0,weightedBetaDt=0,gasMass=0,gasCv=0,gasTemperatureMin=0;
 R rawMomentum[3]{};
 UGKWP_GAS_HD inline void addRaw(R mass,const R* velocity,R theta,R betaDt){
  rawMass+=mass;weightedBetaDt+=mass*betaDt;
  rawParticleEnergy+=R(.5)*mass*(velocity[0]*velocity[0]+velocity[1]*velocity[1]+velocity[2]*velocity[2])+R(1.5)*mass*theta;
  for(int d=0;d<3;++d)rawMomentum[d]+=mass*velocity[d];
 }
 UGKWP_GAS_HD inline void add(R mass,const R* velocity,R theta,R betaDt){
  addRaw(mass,velocity,theta,betaDt);
  const R a=-::expm1(-betaDt),remain=R(1)-a;
  weightedMass+=mass*a;
  for(int d=0;d<3;++d){
   weightedParticleVelocity[d]+=mass*a*velocity[d];
   parcelKineticLinear[d]+=mass*remain*a*velocity[d];
   parcelKineticConstant+=R(.5)*mass*(remain*remain-R(1))*velocity[d]*velocity[d];
  }
  parcelKineticQuadratic+=R(.5)*mass*a*a;
  unresolvedEnergyChange+=R(1.5)*mass*theta*::expm1(-R(2)*betaDt);
 }
};
template<class R> UGKWP_GAS_HD inline MomentDragRelaxationResult<R> nodeMomentGasState(
 const NodeParticleMomentResponse<R>& r,const R* gasVelocity,R gasTemperature){
 const R gasEnergy=r.gasMass*r.gasCv*gasTemperature+R(.5)*r.gasMass*(gasVelocity[0]*gasVelocity[0]+gasVelocity[1]*gasVelocity[1]+gasVelocity[2]*gasVelocity[2]);
 return relaxGasFromParticleMoments(r.gasMass,r.rawMass,R(1),R(1),gasEnergy,
   r.gasMass*r.gasCv*r.gasTemperatureMin,gasVelocity,r.rawMomentum,r.rawParticleEnergy,
   r.weightedBetaDt/(r.rawMass+R(1e-300)));
}
template<class R> UGKWP_GAS_HD inline R nodeMomentGasImpulse(
 const NodeParticleMomentResponse<R>& r,int d,R gasVelocity){
 return r.weightedParticleVelocity[d]-r.weightedMass*gasVelocity;
}
template<class R> UGKWP_GAS_HD inline R nodeMomentGasEnergySource(
 const NodeParticleMomentResponse<R>& r,const R* gasVelocity){
 R particleEnergy=r.parcelKineticConstant+r.unresolvedEnergyChange;
 for(int d=0;d<3;++d)
  particleEnergy+=r.parcelKineticLinear[d]*gasVelocity[d]
                +r.parcelKineticQuadratic*gasVelocity[d]*gasVelocity[d];
 return -particleEnergy;
}
template<class R> UGKWP_GAS_HD inline R nodeMomentGasEnergyDerivative(
 const NodeParticleMomentResponse<R>& r,int d,R gasVelocity){
 return -(r.parcelKineticLinear[d]+R(2)*r.parcelKineticQuadratic*gasVelocity);
}
template<class R> UGKWP_GAS_HD inline R nodeMomentGasHeatSource(
 const NodeParticleMomentResponse<R>& r,R gasTemperature){
 return -gpuFiniteCapacityHeatExchangeCore(r.heatGasCapacity,r.heatSolidCapacity,
   gasTemperature,r.particleTemperature,r.heatRate,r.heatDt);
}
template<class R> struct NodeParticleSourceResponse {
 NodeParticleMomentResponse<R> moments{};R sourceScale=0;
};
// Legacy finite-bath algebra retained for its isolated unit regression; production
// wall closure uses NodeParticleSourceResponse.
template<class R> struct FrozenDragResponse {
 R gasMass=0,weightedMass=0,weightedParticleVelocity[3]{},sourceScale=0,parcelKineticConstant=0;
 R parcelKineticLinear[3]{},parcelKineticQuadratic=0,unresolvedEnergyChange=0;
};
template<class R> UGKWP_GAS_HD inline R frozenDragGasImpulse(
 const FrozenDragResponse<R>& r,int d,R gasVelocity){
 return r.gasMass*(r.weightedParticleVelocity[d]-r.weightedMass*gasVelocity)
       /(r.gasMass+r.weightedMass);
}
template<class R> UGKWP_GAS_HD inline R frozenDragGasImpulseDerivative(
 const FrozenDragResponse<R>& r){
 return -r.gasMass*r.weightedMass/(r.gasMass+r.weightedMass);
}
template<class R> UGKWP_GAS_HD inline R frozenDragCommonVelocity(
 const FrozenDragResponse<R>& r,int d,R gasVelocity){
 return (r.gasMass*gasVelocity+r.weightedParticleVelocity[d])
       /(r.gasMass+r.weightedMass);
}
template<class R> UGKWP_GAS_HD inline R frozenDragGasEnergyChange(
 const FrozenDragResponse<R>& r,const R* gasVelocity){
 R particleEnergy=r.parcelKineticConstant+r.unresolvedEnergyChange;
 for(int d=0;d<3;++d){const R target=frozenDragCommonVelocity(r,d,gasVelocity[d]);
  particleEnergy+=r.parcelKineticLinear[d]*target;
  particleEnergy+=r.parcelKineticQuadratic*target*target;
 }
 return -particleEnergy;
}
template<class R> UGKWP_GAS_HD inline R frozenDragGasEnergyDerivative(
 const FrozenDragResponse<R>& r,int d,const R* gasVelocity){
 const R target=frozenDragCommonVelocity(r,d,gasVelocity[d]);
 const R dTarget=r.gasMass/(r.gasMass+r.weightedMass);
 return -(r.parcelKineticLinear[d]+R(2)*r.parcelKineticQuadratic*target)*dTarget;
}
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
