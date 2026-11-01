#pragma once
#include "MixtureThermo.H"
namespace ugkwp {
// One gas reservoir can represent a macro cell or a wall dual control volume.
// Physical masses/energies are extensive; no application or case identity enters.
template<class R> struct ParticleGasReservoir {
 R mass=0,momentum[3]{},energy=0;
};
template<class R> struct ParticleReservoirTarget {
 R weight=0,weightedVelocity[3]{};
 UGKWP_GAS_HD void add(R mass,R fraction,const R* velocity){
  const R w=mass*fraction;weight+=w;
  for(int d=0;d<3;++d)weightedVelocity[d]+=w*velocity[d];
 }
 UGKWP_GAS_HD void velocity(const ParticleGasReservoir<R>& gas,R* target)const{
  for(int d=0;d<3;++d)target[d]=(gas.momentum[d]+weightedVelocity[d])/(gas.mass+weight);
 }
};
template<class R> struct ParticleReservoirExchange {R momentum[3]{},energy=0;};
// Same finite-bath/exponential design as the existing cell exchange. Actual
// represented particle increments determine the reverse gas transaction.
template<class R> UGKWP_GAS_HD inline ParticleReservoirExchange<R>
relaxParticleMechanical(R mass,R fraction,R betaDt,const R* target,R* velocity,R& theta){
 ParticleReservoirExchange<R> change;
 for(int d=0;d<3;++d){
  const R old=velocity[d];velocity[d]=old+fraction*(target[d]-old);
  const R delta=velocity[d]-old;
  change.momentum[d]=mass*delta;
  change.energy+=mass*R(.5)*delta*(velocity[d]+old);
 }
 const R oldTheta=theta;theta=oldTheta*::exp(-R(2)*betaDt);
 change.energy+=mass*R(1.5)*(theta-oldTheta);
 return change;
}
template<class R> UGKWP_GAS_HD inline void settleParticleReservoir(
 ParticleGasReservoir<R>& gas,const ParticleReservoirExchange<R>& change){
 for(int d=0;d<3;++d)gas.momentum[d]-=change.momentum[d];
 gas.energy-=change.energy;
}
// Existing wall nodes include the two boundaries. These geometric dual CVs
// partition [y0,yN] with endpoint half volumes. Boundaries retain their own
// flux constraints; the returned index does not authorize overwriting them.
template<class R> UGKWP_GAS_HD inline int particleNormalReservoir(const R* nodes,int count,R height){
 int node=0;while(node+1<count&&height>(nodes[node]+nodes[node+1])/R(2))++node;
 return node;
}
template<class R> UGKWP_GAS_HD inline R particleNormalReservoirWidth(const R* nodes,int count,int node){
 const R lo=node?(nodes[node-1]+nodes[node])/R(2):nodes[0];
 const R hi=node+1<count?(nodes[node]+nodes[node+1])/R(2):nodes[count-1];
 return hi-lo;
}
}
