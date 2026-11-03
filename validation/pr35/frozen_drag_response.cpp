#include "../../common/gasWall/WallParticleSource.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace ugkwp::gaswall::detail;
int main(){
 struct P{double m,u[3],theta,betaDt,a;};
 const P p[]={ {2.0,{3.0,-1.0,.5},.4,.22314355131420976,-std::expm1(-.22314355131420976)}, {1.5,{-2.0,4.0,-.5},.35,.4307829160924542,-std::expm1(-.4307829160924542)} };
 FrozenDragResponse<double> r;r.gasMass=7.0;
 for(const auto& q:p){r.weightedMass+=q.m*q.a;
  for(int d=0;d<3;++d){r.weightedParticleVelocity[d]+=q.m*q.a*q.u[d];
   r.parcelKineticLinear[d]+=q.m*(1-q.a)*q.a*q.u[d];
   r.parcelKineticConstant+=.5*q.m*((1-q.a)*(1-q.a)-1)*q.u[d]*q.u[d];}
  r.parcelKineticQuadratic+=.5*q.m*q.a*q.a;
  r.unresolvedEnergyChange+=1.5*q.m*q.theta*std::expm1(-2*q.betaDt);
 }
 double ug[3]={-1.5,2.0,.25},sumGasImpulse[3]={},sumParticleDelta[3]={};
 for(int d=0;d<3;++d){const double ustar=frozenDragCommonVelocity(r,d,ug[d]);
  double before=0,after=0;
  for(const auto& q:p){const double unew=(1-q.a)*q.u[d]+q.a*ustar;
   sumParticleDelta[d]+=q.m*(unew-q.u[d]);before+=.5*q.m*q.u[d]*q.u[d];after+=.5*q.m*unew*unew;}
  sumGasImpulse[d]=frozenDragGasImpulse(r,d,ug[d]);
  assert(std::abs(sumGasImpulse[d]+sumParticleDelta[d])<1e-13);
  const double h=1e-6,plus=frozenDragGasImpulse(r,d,ug[d]+h),minus=frozenDragGasImpulse(r,d,ug[d]-h);
  assert(std::abs((plus-minus)/(2*h)-frozenDragGasImpulseDerivative(r))<1e-9);
 }
 const double calculated=frozenDragGasEnergyChange(r,ug);
 double actual=0;
 for(const auto& q:p){double old2=0,new2=0;const double ustar[3]={frozenDragCommonVelocity(r,0,ug[0]),frozenDragCommonVelocity(r,1,ug[1]),frozenDragCommonVelocity(r,2,ug[2])};
  for(int d=0;d<3;++d){const double unew=q.u[d]+q.a*(ustar[d]-q.u[d]);old2+=q.u[d]*q.u[d];new2+=unew*unew;}
  double kinetic=0;for(int d=0;d<3;++d){const double unew=q.u[d]+q.a*(ustar[d]-q.u[d]);kinetic+=.5*(unew-q.u[d])*(unew+q.u[d]);}
  actual+=q.m*(kinetic+1.5*q.theta*std::expm1(-2*q.betaDt));
 }
 assert(std::abs(calculated+actual)<1e-13);
 for(int d=0;d<3;++d){const double h=1e-6,old=ug[d];ug[d]=old+h;const double plus=frozenDragGasEnergyChange(r,ug);ug[d]=old-h;const double minus=frozenDragGasEnergyChange(r,ug);ug[d]=old;
  assert(std::abs((plus-minus)/(2*h)-frozenDragGasEnergyDerivative(r,d,ug))<1e-8);}
 FrozenDragResponse<double> zero;zero.gasMass=7;
 assert(frozenDragGasImpulse(zero,0,ug[0])==0&&frozenDragGasImpulseDerivative(zero)==0);
 assert(frozenDragGasEnergyChange(zero,ug)==0&&frozenDragGasEnergyDerivative(zero,1,ug)==0);
 std::cout<<"frozen response conservation, energy, derivatives, zero-source: PASS\n";
}
