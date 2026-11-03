#include "../../common/gasWall/WallParticleSource.H"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace ugkwp::gaswall::detail;
int main(){
 struct P{double m,u[3],theta,d;};
 const P p[]={{2.0,{3.0,-1.0,.5},.4,2.0e-4},{1.5,{-2.0,4.0,-.5},.35,3.0e-4}};
 NodeParticleMomentResponse<double> r;
 double mass=0,diameterMoment=0,momentum[3]={},energy=0;
 for(const auto& q:p){r.addRaw(q.m,q.u,q.theta,q.d);mass+=q.m;diameterMoment+=q.m*q.d;
  energy+=.5*q.m*(q.u[0]*q.u[0]+q.u[1]*q.u[1]+q.u[2]*q.u[2])+1.5*q.m*q.theta;
  for(int d=0;d<3;++d)momentum[d]+=q.m*q.u[d];}
 assert(std::abs(r.rawMass-mass)<1e-14);assert(std::abs(r.rawDiameterMoment-diameterMoment)<1e-14);
 assert(std::abs(r.rawParticleEnergy-energy)<1e-13);for(int d=0;d<3;++d)assert(std::abs(r.rawMomentum[d]-momentum[d])<1e-14);
 const double dMean=r.rawDiameterMoment/r.rawMass;assert(std::abs(dMean-(2.0*2e-4+1.5*3e-4)/3.5)<1e-16);
 const double ug[3]={-1.5,2.0,.25};const double us[3]={momentum[0]/mass,momentum[1]/mass,momentum[2]/mass};
 const double slip=std::sqrt((ug[0]-us[0])*(ug[0]-us[0])+(ug[1]-us[1])*(ug[1]-us[1])+(ug[2]-us[2])*(ug[2]-us[2]));
 const double beta=2.0+1000.0*dMean+.01*slip;const double betaDt=.03*beta;r.setUniformBetaDt(betaDt);
 double impulse[3]={};
 for(int d=0;d<3;++d){for(const auto& q:p){const double a=-std::expm1(-betaDt);impulse[d]+=q.m*a*(q.u[d]-ug[d]);}
  assert(std::abs(impulse[d]-nodeMomentGasImpulse(r,d,ug[d]))<1e-13);}
 double expected=0;for(const auto& q:p){const double a=-std::expm1(-betaDt);double dk=0;
  for(int d=0;d<3;++d){const double un=q.u[d]+a*(ug[d]-q.u[d]);dk+=.5*q.m*(un*un-q.u[d]*q.u[d]);}
  expected-=dk+1.5*q.m*q.theta*std::expm1(-2*betaDt);}
 assert(std::abs(expected-nodeMomentGasEnergySource(r,ug))<1e-13);
 for(int d=0;d<3;++d){double plus[3]={ug[0],ug[1],ug[2]},minus[3]={ug[0],ug[1],ug[2]};const double h=1e-6;plus[d]+=h;minus[d]-=h;
  const double fd=(nodeMomentGasEnergySource(r,plus)-nodeMomentGasEnergySource(r,minus))/(2*h);
  const double a=-std::expm1(-betaDt),remain=1-a;const double analytic=-(remain*a*momentum[d]+mass*a*a*ug[d]);
  assert(std::abs(fd-analytic)<1e-8);}
 NodeParticleMomentResponse<double> zero;zero.setUniformBetaDt(betaDt);assert(nodeMomentGasImpulse(zero,0,ug[0])==0&&nodeMomentGasEnergySource(zero,ug)==0);
 r.gasMass=4.0;r.gasCv=700.0;r.gasTemperatureMin=200.0;const auto shared=nodeMomentGasState(r,ug,350.0);
 const double mg=4.0,ms=r.rawMass,us0[3]={r.rawMomentum[0]/ms,r.rawMomentum[1]/ms,r.rawMomentum[2]/ms};
 const double kg0=.5*mg*(ug[0]*ug[0]+ug[1]*ug[1]+ug[2]*ug[2]),ks0=.5*ms*(us0[0]*us0[0]+us0[1]*us0[1]+us0[2]*us0[2]);
 const double gasE=mg*700.0*350.0+kg0,ig0=gasE-kg0,is0=r.rawParticleEnergy-ks0,alphaI=std::exp(-2*betaDt),alphaW=std::exp(-betaDt*(1+ms/mg));
 double w[3],us1[3],ug1[3];for(int d=0;d<3;++d)w[d]=(ug[d]-us0[d])*alphaW;
 for(int d=0;d<3;++d){const double totalP=mg*ug[d]+r.rawMomentum[d];us1[d]=(totalP-mg*w[d])/(mg+ms);ug1[d]=us1[d]+w[d];assert(std::abs(shared.velocity[d]-ug1[d])<1e-12);}
 const double kg1=.5*mg*(ug1[0]*ug1[0]+ug1[1]*ug1[1]+ug1[2]*ug1[2]),ks1=.5*ms*(us1[0]*us1[0]+us1[1]*us1[1]+us1[2]*us1[2]);
 const double expectedGasEnergy=kg1+ig0+(is0-is0*alphaI)+std::max(0.0,kg0+ks0-kg1-ks1);
 assert(std::abs(shared.intrinsicGasEnergy-expectedGasEnergy)<1e-9);
 r.heatGasCapacity=10.0;r.heatSolidCapacity=12.0;r.heatRate=.4;r.heatDt=.02;r.particleTemperature=300.0;
 assert(std::abs(nodeMomentGasHeatSource(r,350.0)+gpuFiniteCapacityHeatExchangeCore(10.0,12.0,350.0,300.0,.4,.02))<1e-12);
 std::cout<<"single-beta node M/P/E/diameter moments, shared Eulerian relaxation, energy derivative and heat law: PASS\n";
}
