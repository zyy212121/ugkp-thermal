#include "tests/TestSupport.H"
#include "particles/ParticleMath.H"
using namespace chmt;using namespace chmt_test;
int main(){
 Vec3 out,impulse;Real work;check(elasticWallRebound({2,-3,0},{.5,0,0},{1,0,0},4,out,impulse,work),"moving elastic rebound");near(out.x,-1,"wall-frame reflected normal");near(out.y,-3,"tangent retained");near(work,.5*4*(dot(out,out)-13),"mechanical support work");
 Real t=0;Vec3 n;check(movingTriangleIntersection({.2,.2,1},{.2,.2,-1},{0,0,0},{1,0,0},{0,1,0},{0,0,.5},{1,0,.5},{0,1,.5},t,n),"moving triangle collision");near(t,.4,"moving triangle crossing analytic fraction");check(!movingTriangleIntersection({2,2,1},{2,2,-1},{0,0,0},{1,0,0},{0,1,0},{0,0,0},{1,0,0},{0,1,0},t,n),"outside triangle misses");
 check(particleStageEventNeedsSplit(1,1,0)&&!particleStageEventNeedsSplit(1,1,1),"predictor endpoint splits full step");
 const Real Cg=5,Cp=2,G=3,dt=.2;const Real exact=(700-300)/(1/Cg+1/Cp)*(1-std::exp(-G*(1/Cg+1/Cp)*dt));near(finiteCapacityHeatReference(Cg,Cp,700,300,G,dt),exact,"finite-capacity analytic heat");
 auto p=physics();p.gravity={0,-9.8,0};ParticleQ base;base.mass=2;base.cell=0;base.velocity={2,3,0};base.energy=1000;ParticleQ mid=base;const Real step=.1;Vec3 force={-1,2,0};mid.velocity=base.velocity+(force/base.mass+p.gravity)*(.5*step);ParticleQ next;ExchangePacket packet;
 check(makeParticleGasExchange(base,mid,{0,0,0},500,2,p,step,force,7,next,packet),"midpoint particle exchange");near(particleTotalEnergy(next)-particleTotalEnergy(base)+packet.energy,base.mass*dot(p.gravity,next.position-base.position),"gas-particle energy plus external gravity work");near(mag((next.velocity-base.velocity)*base.mass+packet.momentum-p.gravity*(base.mass*step)),0,"gas-particle momentum plus gravity",1e-11,1e-12);
 std::cout<<"particle host regressions: wall/event/heat/exchange mathematics passed; no device tracking\n";
}
