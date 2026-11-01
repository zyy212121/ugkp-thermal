"""Independent extensive inventories for each spatial gas reservoir."""
from test_mixture_state import compile_probe

def test_spatial_reservoirs_conserve_each_exchange(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/ParticleReservoirRelaxation.H"
#include <cassert>
int main(){
 const double nodes[]={0,.01,.04,.1,.2,.4,.7,1};double sum=0;
 for(int i=0;i<8;++i)sum+=ugkwp::particleNormalReservoirWidth(nodes,8,i);
 assert(std::abs(sum-1)<1e-15);
 assert(ugkwp::particleNormalReservoir(nodes,8,.003)==0);
 assert(ugkwp::particleNormalReservoir(nodes,8,.5)==5);
 assert(ugkwp::particleNormalReservoir(nodes,8,1.)==7);
 // Different gas velocities must cause different local relaxations.
 for(int node=0;node<8;++node){
  ugkwp::ParticleGasReservoir<double> gas;gas.mass=2;
  gas.momentum[0]=2.*node;gas.energy=10000.+node*node;
  double v[2][3]={{-1,2,3},{3,-2,1}},theta[2]={2,4};
  const double mass[2]={.1,.3},bdt[2]={.01,2.};
  ugkwp::ParticleReservoirTarget<double> target;
  double initialMomentum[3]={gas.momentum[0],gas.momentum[1],gas.momentum[2]};double initialEnergy=gas.energy;
  for(int i=0;i<2;++i){target.add(mass[i],-std::expm1(-bdt[i]),v[i]);
   for(int d=0;d<3;++d)initialMomentum[d]+=mass[i]*v[i][d];
   initialEnergy+=mass[i]*(.5*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2])+1.5*theta[i]);}
  double u[3];target.velocity(gas,u);
  for(int i=0;i<2;++i){const auto change=ugkwp::relaxParticleMechanical(mass[i],-std::expm1(-bdt[i]),bdt[i],u,v[i],theta[i]);ugkwp::settleParticleReservoir(gas,change);}
  double finalEnergy=gas.energy,finalMomentum[3]={gas.momentum[0],gas.momentum[1],gas.momentum[2]};
  for(int i=0;i<2;++i){for(int d=0;d<3;++d)finalMomentum[d]+=mass[i]*v[i][d];
   finalEnergy+=mass[i]*(.5*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2])+1.5*theta[i]);}
  assert(std::abs(finalEnergy-initialEnergy)<4e-12);
  for(int d=0;d<3;++d)assert(std::abs(finalMomentum[d]-initialMomentum[d])<4e-15);
  assert(std::abs(gas.momentum[0]/gas.mass-u[0])<4e-15);
 }
}
''')
