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


def test_node_inventory_reconstruction_preserves_macro_moments(tmp_path):
    compile_probe(tmp_path,r'''
#include "gasTransport/ParticleReservoirReconstruction.H"
#include <cassert>
int main(){
 ugkwp::SpeciesThermoData<double> species[2];double coefficients[]={1000,0,-1e6,1100,0,2e6};
 for(int j=0;j<2;++j){species[j].molarMass=.028;species[j].minTemperature=100;species[j].maxTemperature=3000;species[j].coefficientOffset=3*j;}
 ugkwp::SpeciesThermoView<double,2> thermo;thermo.species=species;thermo.coefficients=coefficients;thermo.coefficientCount=6;
 ugkwp::ThermoInversionControls<double> controls;
 const double mass[2]={.8,1.2};ugkwp::ParticleGasReservoir<double> macro;macro.mass=2;macro.momentum[0]=20;macro.energy=ugkwp::mixtureEnergy(mass,500.,thermo)+100.;
 ugkwp::ParticleReservoirShape<2,double> shape[8];ugkwp::ParticleThermalReservoir<2,double> out[8];
 for(int i=0;i<8;++i){auto&q=shape[i];q.volumeFraction=.125;q.density=.8+.1*i;q.temperature=300+100*i;q.velocity[0]=100*i;q.massFraction[0]=.1+.1*i;q.massFraction[1]=1-q.massFraction[0];}
 assert(ugkwp::reconstructParticleReservoirs(macro,mass,1.,500.,shape,8,thermo,controls,out));
 double totalMass=0,energy=0,momentum[3]{},component[2]{};
 for(const auto&q:out){totalMass+=q.mass;energy+=q.energy;for(int d=0;d<3;++d)momentum[d]+=q.momentum[d];for(int j=0;j<2;++j)component[j]+=q.speciesMass[j];assert(q.temperature>=100&&q.temperature<=3000);}
 assert(std::abs(totalMass-2)<1e-14);assert(std::abs(energy-macro.energy)<1e-8);
 for(int d=0;d<3;++d)assert(std::abs(momentum[d]-macro.momentum[d])<1e-12);
 for(int j=0;j<2;++j)assert(std::abs(component[j]-mass[j])<1e-14);
 assert(out[0].temperature!=out[7].temperature);
}
''')


def test_same_exchange_kernel_updates_independent_node_reservoirs(tmp_path):
    from test_steady_mixture_particle_coupling import paired_fixture
    body=paired_fixture()+r'''
#include <cassert>
struct SpatialState:PairedState {
 ugkwp::MolecularViscosityView<double> gasViscosity;
 double heights[4]={.01,.99,.5,.5};
};
struct NodeSelection {
 int node;
 template<class S> bool operator()(const S&s,int i)const{
  const double coordinates[2]={0,1};
  return ugkwp::particleNormalReservoir(coordinates,2,s.heights[i])==node;
 }
};
int main(){
 for(bool heat:{false,true}){
  SpatialState s;setup(s);s.solveParticleTemperature=heat;s.particleGasHeatTransferModelId=heat?1:0;
  const double volume=s.epsGPrev[0]*s.V[0];
  ugkwp::ParticleGasReservoir<double> macro;macro.mass=volume*s.rho[0];macro.energy=volume*s.rhoE[0];
  macro.momentum[0]=volume*s.rhoUx[0];macro.momentum[1]=volume*s.rhoUy[0];macro.momentum[2]=volume*s.rhoUz[0];
  double species[2]={volume*s.gasSpecies.rho[0],volume*s.gasSpecies.rho[2]};
  ugkwp::ParticleReservoirShape<2,double> shape[2];
  for(int i=0;i<2;++i){shape[i].volumeFraction=.5;shape[i].density=s.rho[0];shape[i].temperature=i?1000:400;
   shape[i].velocity[0]=i?20:-10;shape[i].velocity[1]=-2;shape[i].velocity[2]=1;
   for(int j=0;j<2;++j)shape[i].massFraction[j]=species[j]/macro.mass;}
  ugkwp::ParticleThermalReservoir<2,double> nodes[2];
  assert(ugkwp::reconstructParticleReservoirs(macro,species,volume,s.Tgas[0],shape,2,s.gasSpecies.thermo,s.gasSpecies.thermoControls,nodes));
  double initial[4]={macro.momentum[0],macro.momentum[1],macro.momentum[2],macro.energy};
  const double oldT[2]={s.pT[0],s.pT[1]};
  for(int i=0;i<2;++i){initial[0]+=s.pm[i]*s.pux[i];initial[1]+=s.pm[i]*s.puy[i];initial[2]+=s.pm[i]*s.puz[i];initial[3]+=s.pm[i]*(.5*(s.pux[i]*s.pux[i]+s.puy[i]*s.puy[i]+s.puz[i]*s.puz[i])+1.5*s.pTheta[i]+800*s.pT[i]);}
  for(int node=0;node<2;++node){
   NodeSelection select{node};mixtureParticles::NodeReservoir<SpatialState,2,NodeSelection> reservoir{nodes[node],select};
   mixtureParticleCouplingReservoir<ugkwpGpuDrag::SchillerNaumannDeviceDrag,true>(s,0,.001,true,true,false,mixtureParticles::SerialCouplingTeam{},reservoir);
   assert(s.gasSpecies.cellStatus[0]==0);
  }
  double final[4]={};
  for(int i=0;i<2;++i){for(int d=0;d<3;++d)final[d]+=nodes[i].momentum[d];final[3]+=nodes[i].energy;}
  for(int i=0;i<2;++i){final[0]+=s.pm[i]*s.pux[i];final[1]+=s.pm[i]*s.puy[i];final[2]+=s.pm[i]*s.puz[i];final[3]+=s.pm[i]*(.5*(s.pux[i]*s.pux[i]+s.puy[i]*s.puy[i]+s.puz[i]*s.puz[i])+1.5*s.pTheta[i]+800*s.pT[i]);}
  for(int d=0;d<3;++d)assert(std::abs(final[d]-initial[d])<1e-12);
  assert(std::abs(final[3]-initial[3])<5e-8);
  if(!heat)assert(s.pT[0]==oldT[0]&&s.pT[1]==oldT[1]);
  else assert(s.pT[0]!=oldT[0]&&s.pT[1]!=oldT[1]);
 }
}
'''
    compile_probe(tmp_path,body)


def test_production_wall_router_uses_position_and_counts_exchange_once(tmp_path):
    from test_steady_mixture_particle_coupling import paired_fixture
    compile_probe(tmp_path,paired_fixture()+r'''
#include <cassert>
struct WallSpatialState:PairedState {
 ugkwp::MolecularViscosityView<double> gasViscosity;
 double px[4]{},py[4]={.001,.69,0,0},pz[4]{};
 double faceCx[1]{},faceCy[1]{},faceCz[1]{},magSf[1]={1};
 int ownerMap[2]={0,-1},faceMap[1]={0},faces[1]={0};
 struct Wall {bool enabled=true;int*ownerSlot=nullptr,*faceSlot=nullptr;} gasBoundaryLayer;
 struct Model {
  ugkwp::gaswall::WallModelConfig<double> config;
  ugkwp::gaswall::WallInput<double,2>*input=nullptr;
  double*particleSource=nullptr,*particleHeat=nullptr,*particleReservoir=nullptr,*candidateHistory=nullptr;
  int*faces=nullptr;
 }gasBoundaryLayerModel;
 ugkwp::gaswall::WallInput<double,2> input[1];
 double source[32]{},heat[8]{},reservoirs[72]{},history[48]{};
};
int main(){
 for(bool thermal:{false,true}){
  WallSpatialState s;setup(s);s.solveParticleTemperature=thermal;s.particleGasHeatTransferModelId=thermal?1:0;
  s.gasBoundaryLayer.ownerSlot=s.ownerMap;s.gasBoundaryLayer.faceSlot=s.faceMap;
  auto&m=s.gasBoundaryLayerModel;m.config.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
  m.config.nodes=8;m.config.stretch=1;m.input=s.input;m.particleSource=s.source;
  m.particleHeat=s.heat;m.particleReservoir=s.reservoirs;m.candidateHistory=s.history;m.faces=s.faces;
  auto&in=s.input[0];in.normal[1]=1;in.matchingDistance=.7;in.ownerDistance=.35;in.temperature=400;
  in.pressure=101325;in.matching.temperature=900;in.matching.velocity[0]=80;
  in.model.thermo=s.gasSpecies.thermo;in.model.viscosity=1.8e-5;
  for(int j=0;j<2;++j)in.matching.massFraction[j]=s.gasSpecies.rho[j*s.nCells]/s.rho[0];
  ugkwp::gaswall::detail::LayerContext<double,2> ctx;
  assert(ugkwp::gaswall::detail::makeContext(in,m.config,ctx));
  for(int i=0;i<8;++i){s.history[i*6]=80.*i/7.;s.history[i*6+2]=500.*i/7.;
   s.history[i*6+3]=in.matching.massFraction[ctx.independent[0]];s.history[i*6+5]=1;}
  for(int i=0;i<2;++i){s.pux[i]=s.puy[i]=s.puz[i]=0;s.pd[i]=.001;s.pm[i]=1e-4;s.pTheta[i]=0;s.pT[i]=300;}
  auto before=budget(s);const double species[2]={s.gasSpecies.rho[0],s.gasSpecies.rho[2]};
  const double oldP[3]={s.rhoUx[0],s.rhoUy[0],s.rhoUz[0]},oldE=s.rhoE[0];
  mixtureParticleCouplingCell<ugkwpGpuDrag::SchillerNaumannDeviceDrag,true>(s,0,.001,true,true,false,mixtureParticles::SerialCouplingTeam{});
  assert(s.gasSpecies.cellStatus[0]==0);conserved(before,budget(s));
  assert(s.pux[0]<s.pux[1]); // identical parcels; only their wall-normal positions differ
  const double volume=s.epsGPrev[0]*s.V[0],areaDt=s.epsGPrev[0]*s.magSf[0]*.001;
  double integrated[4]={},sumV=0,sumM=0;
  for(int i=0;i<8;++i){for(int d=0;d<4;++d)integrated[d]+=s.source[4*i+d]*areaDt;
   sumV+=s.reservoirs[9*i];sumM+=s.reservoirs[9*i+1];
   if(i>0&&i<7)for(int d=0;d<4;++d)assert(s.source[4*i+d]==0);
   if(!thermal)assert(s.heat[i]==0);
  }
  for(int d=0;d<3;++d){const double p=d==0?s.rhoUx[0]:d==1?s.rhoUy[0]:s.rhoUz[0];closeValue(integrated[d],volume*(p-oldP[d]),1e-12,"source impulse");}
  closeValue(integrated[3],volume*(s.rhoE[0]-oldE),5e-8,"source energy");
  closeValue(sumV,volume,1e-14,"reservoir volumes");closeValue(sumM,volume*s.rho[0],1e-14,"reservoir masses");
  assert(s.gasSpecies.rho[0]==species[0]&&s.gasSpecies.rho[2]==species[1]);
  if(!thermal)assert(s.pT[0]==300&&s.pT[1]==300);
 }
}
''')
