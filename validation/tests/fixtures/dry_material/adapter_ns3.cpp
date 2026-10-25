#include "materials/DryMaterialAdapter.H"
#include "materials/MaterialTransport.H"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
using namespace chmt;
int main(){PhysicsConfig p;p.spatialOrder=1;
 for(auto& c:p.condensed){c.rho=1000;c.cp0=1000;c.conductivity=1;c.Tmin=100;c.Tmax=3000;}
 for(auto& s:p.species){s.R=287;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
 p.poreEffectiveDiffusivity[0]=1e-6;p.poreEffectiveDiffusivity[1]=1e-4;p.poreEffectiveDiffusivity[2]=1e-6;
 HostMesh m;m.volumes={1,1};m.cellCentres={{0,0,0},{1,0,0}};m.owner={0};m.neighbour={1};m.areaVectors={{1,0,0}};m.faceCentres={{.5,0,0}};m.periodicPartner={-1};m.boundaryKind={BoundaryKind::Internal};m.cellFaceOffsets={0,1,2};m.cellFaces={0,0};m.cellFaceSigns={1,-1};
 std::vector<SolidQ> q(2);q[0].pore[1]=q[0].pore[2]=.25;q[1].pore[0]=.05;q[1].pore[1]=.45;
 for(auto& a:q){a.condensed[0]=500;a.energy=dry::caloricState(dryAdapter::state(a),dryAdapter::condensed(p),dryAdapter::GasCalorics{p},600).energy;}
 MaterialPrimitive a,b;recoverMaterial(q[0],1,p,a);recoverMaterial(q[1],1,p,b);
 std::array<double,Ns>D{{1e-6,1e-4,1e-6}};auto core=dry::poreFaceFlux(dryAdapter::poreState(q[0],a),dryAdapter::poreState(q[1],b),1,0,D,0,dryAdapter::GasCalorics{p});
 assert(std::abs(core.species[0]+1e-7)<1e-19);
 double sum=0;for(auto x:core.species)sum+=x;assert(std::abs(sum)<1e-18);
 const auto reversed=dry::poreFaceFlux(dryAdapter::poreState(q[1],b),dryAdapter::poreState(q[0],a),1,0,D,0,dryAdapter::GasCalorics{p});
 for(int s=0;s<Ns;++s)assert(std::abs(core.species[s]+reversed.species[s])<1e-18);
 MaterialTransportResult out;std::string error;bool ok=evaluateMaterialTransport(m,q,p,{},.01,.4,out,error);assert(ok&&out.dtLimit>0&&std::isfinite(out.dtLimit));
 for(int c=0;c<2;++c)for(int s=0;s<Ns;++s){
     assert(q[c].pore[s]+out.dtLimit*out.rates[c].pore[s]>=0);
     assert(std::abs(out.rates[0].pore[s]+out.rates[1].pore[s])<1e-18);
 }
 // Deterministic donor-support property across unequal coefficients and
 // either zero-support side, including reversal of correction velocity.
 std::mt19937 random(241024);std::uniform_real_distribution<double> unit(.01,1);
 for(int trial=0;trial<300;++trial){
  dry::PoreFaceState<Ns> left,right;left.density=right.density=1;left.temperature=right.temperature=600;
  std::array<double,Ns> coeff{};double lsum=0,rsum=0;
  for(int species=0;species<Ns;++species){left.massFraction[species]=unit(random);right.massFraction[species]=unit(random);coeff[species]=std::pow(10.,-8+5*unit(random));}
  left.massFraction[trial%Ns]=0;right.massFraction[(trial+1)%Ns]=0;
  for(int species=0;species<Ns;++species){lsum+=left.massFraction[species];rsum+=right.massFraction[species];}
  for(int species=0;species<Ns;++species){left.massFraction[species]/=lsum;right.massFraction[species]/=rsum;}
  const auto flux=dry::poreFaceFlux(left,right,1,0,coeff,0,dryAdapter::GasCalorics{p});
  double total=0,scale=0;
  for(int species=0;species<Ns;++species){
   total+=flux.species[species];scale+=std::abs(flux.species[species]);
   if(left.massFraction[species]==0)assert(flux.species[species]<=1e-20);
   if(right.massFraction[species]==0)assert(flux.species[species]>=-1e-20);
  }
  assert(std::abs(total)<1e-14*scale+1e-20);
 }
 std::cout<<"PASS Ns3 production zero-support conservative diffusion; dt="<<out.dtLimit<<'\n';
}
