#include <new>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
static unsigned long long allocations=0,allocatedBytes=0;
void* operator new(std::size_t n) { ++allocations;allocatedBytes+=n;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
#define main original_film_main
#include "FILM_FIXTURE"
#undef main
#include "FILM_OWNER"
void qout(const FilmQ& q){std::cout<<q.mass<<' '<<q.enthalpy;for(double x:q.species)std::cout<<' '<<x;std::cout<<'\n';}
void vout(const Vec3& v){std::cout<<v.x<<' '<<v.y<<' '<<v.z<<'\n';}
void aout(const FilmAux& a){std::cout<<a.pressure<<' '<<a.thickness<<' '<<a.temperature<<' '<<a.area<<' '<<a.kineticEnergy<<' '<<a.normalVelocity<<' '<<a.solidNormalVelocity<<' '<<a.solidFront<<' '<<a.gasFront<<'\n';vout(a.baseVelocity);vout(a.meanVelocity);vout(a.topVelocity);vout(a.normal);}
void bout(const FilmRateBudget& b){std::cout<<b.bodyPower<<' '<<b.supportPower<<' '<<b.prescribedTopPower<<' '<<b.radiationPower<<' '<<b.edgeEnergyOutflow<<' '<<b.edgeKineticOutflow<<' '<<b.interfaceKineticOutflow<<' '<<b.dissipationPower<<'\n';}
void budgetout(const Budget& b){
 std::cout<<b.boundaryMass<<' '<<b.boundaryEnergy<<' '<<b.supportWork<<' '<<b.bodyWork<<' '<<b.radiation<<' '<<b.gclResidual<<' '
 <<b.filmPressureVolume<<' '<<b.filmKineticAdvection<<' '<<b.filmKineticStorage<<' '<<b.filmReducedResidual<<' '<<b.filmKineticDefect<<' '
 <<b.thinLayerGeometryDefect<<' '<<b.thinLayerStressDefect<<' '<<b.numericalEnergyResidual<<' '<<b.turbulenceOmegaConstraint<<' '
 <<b.turbulenceInventory<<' '<<b.turbulenceProduction<<' '<<b.turbulenceDissipation<<' '<<b.turbulenceBoundaryFlux<<' '<<b.consumedPackets<<'\n';
 for(auto x:b.boundarySpecies)std::cout<<x<<' ';for(auto x:b.boundaryElements)std::cout<<x<<' ';std::cout<<'\n';
 vout(b.boundaryMomentum);vout(b.supportImpulse);
 for(int k=0;k<NParticipants;++k){std::cout<<b.exchangeMass[k]<<' '<<b.exchangeEnergy[k]<<' ';for(auto x:b.exchangeSpecies[k])std::cout<<x<<' ';std::cout<<'\n';vout(b.exchangeMomentum[k]);}
}
void specialWitnesses() {
 auto p=physics();p.spatialOrder=2;p.reconstruction=ReconstructionMode::LimitedLinear;p.liquid.e0=0;
 auto h=state(p,3,1,Vec3{.2,0,0});
 h.surface.normal.assign(3,Vec3{0,0,1});h.surface.edgeOwner={0,1};h.surface.edgeNeighbour={1,2};
 h.surface.edgeLength={1,1};h.surface.edgeConormal.assign(2,Vec3{1,0,0});
 h.surface.edgeOwnerOffset.assign(2,Vec3{.5,0,0});h.surface.edgeNeighbourOffset.assign(2,Vec3{-.5,0,0});
 const Real mass[]={2,1,.5},temperature[]={500,200,100};
 for(int f=0;f<3;++f){h.film[f]=FilmQ{};h.film[f].mass=mass[f];h.film[f].species[0]=mass[f];
  h.film[f].enthalpy=mass[f]*liquidH(p,temperature[f],100000);h.filmAux[f]=FilmAux{};
  require(recoverFilm(h.film[f],1,100000,p,h.filmAux[f]),"retry fixture recovery");}
 auto d=drive(h);Evaluation e;std::string error;require(recoverAndProfile(h,p,d,e,error),error);
 FilmQ reconstructed;acceptedTrial=999;require(reconstruct(1,Vec3{.5,0,0},e,h.surface,p,reconstructed),"limited retry succeeds");
 require(acceptedTrial==1,"limited reconstruction retains successful second trial");qout(reconstructed);
 for(int f=0;f<3;++f)d[f].endpointPressure=100000+500*f*(f+1);
 CpuFilmCandidate c;require(advanceCpuFilmCandidate(h,p,.01,std::vector<CpuFilmForcing>(3),d,c,error),error);
 require(near(c.report.edgeFlux[1].mass,67.0/380)&&near(c.report.edgeFlux[1].enthalpy,50000),"limited retry actual edge flux");
 const Real endpointGradient[]={1000,1500,2000};
 for(int f=0;f<3;++f){const Real thickness=c.film[f].mass/1000;
  const auto profile=filmProfile(thickness,p.liquidViscosity,Vec3{.2,0,0},{},Vec3{endpointGradient[f],0,0});
  require(near(c.filmAux[f].meanVelocity.x,profile.meanVelocity.x,1e-15)&&near(c.filmAux[f].topVelocity.x,profile.topVelocity.x,1e-15),"endpoint incidence uses changed pressure and thickness");
  require(near(c.filmAux[f].kineticEnergy,p.liquid.rho*profile.kineticPerArea,1e-15),"endpoint kinetic profile");}
 for(auto q:c.film)qout(q);for(auto a:c.filmAux)aout(a);
 // Repeated internal edges, nonuniform degree, a boundary-only face, and a
 // shuffled global order. The baseline's serial scan is the exact reference.
 for(int variant=0;variant<3;++variant){
  auto mesh=h.surface;mesh.edgeOwner={1,2,0,0,1};mesh.edgeNeighbour={2,-1,1,1,0};
  mesh.edgeLength.assign(5,1);mesh.edgeConormal.assign(5,Vec3{1,0,0});
  mesh.edgeOwnerOffset.assign(5,Vec3{.5,0,0});mesh.edgeNeighbourOffset.assign(5,Vec3{-.5,0,0});
  if(variant==1){std::reverse(mesh.edgeOwner.begin(),mesh.edgeOwner.end());std::reverse(mesh.edgeNeighbour.begin(),mesh.edgeNeighbour.end());}
  if(variant==2){mesh.edgeOwner={2};mesh.edgeNeighbour={-1};mesh.edgeLength.resize(1);mesh.edgeConormal.resize(1);mesh.edgeOwnerOffset.resize(1);mesh.edgeNeighbourOffset.resize(1);}
  h.surface=mesh;for(int f=0;f<3;++f)d[f].pressure=100000+17*f*f;
  require(recoverAndProfile(h,p,d,e,error),error);
  for(int f=0;f<3;++f){vout(surfaceGradient(f,-1,e,h.surface));aout(e.aux[f]);}
  // Reusing Evaluation after changing actual geometry must read the new offsets.
  for(auto& x:h.surface.edgeOwnerOffset)x=x*1.75;
  for(auto& x:h.surface.edgeNeighbourOffset)x=x*1.75;
  require(recoverAndProfile(h,p,d,e,error),error);
  for(int f=0;f<3;++f)vout(surfaceGradient(f,-1,e,h.surface));
 }
 HostState empty;const auto emptyStart=allocations;const bool emptyOk=recoverAndProfile(empty,p,{},e,error);const auto emptyAllocations=allocations-emptyStart;require(emptyOk,error);require(emptyAllocations==0,"empty recovery allocates nothing");require(e.q.empty()&&e.aux.empty()&&e.gradient.empty(),"empty recovery after populated Evaluation");
 CpuFilmCandidate emptyCandidate;const auto emptyAdvanceStart=allocations;const bool emptyAdvanced=advanceCpuFilmCandidate(empty,p,.01,{}, {},emptyCandidate,error);const auto emptyAdvanceAllocations=allocations-emptyAdvanceStart;require(emptyAdvanced,error);require(emptyAdvanceAllocations==0,"fresh empty candidate allocates nothing");
 for(int boundary:{0,1}){
  auto isolated=state(p,1,1,tangentX*.2);if(boundary){isolated.surface.edgeOwner={0};isolated.surface.edgeNeighbour={-1};isolated.surface.edgeLength={1};isolated.surface.edgeConormal={tangentX};isolated.surface.edgeOwnerOffset={tangentX*.5};isolated.surface.edgeNeighbourOffset={Vec3{}};}
  auto isolatedDrive=drive(isolated);Evaluation isolatedEvaluation;
  const auto isolatedAc=allocations,isolatedBc=allocatedBytes;
  const bool isolatedOk=recoverAndProfile(isolated,p,isolatedDrive,isolatedEvaluation,error);
  const auto isolatedAllocations=allocations-isolatedAc,isolatedBytes=allocatedBytes-isolatedBc;
  require(isolatedOk,error);require(mag(isolatedEvaluation.gradient[0])==0,"no internal neighbors gives zero pressure gradient");
  std::cerr<<"ISOLATED "<<boundary<<' '<<isolatedAllocations<<' '<<isolatedBytes<<'\n';aout(isolatedEvaluation.aux[0]);
 }
 std::cout<<"special witnesses passed\n";
}
int main(){std::cout<<std::hexfloat;
for(int ny:{8,16,32,64})for(int order:{1,2})for(int mode:{0,1})for(int sign:{-1,1}){
 auto p=physics();p.spatialOrder=order;p.reconstruction=static_cast<ReconstructionMode>(mode);p.liquid.conductivity=10;
 auto h=state(p,3,ny,tangentX*(sign*.1)+tangentY*.03);auto d=drive(h);
 for(size_t f=0;f<d.size();++f){d[f].pressure+=f*3;d[f].endpointPressure=100500+f*f*.25;}
 std::vector<FilmQ> rhs;std::vector<FilmRateBudget> budgets;std::vector<FilmAux> aux;std::vector<Vec3> gradients;std::vector<Real> side;std::string error;
 gradientVisits=limiterVisits=buildEdgeVisits=buildEntries=buildFaces=0;const auto ac=allocations,bc=allocatedBytes;
 const bool ok=evaluateCpuFilmTransport(h,p,.001,d,rhs,budgets,error,&aux,&gradients,&side);
 const auto allocated=allocations-ac,bytes=allocatedBytes-bc;require(ok,error);
 if(order==2&&mode==0&&sign==1)std::cerr<<"COUNT "<<h.film.size()<<' '<<h.surface.edgeOwner.size()<<' '<<gradientVisits<<' '<<limiterVisits<<' '<<allocated<<' '<<bytes<<'\n';
 if(order==2&&mode==0&&sign==1)std::cerr<<"BUILD "<<h.film.size()<<' '<<h.surface.edgeOwner.size()<<' '<<buildEdgeVisits<<' '<<buildEntries<<' '<<buildFaces<<'\n';
 for(auto q:rhs)qout(q);for(auto g:gradients)vout(g);for(auto x:side)std::cout<<x<<'\n';for(auto a:aux)aout(a);for(auto b:budgets)bout(b);
 std::vector<CpuFilmForcing> forcing(h.film.size());Real dt=0;require(estimateCpuFilmStep(h,p,forcing,d,.001,.001,dt,error),error);std::cout<<dt<<'\n';
 CpuFilmCandidate c;profileCalls=0;const auto advanceAc=allocations,advanceBc=allocatedBytes;
 const bool advanced=advanceCpuFilmCandidate(h,p,.001,forcing,d,c,error);
 const auto advanceAllocations=allocations-advanceAc,advanceBytes=allocatedBytes-advanceBc;
 require(advanced,error);require(profileCalls==2,"initial and direct endpoint profile paths");
 if(order==2&&mode==0&&sign==1)std::cerr<<"ADVCOUNT "<<h.film.size()<<' '<<advanceAllocations<<' '<<advanceBytes<<'\n';
 for(auto q:c.film)qout(q);for(auto q:c.report.edgeFlux)qout(q);for(auto a:c.filmAux)aout(a);for(auto b:c.report.rateBudget)bout(b);budgetout(c.report.budgetDelta);std::cout<<c.report.phaseEvent<<'\n';
 std::cout<<c.report.internalEnergyBefore<<' '<<c.report.internalEnergyAfter<<' '<<c.report.reducedEnergyResidual<<' '<<c.report.kineticDefect<<' '<<c.report.numericalMassRoundoff<<'\n';
}
specialWitnesses();return 0;}
