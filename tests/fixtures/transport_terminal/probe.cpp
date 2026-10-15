#include <new>
#include <cstdlib>
static unsigned long long allocations=0,bytes=0;static bool failAllocation=false;
void* operator new(std::size_t n){if(failAllocation){failAllocation=false;throw std::bad_alloc();}++allocations;bytes+=n;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,std::size_t)noexcept{std::free(p);}
#include "owner.C"
#include "fixture.hpp"
void emit(const SolidQ& q){for(auto x:q.condensed)std::cout<<x<<' ';for(auto x:q.pore)std::cout<<x<<' ';for(auto x:q.progress)std::cout<<x<<' ';std::cout<<q.energy<<' '<<q.porosity<<' ';}
void output(const MaterialTransportResult& out){std::cout<<out.dtLimit<<' '<<out.bodyPower<<' ';for(const auto& q:out.rates)emit(q);for(const auto& q:out.faceFlux)emit(q);for(const auto& v:out.poreVelocity)std::cout<<v.x<<' '<<v.y<<' '<<v.z<<' ';std::cout<<'\n';}
int main(){PhysicsConfig p;HostMesh m;std::vector<SolidQ> q;std::string error;MaterialTransportResult out;
 std::cout<<std::hexfloat;
 for(int mode=0;mode<6;++mode){fixture(p,m,q);p.gravity={.2,-.1,.3};if(mode>0)p.spatialOrder=2;if(mode==2)p.permeability=0;if(mode==3)p.reconstruction=ReconstructionMode::SmoothVerification;
  std::vector<Real> swept=mode==4?std::vector<Real>{1e-7,-2e-7,3e-7}:std::vector<Real>{};
  rejectBefore=mode==5?47:0;allocations=bytes=trials=0;assert(evaluateMaterialTransport(m,q,p,swept,.01,.4,out,error));auto a=allocations,b=bytes,t=trials;output(out);std::cout<<"COUNT "<<mode<<' '<<a<<' '<<b<<' '<<t<<'\n';}
#if TERMINAL_TEST
 rejectBefore=0;fixture(p,m,q);p.poreViscosity=0;bool numerical=false;
 assert(!evaluateMaterialTransport(m,q,p,{},.01,.4,out,error,&numerical)&&numerical);
 assert(detail::completeMaterialTransport(m,q,p,{},.01,.4,out,error,&numerical)&&!numerical);
 assert(out.rates.size()==4&&out.faceFlux.size()==3&&out.poreVelocity.size()==4);
 assert(!chmt::finite(out.poreVelocity[0].x)&&!chmt::finite(out.faceFlux[0].pore[0]));
 fixture(p,m,q);p.gravity={.2,-.1,.3};MaterialTransportResult normal;assert(evaluateMaterialTransport(m,q,p,{},.01,.4,normal,error));
 assert(detail::completeMaterialTransport(m,q,p,{},.01,0,out,error,&numerical));assert(out.dtLimit==0&&out.bodyPower==normal.bodyPower);
 for(std::size_t c=0;c<q.size();++c){SolidQ consumed=addSolid(q[c],out.rates[c],.01);assert(consumed.energy==q[c].energy+.01*out.rates[c].energy);}
 auto unused=m;unused.boundaryKind[1]=unused.boundaryKind[2]=BoundaryKind::Interface;unused.periodicPartner.clear();unused.areaVectors.resize(1);unused.faceCentres.resize(1);
 assert(detail::completeMaterialTransport(unused,q,p,{1e-7},.01,.4,out,error,&numerical));assert(out.faceFlux.size()==3);
 bool threw=false;failAllocation=true;try{detail::completeMaterialTransport(m,q,p,{},.01,.4,out,error,&numerical);}catch(const std::bad_alloc&){threw=true;}assert(threw&&!failAllocation);
 auto missing=m;missing.cellCentres.clear();assert(!detail::completeMaterialTransport(missing,q,p,{},.01,.4,out,error,&numerical)&&!numerical);
 auto unknown=p;unknown.species[0].model=static_cast<ugkwp::SpeciesThermoModel>(99);assert(!detail::completeMaterialTransport(m,q,unknown,{},.01,.4,out,error,&numerical)&&!numerical);
 Real thickness=7;assert(!filmThicknessFromMass(-2,3,1000,thickness)&&thickness==7);assert(detail::completeFilmThicknessFromMass(-2,3,1000,thickness)&&thickness==-2./3000);
 // Source-only instrumentation rejects each ordinary EOS trial after its candidate is actually formed.
 fixture(p,m,q);p.spatialOrder=2;p.reconstruction=ReconstructionMode::SmoothVerification;q[1].energy*=1e20;rejectTrials=true;trials=0;SolidQ density;
 assert(reconstruct<true>(m,q,p,0,0,density));assert(trials==48);
 assert(density.energy==lastCandidate.energy);assert(lastBound==std::ldexp(1.,-47));assert(density.energy!=nextEnergy);
 rejectTrials=false;
 std::cout<<"terminal actual RHS consumer, signed thickness, trial48 and missing input passed\n";
#endif
}
