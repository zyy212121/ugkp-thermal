#define main original_film_main
#include "FILM_FIXTURE"
#undef main
#include "FILM_OWNER"
#include <cassert>
int main(){
 auto p=physics();p.spatialOrder=2;p.reconstruction=ReconstructionMode::SmoothVerification;
 auto h=state(p,3,3,tangentX*.2);auto d=drive(h);Evaluation e;std::string error;
 assert(recoverAndProfile(h,p,d,e,error));
 // Every density candidate has an invalid species sum, while the real last
 // computed gradient correction remains distinguishable from the next proposal.
 for(auto& q:e.q)q.species[0]+=100;
 const Vec3 offset=tangentX*.5;FilmQ output;output.mass=987;
 filmTrials=0;assert(!reconstruct(0,offset,e,h.surface,p,output));assert(filmTrials==48&&output.mass==987);
 filmTrials=0;assert(reconstruct<true>(0,offset,e,h.surface,p,output));assert(filmTrials==48);
 const Real scale=::ldexp(1.,-47);
 assert(output.mass==surfaceValue(0,0,e,h.surface)+scale*dot(surfaceGradient(0,0,e,h.surface),offset));
 assert(output.enthalpy==surfaceValue(0,1,e,h.surface)+scale*dot(surfaceGradient(0,1,e,h.surface),offset));
 for(int s=0;s<Ns;++s)assert(output.species[s]==surfaceValue(0,s+2,e,h.surface)+scale*dot(surfaceGradient(0,s+2,e,h.surface),offset));
 assert(output.mass!=surfaceValue(0,0,e,h.surface)+scale*.5*dot(surfaceGradient(0,0,e,h.surface),offset));
 std::cout<<"film_terminal_actual_trial 48\n";
 // Early recovery failure must still produce actual edge flux/rate/profile tails.
 p.spatialOrder=1;h=state(p,3,3,tangentX*.2);d=drive(h);h.film[0].enthalpy+=1e10;
 std::vector<FilmQ> rates;std::vector<FilmAux> aux;
 std::vector<Vec3> gradients;std::vector<Real> sides;bool numeric=false;
 assert(!evaluateCpuFilmTransport(h,p,.01,d,rates,error,nullptr,nullptr,nullptr,&numeric)&&numeric);
 assert(detail::completeCpuFilmTransport(h,p,.01,d,rates,error,&aux,&gradients,&sides,&numeric));
 assert(rates.size()==9&&aux.size()==9&&gradients.size()==9&&sides.size()==9);
 assert(rates[0].enthalpy!=0&&aux[0].temperature==p.liquid.Tmax);
 // Endpoint-only thermal failure: actual energy source reaches endpoint and
 // unchanged profile and kinetic formulas are evaluated at the actual mass.
 h=state(p,1,1,tangentX*.2);d=drive(h);std::vector<CpuFilmForcing> forcing(1);forcing[0].energy=1e10;
 CpuFilmCandidate candidate;numeric=false;
 assert(!advanceCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric)&&numeric);
 assert(detail::completeCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric));
 assert(candidate.film[0].enthalpy==h.film[0].enthalpy+forcing[0].energy);
 assert(candidate.filmAux[0].temperature==p.liquid.Tmax);
 assert(candidate.filmAux[0].thickness==candidate.film[0].mass/p.liquid.rho);
 assert(filmInternalEnergy(candidate.film[0],candidate.filmAux[0].pressure,p)==h.film[0].enthalpy+forcing[0].energy-candidate.filmAux[0].pressure*candidate.film[0].mass/p.liquid.rho);
 assert(candidate.filmAux[0].kineticEnergy==h.filmAux[0].kineticEnergy);
 // Missing actual storage remains a failure, even on terminal evaluation.
 h.filmAux.clear();numeric=true;
 assert(!detail::completeCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric)&&!numeric);
 assert(error.find("dimensions")!=std::string::npos);
 // Extra supplied storage is not missing input or an identity prerequisite.
 h=state(p,1,1,{});d=drive(h);forcing.assign(1,CpuFilmForcing{});
 h.filmAux.push_back(h.filmAux[0]);h.surface.area.push_back(7);h.surface.normal.push_back(normal);
 d.push_back(d[0]);forcing.push_back(CpuFilmForcing{});
 assert(!advanceCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric)&&numeric);
 assert(detail::completeCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric));
 assert(candidate.film[0].mass==h.film[0].mass&&candidate.film[0].enthalpy==h.film[0].enthalpy);
 h=state(p,1,1,{});d=drive(h);forcing.assign(1,CpuFilmForcing{});
 h.surface.gasFace.push_back(-1);h.surface.edgeNeighbour.push_back(-1);
 assert(!advanceCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric)&&numeric);
 assert(detail::completeCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric));
 assert(candidate.film[0].mass==h.film[0].mass&&candidate.film[0].enthalpy==h.film[0].enthalpy);
 h=state(p,1,1,{});h.surface.gasFace[0]=0;d=drive(h);forcing.assign(1,CpuFilmForcing{});numeric=true;
 assert(!detail::completeCpuFilmCandidate(h,p,.01,forcing,d,candidate,error,&numeric)&&!numeric);
 assert(error.find("missing coupled normal trace")!=std::string::npos);
 std::cout<<"film_early_and_endpoint_tails complete missing_storage_and_trace propagated\n";
}
