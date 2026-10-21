// Bounded CPU collective/ownership coverage, not CUDA qualification.
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCoherentSolver.H"
#include <memory>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
struct EngineBarrier {std::mutex mutex;std::condition_variable cv;int size,arrived=0,generation=0;explicit EngineBarrier(int n):size(n){};void wait(){std::unique_lock<std::mutex> lock(mutex);int g=generation;if(++arrived==size){arrived=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
struct EngineTeamStorage {EngineBarrier barrier;bool flags[32]{},flag=false;double scalar=0;explicit EngineTeamStorage(int n):barrier(n){}};
struct EngineHostTeam {EngineTeamStorage&s;int lane,lanes;int rank()const{return lane;}int size()const{return lanes;}bool leader()const{return lane==0;}void sync()const{s.barrier.wait();}bool all(bool v)const{s.flags[lane]=v;sync();if(leader()){s.flag=true;for(int i=0;i<lanes;++i)s.flag=s.flag&&s.flags[i];}sync();v=s.flag;sync();return v;}bool broadcast(bool v)const{if(leader())s.flag=v;sync();v=s.flag;sync();return v;}template<class T>T broadcast(T v)const{if(leader())s.scalar=double(v);sync();v=T(s.scalar);sync();return v;}};

using namespace ugkwp::gaswall;
using namespace ugkwp::gaswall::reconstruction;
struct Stats {int iterations=0,residualEvaluations=0,factorizations=0;double physicalResidual=0,seconds=0;};
bool equalNumber(double a,double b){return a==b||(std::isnan(a)&&std::isnan(b));}
void compareOutput(const WallOutput<double,2>&a,const WallOutput<double,2>&b){
#define FIELD(x) assert(equalNumber(a.x,b.x))
 FIELD(traceTemperature);FIELD(traceDensity);FIELD(conductiveHeatFlux);FIELD(integratedKSource);FIELD(ownerOmega);FIELD(wallKFlux);FIELD(ownerKContent);FIELD(ownerKDestruction);FIELD(ownerKBalanceResidual);FIELD(heatConduction);FIELD(heatViscous);FIELD(heatTransport);FIELD(heatBalanceResidual);FIELD(residual);
 for(int i=0;i<2;++i){FIELD(traceMassFraction[i]);FIELD(wallSpeciesFlux[i]);FIELD(matchingSpeciesFlux[i]);FIELD(reactionIntegral[i]);FIELD(chemistryMismatch[i]);FIELD(speciesBalanceResidual[i]);}
 for(int i=0;i<3;++i){FIELD(traceVelocity[i]);FIELD(traction[i]);}
 for(int i=0;i<129;++i){FIELD(kFluxHeight[i]);FIELD(kTransportFlux[i]);FIELD(omegaDiffusiveFlux[i]);}
 assert(a.kFluxCount==b.kFluxCount&&a.chemistryMismatchAvailable==b.chemistryMismatchAvailable&&a.iterations==b.iterations);
#undef FIELD
}
template<int M>struct Bundle {WallWorkspace<double,2,M>w;coherent::Cache<M>cache;coherent::Scratch<M>scratch;WallOutput<double,2>out;Stats stats;};
template<int M>bool teamSolve(Bundle<M>&b,const Input&in,const Config&cfg,int lanes){EngineTeamStorage shared(lanes);std::vector<std::thread> threads;bool result=false;for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{EngineHostTeam team{shared,lane,lanes};coherent::Engine<M,EngineHostTeam> e(b.scratch,team);bool ok=e.solve(in,cfg,b.w,b.cache,b.out,b.stats);if(team.leader())result=ok;});for(auto&t:threads)t.join();return result;}
template<int M>void compare(const Bundle<M>&a,const Bundle<M>&b,int nodes){
 for(int i=0;i<nodes;++i){assert(a.w.y[i]==b.w.y[i]);for(int j=0;j<6;++j){assert(equalNumber(a.w.state[i][j],b.w.state[i][j]));assert(equalNumber(a.w.residual[i][j],b.w.residual[i][j]));}}
 compareOutput(a.out,b.out);assert(a.stats.iterations==b.stats.iterations&&a.stats.residualEvaluations==b.stats.residualEvaluations&&a.stats.factorizations==b.stats.factorizations&&a.stats.physicalResidual==b.stats.physicalResidual);
 assert(a.cache.haveState==b.cache.haveState&&a.cache.factors.valid==b.cache.factors.valid&&a.cache.branches==b.cache.branches);
 assert(a.scratch.fdRoundoff==b.scratch.fdRoundoff&&a.scratch.fdBranchHalves==b.scratch.fdBranchHalves&&a.scratch.fdFailures==b.scratch.fdFailures&&a.scratch.factorFailures==b.scratch.factorFailures&&a.scratch.centralDirections==b.scratch.centralDirections);
 assert(a.scratch.currentNutBranches==b.scratch.currentNutBranches);
}
template<int M>void initialize(Bundle<M>&b,const Input&in,const Config&cfg){
 // Resampled captured transformed state tests bounded indexing, not a fine root.
 WallWorkspace<double,2,24> captured;loadCapturedSstIterate(captured);
 for(int i=0;i<cfg.nodes;++i){double x=double(i)*23/(cfg.nodes-1);int l=std::min(22,int(x));double t=x-l;b.w.y[i]=in.matchingDistance*std::pow(double(i)/(cfg.nodes-1),cfg.stretch);for(int j=0;j<6;++j)b.w.state[i][j]=captured.state[l][j]+t*(captured.state[l+1][j]-captured.state[l][j]);}
 b.cache.haveState=true;
}
template<int M>void advisory(int lanes,int nodes,bool sst){CapturedSstWall m;auto cfg=m.config;cfg.nodes=nodes;cfg.maxIterations=0;cfg.enableSst=sst;m.input.matching.k=0;m.input.volumeSstClosure=true;m.input.temperature=300;m.input.matching.temperature=290;for(int d=0;d<3;++d)m.input.matching.velocity[d]=0;auto a=std::make_unique<Bundle<M>>(),b=std::make_unique<Bundle<M>>();initialize(*a,m.input,cfg);initialize(*b,m.input,cfg);
 // Manufactured finite laminar advisory iterate; the zero-k SST extension
 // is a matched publication-status control, not a converged fine SST root.
 for(int i=0;i<nodes;++i){double eta=double(i)/(nodes-1);for(auto* bundle:{a.get(),b.get()}){bundle->w.state[i][0]=bundle->w.state[i][1]=0;bundle->w.state[i][2]=-10*eta*eta*eta*eta;bundle->w.state[i][3]=0;bundle->w.state[i][4]=0;bundle->w.state[i][5]=sst?eta*eta/std::sqrt(m.input.matching.omega):1;}}
 coherent::Engine<M> e(a->scratch);bool expected=e.solve(m.input,cfg,a->w,a->cache,a->out,a->stats);bool actual=teamSolve(*b,m.input,cfg,lanes);assert(expected==actual);if(!sst)assert(expected);assert(a->stats.physicalResidual>cfg.absoluteTolerance+cfg.relativeTolerance);compare(*a,*b,nodes);std::cout<<"advisory capacity="<<M<<" nodes="<<nodes<<" lanes="<<lanes<<" sst="<<sst<<" published="<<expected<<" residual="<<a->stats.physicalResidual<<"\n";}
int main(){for(int lanes:{1,4,7,32})for(bool sst:{false,true}){advisory<24>(lanes,3,sst);advisory<24>(lanes,7,sst);advisory<24>(lanes,24,sst);advisory<48>(lanes,48,sst);advisory<128>(lanes,128,sst);}std::cout<<"20 finite laminar advisory + 20 zero-k SST controls exact (publication status retained)\n";}
