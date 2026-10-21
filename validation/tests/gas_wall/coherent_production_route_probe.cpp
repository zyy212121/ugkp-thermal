// Self-contained recorded warm-FD regression + actual production history route.
// CPU collectives only; no CUDA launch/occupancy claim.
#include "fixtures/coherent_warm_regression.H"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cassert>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <iostream>
struct RouteBarrier{std::mutex m;std::condition_variable cv;int size,arrived=0,generation=0;explicit RouteBarrier(int n):size(n){}void wait(){std::unique_lock<std::mutex> lock(m);int g=generation;if(++arrived==size){arrived=0;++generation;cv.notify_all();}else cv.wait(lock,[&]{return generation!=g;});}};
struct RouteShared{RouteBarrier barrier;bool flags[32]{},flag=false;double scalar=0;explicit RouteShared(int n):barrier(n){}};
struct RouteTeam{RouteShared&s;int lane,lanes;int rank()const{return lane;}int size()const{return lanes;}bool leader()const{return lane==0;}void sync()const{s.barrier.wait();}bool broadcast(bool v)const{if(leader())s.flag=v;sync();v=s.flag;sync();return v;}double broadcast(double v)const{if(leader())s.scalar=v;sync();v=s.scalar;sync();return v;}bool all(bool v)const{s.flags[lane]=v;sync();if(leader()){s.flag=true;for(int i=0;i<lanes;++i)s.flag=s.flag&&s.flags[i];}sync();v=s.flag;sync();return v;}};
using namespace ugkwp;using namespace ugkwp::gaswall;
struct RouteState{double*rho=nullptr;GasSpeciesState<double,2>gasSpecies;GasBoundaryLayerModelState<double,2>gasBoundaryLayerModel;};
using Work=WallWorkspace<double,2,24>;
bool same(double a,double b){return a==b||(std::isnan(a)&&std::isnan(b));}
void compare(const WallOutput<double,2>&a,const WallOutput<double,2>&b){
 assert(a.conductiveHeatFlux==b.conductiveHeatFlux&&a.residual==b.residual&&a.iterations==b.iterations);
 for(int d=0;d<3;++d)assert(a.traction[d]==b.traction[d]&&a.traceVelocity[d]==b.traceVelocity[d]);
 for(int j=0;j<2;++j)assert(a.traceMassFraction[j]==b.traceMassFraction[j]&&a.wallSpeciesFlux[j]==b.wallSpeciesFlux[j]&&a.matchingSpeciesFlux[j]==b.matchingSpeciesFlux[j]&&a.reactionIntegral[j]==b.reactionIntegral[j]&&same(a.chemistryMismatch[j],b.chemistryMismatch[j]));
 assert(a.ownerOmega==b.ownerOmega&&a.integratedKSource==b.integratedKSource&&a.heatConduction==b.heatConduction&&a.heatViscous==b.heatViscous&&a.heatTransport==b.heatTransport);
 for(int i=0;i<a.kFluxCount;++i)assert(a.kTransportFlux[i]==b.kTransportFlux[i]&&a.omegaDiffusiveFlux[i]==b.omegaDiffusiveFlux[i]);
}
int main(int argc,char**argv){
 const int lanes=argc>1?std::atoi(argv[1]):1;CoherentWarmRegression fixture;auto cfg=fixture.config;const auto&in=fixture.input;
 auto scratch=std::make_unique<coherent::Scratch<24>>(),referenceScratch=std::make_unique<coherent::Scratch<24>>();
 auto walls=std::make_unique<coherent::PersistentWallState<24>[]>(2);Work work,control[2];coherent::Cache<24> controlCache[2];
 detail::LayerContext<double,2>ctx;assert(detail::makeContext(in,cfg,ctx));auto frame=historyFrame(in,cfg,ctx);
 CoherentWarmRegression::failedSeed(work);auto coherentConfig=cfg;coherentConfig.coherentDiscretization=true;coherent::Engine<24> scalar(*referenceScratch);WallOutput<double,2> first,second;double n1=0,n2=0;
 assert(scalar.refresh(in,coherentConfig,work,ctx,n1,first));double residual[24][6];for(int i=0;i<24;++i)for(int j=0;j<6;++j)residual[i][j]=work.residual[i][j];
 assert(scalar.refresh(in,coherentConfig,work,ctx,n2,second));assert(n1==n2&&first.conductiveHeatFlux==second.conductiveHeatFlux);for(int i=0;i<24;++i)for(int j=0;j<6;++j)assert(residual[i][j]==work.residual[i][j]);
 RouteState state;auto&m=state.gasBoundaryLayerModel;m.config=cfg;WallInput<double,2>inputs[2]={in,in};WallOutput<double,2>outputs[2];WallStatus statuses[2];WallReferenceStats stats[2];WallReferenceCounters counters[2];WallReferenceScratch<2,24> oldScratch;WallHistoryFrame<2> acceptedFrames[2]={frame,frame},candidateFrames[2];double accepted[288],candidate[288];ConservativeWallColumn columns[2];for(auto&c:columns)c.cells[0]=0;
 m.input=inputs;m.output=outputs;m.status=statuses;m.workspace=&work;m.workspaceCount=1;m.workspaceCapacity=24;m.referenceScratch=&oldScratch;m.referenceStats=stats;m.referenceCounters=counters;m.acceptedHistory=accepted;m.candidateHistory=candidate;m.acceptedFrame=acceptedFrames;m.candidateFrame=candidateFrames;m.coherentEnabled=true;m.coherentScratch=scratch.get();m.coherentState=walls.get();m.conservative.enabled=true;m.conservative.columns=columns;
 for(int slot=0;slot<2;++slot){if(slot)CoherentWarmRegression::failedSeed(control[slot]);else CoherentWarmRegression::acceptedSeed(control[slot]);walls[slot].frame=frame;walls[slot].cache.haveState=controlCache[slot].haveState=true;for(int i=0;i<24;++i)for(int j=0;j<6;++j)accepted[slot*144+i*6+j]=walls[slot].state[i][j]=control[slot].state[i][j];}
 const std::vector<double> immutableAccepted(accepted,accepted+288);
 auto invoke=[&](int slot){RouteShared shared(lanes);std::vector<std::thread>threads;bool result[32]{};for(int lane=0;lane<lanes;++lane)threads.emplace_back([&,lane]{result[lane]=evaluateGasBoundaryLayerHistory<false,24>(state,slot,0,RouteTeam{shared,lane,lanes});});for(auto&t:threads)t.join();for(int lane=1;lane<lanes;++lane)assert(result[lane]==result[0]);return result[0];};
 for(int slot:{0,1,0,1}){WallOutput<double,2>expected;coherent::SolveStats measured;assert(scalar.solve(in,cfg,control[slot],controlCache[slot],expected,measured));assert(measured.physicalResidual<=cfg.absoluteTolerance+cfg.relativeTolerance);assert(invoke(slot));compare(outputs[slot],expected);
  // Recorded first warm solve, independent of the live scalar/team comparison.
  // These constants retain the original failure's repaired numerical root.
  if(slot==0&&counters[slot].solves==1){
   assert(stats[slot].iterations==2&&stats[slot].residualEvaluations==65&&stats[slot].jacobianEvaluations==2);
   assert(stats[slot].residual==5.4443892631137862e-12);
   assert(outputs[slot].conductiveHeatFlux==-970.70635387845334);
  }
  for(int i=0;i<24;++i)for(int j=0;j<6;++j){assert(work.state[i][j]==control[slot].state[i][j]);assert(work.residual[i][j]==control[slot].residual[i][j]);assert(candidate[slot*144+i*6+j]==work.state[i][j]);}assert(stats[slot].usedNumericalState&&!stats[slot].usedHistory&&stats[slot].converged&&candidateFrames[slot].valid);assert(stats[slot].iterations==measured.iterations&&stats[slot].residualEvaluations==measured.residualEvaluations&&stats[slot].jacobianEvaluations==measured.factorizations);}
 // Deliberately bounded test-only budget checks the existing advisory contract,
 // not a production-control change or a convergence relaxation.
 m.config.maxIterations=1;m.config.relativeTolerance=1e-30;m.config.absoluteTolerance=0;CoherentWarmRegression::failedSeed(control[0]);walls[0].cache.factors.valid=false;for(int i=0;i<24;++i)for(int j=0;j<6;++j)walls[0].state[i][j]=control[0].state[i][j];assert(invoke(0));assert(!stats[0].converged&&!candidateFrames[0].valid&&walls[0].frame.valid&&statuses[0].code==WallCode::Success);assert(counters[0].failures==1&&counters[0].history==0&&counters[0].numericalHistory==3);
 assert(std::vector<double>(accepted,accepted+288)==immutableAccepted);
 m.coherentScratch=nullptr;assert(!gasBoundaryLayerUsesCoherent(m,0));GasBoundaryLayerModelState<float,2>fp32;fp32.coherentEnabled=true;assert(!gasBoundaryLayerUsesCoherent(fp32,0));GasBoundaryLayerModelState<double,3>ns3;ns3.coherentEnabled=true;assert(!gasBoundaryLayerUsesCoherent(ns3,0));
 std::cout<<"lanes="<<lanes<<" knownWarmRoot=PASS repeatedRefresh=PASS pooledHistory=PASS advisoryPublication=PASS legacyDispatch=PASS\n";
}
