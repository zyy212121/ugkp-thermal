#include "equilibrium_wall_fixture.H"
#include "gasTransport/GasCoherentBoundaryLayer.H"
int main(){
 Fixture f;WallModelConfig<double> c;configure(17,f,c);c.model=BoundaryLayerModel::ReactingSst;c.nodes=8;c.maxIterations=100;
 {
  WallWorkspace<double,2,24> work;gaswall::coherent::Scratch<24> scratch;gaswall::coherent::Cache<24> cache;
  WallOutput<double,2> output;gaswall::coherent::SolveStats stats;
  gaswall::coherent::ScalarEngine<24> engine(scratch);
  assert(engine.solve(f.in,c,work,cache,output,stats));
  assert(stats.physicalResidual<=c.relativeTolerance+c.absoluteTolerance);
  assert(std::abs(output.traceTemperature-203.6)<1e-6&&std::abs(output.conductiveHeatFlux)<1e-5);
 }
 WallWorkspace<double,2,24> w;gaswall::coherent::Scratch<24> scratch;
 gaswall::coherent::PersistentWallState<24> persistent;WallOutput<double,2> out;WallStatus status;gaswall::WallReferenceStats stats;
 for(bool ad:{true,false,true}){f.in.adiabatic=ad;
  bool ok=evaluateGasCoherentWallWorkspace(f.in,c,w,scratch,persistent,nullptr,out,status,stats);
  std::cout<<"adiabatic="<<ad<<" ok="<<ok<<" residual="<<stats.residual<<" Tw="<<out.traceTemperature<<" q="<<out.conductiveHeatFlux<<"\n";
  assert(ok&&stats.converged);if(ad){assert(std::abs(out.conductiveHeatFlux)<1e-5);assert(std::abs(out.traceTemperature-203.6)<1e-6);}else assert(std::abs(out.traceTemperature-300)<1e-10);
 }
}
