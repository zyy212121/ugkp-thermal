#include "equilibrium_wall_fixture.H"

void check(int index,const Fixture& f,const WallModelConfig<double>& c,const Workspace& w,
           const WallOutput<double,2>& out,const WallStatus& status,
           const ugkp::diagnostics::DeviceContext& diagnostics,bool ok){
 if(index==13){
  assert(!ok);assert(status.code==WallCode::NonConvergence);
  assert(out.conductiveHeatFlux==123.);
  assert(diagnostics.firstFatal.code==ugkp::diagnostics::Code::InvalidPhysicalState);
  assert(diagnostics.firstFatal.quantityId==ugkp::diagnostics::Quantity::Accuracy);
  return;
 }
 if(!ok){std::cerr<<"index "<<index<<" failed: "<<int(status.code)<<" residual "<<status.residual<<'\n';std::abort();}
 assert(!ugkp::diagnostics::hasFatal(&diagnostics));
 if(f.in.adiabatic){assert(std::abs(out.conductiveHeatFlux)<1e-4);assert(out.traceTemperature>f.in.matching.temperature);}
 if(index==17||index==18)assert(std::abs(out.traceTemperature-203.6)<1e-6);
 if(index==18)return;
 assert(out.residual<=c.relativeTolerance+c.absoluteTolerance);
 assert(out.wallSpeciesFlux[0]==0&&out.wallSpeciesFlux[1]==f.in.massFlux[1]);
 for(int j=0;j<2;++j){assert(out.traceMassFraction[j]>=0&&out.traceMassFraction[j]<=1);assert(std::abs(out.speciesBalanceResidual[j])<1e-8);}
 assert(std::abs(out.traceMassFraction[0]+out.traceMassFraction[1]-1)<1e-14);
 detail::LayerContext<double,2> ctx;assert(detail::makeContext(f.in,c,ctx));
 ctx.equilibriumWallStress=std::sqrt(out.traction[0]*out.traction[0]+out.traction[1]*out.traction[1]+out.traction[2]*out.traction[2]);
 for(int a=0;a<2;++a)for(int d=0;d<3;++d)ctx.equilibriumWallTraction[a]+=out.traction[d]*ctx.tangent[a][d];
 detail::LayerFlux<double,2> first;
 assert(detail::intervalFlux(f.in,c,ctx,w.state[0],w.state[1],w.y[0],w.y[1],first));
 double energyActivity=std::max(1.,std::abs(first.energy));
 for(int d=0;d<3;++d)energyActivity+=std::abs(out.traction[d]*f.in.matching.velocity[d]);
 // Check dimensional equation balances along the profile, independently of
 // the nonlinear solver's normalization and published balance residual.
 for(int i=1;i<c.nodes-1;++i){
  detail::LayerFlux<double,2> flux;assert(detail::intervalFlux(f.in,c,ctx,w.state[i],w.state[i+1],w.y[i],w.y[i+1],flux));
  assert(std::abs(flux.energy-first.energy)<=2e-7*energyActivity);
  for(int j=0;j<2;++j){
   assert(std::abs(flux.species[j]-f.in.massFlux[j])<1e-8);
   assert(std::abs(flux.momentum[j]-first.momentum[j])<=2e-7*std::max(1.,std::abs(first.momentum[j])));
  }
 }
 double energy=out.conductiveHeatFlux,kinetic=0,work=0,mass=0;
 for(int j=0;j<2;++j){mass+=f.in.massFlux[j];energy+=f.in.massFlux[j]*speciesH(j,out.traceTemperature,f.in.model.thermo);}
 for(int d=0;d<3;++d){kinetic+=out.traceVelocity[d]*out.traceVelocity[d];work+=out.traction[d]*out.traceVelocity[d];}
 energy+=.5*mass*kinetic-work;
 assert(std::abs(energy-(first.energy+ctx.mass*ctx.energyReference))<2e-7*std::max(1.,std::abs(energy)));
 struct Adapter {GasBoundaryLayerState<double> gasBoundaryLayer;ugkp::diagnostics::DeviceContext* diagnostics=nullptr;} adapter;
 GasBoundaryLayerExchange<double> exchange;GasBoundaryLayerSstClosure<double> sst;int code=-1;double speciesFlux[2]{};
 auto& wall=adapter.gasBoundaryLayer;wall.enabled=true;wall.count=1;wall.exchange=&exchange;wall.sst=&sst;wall.status=&code;wall.speciesFlux=speciesFlux;
 const double area=.003;
 assert(publishGasBoundaryLayerOutput(adapter,0,f.in,out,area,0.));
 assert(exchange.ready&&code==0&&exchange.mass==-area*mass);
 assert(speciesFlux[0]==0&&speciesFlux[1]==-area*f.in.massFlux[1]);
 assert(std::abs(exchange.energy+area*energy)<1e-12*std::max(1.,std::abs(exchange.energy)));
 assert(sst.kFluxCount==0&&exchange.k==0);
 if(index==14){const double rho=f.in.pressure/(universalGasConstant<double>()/.028*300.);
  const double expected=-std::expm1(-f.in.massFlux[1]*f.in.matchingDistance/(rho*1e-4));
  assert(std::abs(out.traceMassFraction[1]-expected)<2e-6);
  assert(std::abs(out.conductiveHeatFlux)<1e-7);
 }
}
#if defined(__CUDACC__)
__global__ void runDevice(int index,Workspace* w,WallOutput<double,2>* out,
 WallStatus* status,ugkp::diagnostics::DeviceContext* diagnostics,int* ok){
 Fixture f;WallModelConfig<double> c;configure(index,f,c);f.in.diagnostics=diagnostics;
 out->conductiveHeatFlux=123.;*ok=evaluateWallModel(f.in,c,*w,*out,*status);
}
void cudaCheck(cudaError_t result){if(result!=cudaSuccess){std::cerr<<cudaGetErrorString(result)<<'\n';std::abort();}}
#endif
int main(){
 std::cout<<std::setprecision(17);
 for(int index=0;index<20;++index){
  Fixture f;WallModelConfig<double> c;configure(index,f,c);
  Workspace w;WallOutput<double,2> out;WallStatus status;ugkp::diagnostics::DeviceContext diagnostics{};
  f.in.diagnostics=&diagnostics;out.conductiveHeatFlux=123.;
  bool ok=evaluateWallModel(f.in,c,w,out,status);check(index,f,c,w,out,status,diagnostics,ok);
#if defined(__CUDACC__)
  Workspace* dw;WallOutput<double,2>* dout;WallStatus* ds;ugkp::diagnostics::DeviceContext* dd;int* dok;
  cudaCheck(cudaMallocManaged(&dw,sizeof(*dw)));cudaCheck(cudaMallocManaged(&dout,sizeof(*dout)));
  cudaCheck(cudaMallocManaged(&ds,sizeof(*ds)));cudaCheck(cudaMallocManaged(&dd,sizeof(*dd)));cudaCheck(cudaMallocManaged(&dok,sizeof(*dok)));
  *dw={};*dout={};*ds={};*dd={};*dok=0;
  runDevice<<<1,1>>>(index,dw,dout,ds,dd,dok);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
  check(index,f,c,*dw,*dout,*ds,*dd,*dok);
  if(ok){assert(std::abs(dout->conductiveHeatFlux-out.conductiveHeatFlux)<2e-7*std::max(1.,std::abs(out.conductiveHeatFlux)));}
  cudaCheck(cudaFree(dw));cudaCheck(cudaFree(dout));cudaCheck(cudaFree(ds));cudaCheck(cudaFree(dd));cudaCheck(cudaFree(dok));
#endif
  std::cout<<"index="<<index<<" ok="<<ok<<" iterations="<<status.iteration<<" residual="<<status.residual<<'\n';
 }
 return 0;
}
