// Reuse the existing dimensional mass/species/momentum/energy checks.
#define main originalEquilibriumRegression
#include "../pr32/equilibrium_co2_wall.cu"
#undef main
UGKWP_GAS_HD void localFixture(int index,Fixture& f,WallModelConfig<double>& c){
 configure(index%12,f,c);c.maxIterations=200;
 if(index>=12){
  f.in.pressure=101325;f.in.matching.temperature=300;f.in.temperature=280;
  f.in.matching.massFraction[0]=.6;f.in.matching.massFraction[1]=.4;
  f.in.massFlux[0]=-.6*.002;f.in.massFlux[1]=-.4*.002;
  f.in.velocity[0]=20;f.in.velocity[2]=-10;
  f.in.matching.velocity[0]=index==12?120:80;
  f.in.matching.velocity[2]=index==12?-10:70;
 }
}
#ifdef __CUDACC__
#include "common/gasWall/WallCudaTeam.cuh"
__global__ void correctedDevice(int index,Workspace* w,WallOutput<double,2>* out,
 WallStatus* status,ugkp::diagnostics::DeviceContext* diagnostics,int* ok){
 Fixture f;WallModelConfig<double> c;localFixture(index,f,c);c.localStressCorrection=true;
 c.maxIterations=200;f.in.diagnostics=diagnostics;
 *ok=evaluateWallModel(f.in,c,*w,*out,*status);
}
__global__ void correctedTeamDevice(int index,Workspace* w,WallOutput<double,2>* out,
 WallStatus* status,ugkp::diagnostics::DeviceContext* diagnostics,int* ok){
 Fixture f;WallModelConfig<double> c;localFixture(index,f,c);c.localStressCorrection=true;
 f.in.diagnostics=diagnostics;detail::CudaWallTeam team;WallOutput<double,2> result;WallStatus state;
 const bool accepted=evaluateEquilibriumOdeWallLayerImpl(f.in,c,*w,result,state,team);
 if(team.leader()){*ok=accepted;*out=result;*status=state;}
}

#endif
int main(){
 // Independent local constitutive value: the prefactor changes, damping does not.
 for(double j:{-.002,0.,.002,.2}){
  double rho=1.2,wall=2.,u=300.,y=.0004,mu=1.8e-5,kappa=.41,A=17.;
  double local=std::abs(wall+j*u),d=-std::expm1(-y*std::sqrt(rho*wall)/(mu*A));
  double expected=kappa*y*std::sqrt(rho*local)*d*d;
  double actual=detail::equilibriumLocalStressViscosity(rho,wall,local,y,mu,kappa,A);
  assert(actual==expected);
 }
 double rotatedHeat=0,rotatedTraction=0;
 for(int index=0;index<14;++index){
  Fixture f;WallModelConfig<double> c;localFixture(index,f,c);c.maxIterations=200;
  Workspace baseline,corrected;WallOutput<double,2> a,b;WallStatus sa,sb;
  ugkp::diagnostics::DeviceContext diagnostics{};f.in.diagnostics=&diagnostics;
  c.localStressCorrection=false;bool ok=evaluateWallModel(f.in,c,baseline,a,sa);assert(ok);
  c.localStressCorrection=true;ok=evaluateWallModel(f.in,c,corrected,b,sb);
  if(index<12)check(index,f,c,corrected,b,sb,diagnostics,ok);
  else {
   assert(ok&&b.residual<=c.relativeTolerance+c.absoluteTolerance);
   detail::LayerContext<double,2> ctx;assert(detail::makeContext(f.in,c,ctx));
   for(int a=0;a<2;++a)for(int d=0;d<3;++d)ctx.equilibriumWallTraction[a]+=b.traction[d]*ctx.tangent[a][d];
   ctx.equilibriumWallStress=std::hypot(ctx.equilibriumWallTraction[0],ctx.equilibriumWallTraction[1]);
   detail::LayerFlux<double,2> first;assert(detail::intervalFlux(f.in,c,ctx,corrected.state[0],corrected.state[1],corrected.y[0],corrected.y[1],first));
   for(int n=1;n<c.nodes-1;++n){detail::LayerFlux<double,2> flux;
    assert(detail::intervalFlux(f.in,c,ctx,corrected.state[n],corrected.state[n+1],corrected.y[n],corrected.y[n+1],flux));
    assert(std::abs(flux.energy-first.energy)<1e-6*std::max(1.,std::abs(first.energy)));
    for(int d=0;d<2;++d){assert(std::abs(flux.momentum[d]-first.momentum[d])<1e-7*std::max(1.,std::abs(first.momentum[d])));assert(std::abs(flux.species[d]-f.in.massFlux[d])<1e-9);}
   }
  }
  if(index==12){rotatedHeat=b.conductiveHeatFlux;rotatedTraction=b.traction[0];}
  if(index==13){
   assert(std::abs(b.conductiveHeatFlux-rotatedHeat)<2e-7*std::max(1.,std::abs(rotatedHeat)));
   assert(std::abs(b.traction[0]-.6*rotatedTraction)<2e-7*std::max(1.,std::abs(rotatedTraction)));
   assert(std::abs(b.traction[2]-.8*rotatedTraction)<2e-7*std::max(1.,std::abs(rotatedTraction)));
  }
  if(f.in.massFlux[1]==0){
   assert(a.conductiveHeatFlux==b.conductiveHeatFlux&&a.iterations==b.iterations&&a.residual==b.residual);
   for(int d=0;d<3;++d)assert(a.traction[d]==b.traction[d]);
   for(int i=0;i<c.nodes;++i)for(int v=0;v<6;++v)assert(baseline.state[i][v]==corrected.state[i][v]);
  }
#ifdef __CUDACC__
  Workspace* dw;WallOutput<double,2>* dout;WallStatus* ds;ugkp::diagnostics::DeviceContext* dd;int* dok;
  cudaCheck(cudaMallocManaged(&dw,sizeof(*dw)));cudaCheck(cudaMallocManaged(&dout,sizeof(*dout)));
  cudaCheck(cudaMallocManaged(&ds,sizeof(*ds)));cudaCheck(cudaMallocManaged(&dd,sizeof(*dd)));cudaCheck(cudaMallocManaged(&dok,sizeof(*dok)));
  *dw={};*dout={};*ds={};*dd={};*dok=0;
  correctedDevice<<<1,1>>>(index,dw,dout,ds,dd,dok);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
  if(index<12)check(index,f,c,*dw,*dout,*ds,*dd,*dok);else assert(*dok&&dout->residual<=c.relativeTolerance+c.absoluteTolerance);
  assert(std::abs(dout->conductiveHeatFlux-b.conductiveHeatFlux)<2e-7*std::max(1.,std::abs(b.conductiveHeatFlux)));
  *dw={};*dout={};*ds={};*dd={};*dok=0;
  correctedTeamDevice<<<1,32>>>(index,dw,dout,ds,dd,dok);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
  assert(*dok&&dout->residual<=c.relativeTolerance+c.absoluteTolerance);
  assert(std::abs(dout->conductiveHeatFlux-b.conductiveHeatFlux)<2e-7*std::max(1.,std::abs(b.conductiveHeatFlux)));
  for(int d=0;d<3;++d)assert(std::abs(dout->traction[d]-b.traction[d])<2e-7*std::max(1.,std::abs(b.traction[d])));
  cudaCheck(cudaFree(dw));cudaCheck(cudaFree(dout));cudaCheck(cudaFree(ds));cudaCheck(cudaFree(dd));cudaCheck(cudaFree(dok));
#endif
  std::cout<<std::setprecision(17)<<"PASS case="<<index<<" baseline_heat="<<a.conductiveHeatFlux<<" local_heat="<<b.conductiveHeatFlux<<" iterations="<<b.iterations<<" residual="<<b.residual<<'\n';
 }
}
