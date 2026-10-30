#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallAcceptedHistory.H"
#include <cassert>
#include <iostream>
#include <cmath>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
using namespace ugkwp::gaswall;
struct Result {double coldHeat=0,warmHeat=0,residual=0;int coldIterations=0,warmIterations=0;bool valid=false;};
UGKWP_GAS_HD void run(bool equilibrium,WallWorkspace<double,2,24>& w,Result& result){
 CapturedSstWall f;if(equilibrium){f.config.model=BoundaryLayerModel::EquilibriumOde;f.config.enableSst=false;f.config.maxIterations=200;}
 WallOutput<double,2> out{};WallStatus status{};detail::SerialWallTeam team;
 bool ok=equilibrium?evaluateEquilibriumOdeWallLayerImpl(f.input,f.config,w,out,status):evaluateReactingWallLayerImpl(f.input,f.config,w,out,status);
 if(!ok)return;
 result.coldHeat=out.conductiveHeatFlux;result.coldIterations=out.iterations;
 double history[24*6];for(int i=0;i<24;++i)for(int j=0;j<6;++j)history[i*6+j]=w.state[i][j];
 double stress=0;for(int d=0;d<3;++d)stress+=out.traction[d]*out.traction[d];stress=std::sqrt(stress);
 ok=equilibrium?evaluateEquilibriumOdeWallLayerImpl(f.input,f.config,w,out,status,team,history,stress)
               :evaluateReactingWallLayerImpl(f.input,f.config,w,out,status,team,history);
 result.valid=ok;result.warmHeat=out.conductiveHeatFlux;result.warmIterations=out.iterations;result.residual=out.residual;
}
#ifdef __CUDACC__
__global__ void deviceRun(bool equilibrium,WallWorkspace<double,2,24>* w,Result* result){run(equilibrium,*w,*result);}
#endif
int main(){
 for(bool equilibrium:{false,true}){
  WallWorkspace<double,2,24> w{};Result r;run(equilibrium,w,r);
  assert(r.valid&&r.warmIterations==0&&r.coldIterations>0&&r.residual<=1.01e-8);
  assert(std::abs(r.warmHeat-r.coldHeat)<1e-7*std::max(1.,std::abs(r.coldHeat)));
#ifdef __CUDACC__
  WallWorkspace<double,2,24>* dw;Result* dr;
  assert(cudaMallocManaged(&dw,sizeof(*dw))==cudaSuccess);assert(cudaMallocManaged(&dr,sizeof(*dr))==cudaSuccess);*dw={};*dr={};
  deviceRun<<<1,1>>>(equilibrium,dw,dr);assert(cudaDeviceSynchronize()==cudaSuccess);
  assert(dr->valid&&dr->warmIterations==0&&dr->coldIterations>0&&dr->residual<=1.01e-8);
  assert(std::abs(dr->warmHeat-dr->coldHeat)<1e-7*std::max(1.,std::abs(dr->coldHeat)));
  assert(cudaFree(dw)==cudaSuccess);assert(cudaFree(dr)==cudaSuccess);
#endif
  std::cout<<"PASS model="<<equilibrium<<" cold="<<r.coldIterations<<" warm="<<r.warmIterations<<" heat="<<r.warmHeat<<" residual="<<r.residual<<'\n';
 }
 CapturedSstWall f;detail::LayerContext<double,2> ctx;assert(detail::makeContext(f.input,f.config,ctx));
 auto frame=historyFrame(f.input,f.config,ctx);assert(compatibleHistory(frame,frame));
 for(int change=0;change<7;++change){auto other=frame;
  if(change==0)other.model+=1;if(change==1)other.nodes+=1;if(change==2)other.distance*=2;
  if(change==3)other.speciesOrder+=1;if(change==4)other.thermoHash+=1;if(change==5)other.normal[0]=1;
  if(change==6)other.turbulentPrandtl*=2;
  assert(!compatibleHistory(frame,other));
 }
}
