// Shared nonlinear wall functions: fixed versus dynamic work assignment and
// phase cycle counts. No timing callback or instrumentation enters production.
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace ugkwp::gaswall;
using W=WallWorkspace<double,2,24>;
struct Result {double heat=0,residual=0;int iterations=0;bool ok=false;};
UGKWP_GAS_HD void configure(int face,bool ewm,CapturedSstWall& f){
 f.config.maxIterations=200;
 if(ewm){f.config.model=BoundaryLayerModel::EquilibriumOde;f.config.nodes=8;f.config.enableSst=false;}
 // A repeating, nonuniform input distribution, shared by both schedulers.
 const double scales[5]={.5,.75,1.,1.25,1.5};
 f.input.matching.velocity[0]*=scales[face%5];
}
__global__ void solveWalls(bool dynamic,bool ewm,int faces,int workers,int* cursor,W* workspace,Result* result){
 const int worker=blockIdx.x;if(worker>=workers)return;detail::CudaWallTeam team;
 int slot=worker;
 while(true){
  if(dynamic){int next=0;if(team.leader())next=atomicAdd(cursor,1);slot=int(team.broadcast(double(next)));}
  if(slot>=faces)break;
  CapturedSstWall f;configure(slot,ewm,f);WallOutput<double,2> output;WallStatus status;
  bool ok=ewm?evaluateEquilibriumOdeWallLayerImpl(f.input,f.config,workspace[worker],output,status,team)
             :evaluateReactingWallLayerImpl(f.input,f.config,workspace[worker],output,status,team);
  if(team.leader())result[slot]={output.conductiveHeatFlux,status.residual,status.iteration,ok};
  team.sync();if(!dynamic)slot+=workers;
 }
}
__global__ void phaseCycles(bool ewm,W* w,unsigned long long* cycles,bool* valid){
 detail::CudaWallTeam team;CapturedSstWall f;configure(2,ewm,f);
 detail::LayerContext<double,2> ctx;detail::makeContext(f.input,f.config,ctx);
 const int B=6,n=f.config.nodes;
 unsigned long long start=clock64();
 for(int i=team.rank();i<n;i+=team.size()){
  double a=pow(double(i)/(n-1),f.config.stretch);w->y[i]=f.input.matchingDistance*a;
  for(int d=0;d<2;++d)w->state[i][d]=a*ctx.edgeU[d];
  w->state[i][2]=a*(f.input.matching.temperature-f.input.temperature);
  w->state[i][3]=f.input.matching.massFraction[ctx.independent[0]];
  w->state[i][4]=f.config.enableSst?sqrt(f.input.matching.k)*a:0;
  w->state[i][5]=f.config.enableSst?a/sqrt(f.input.matching.omega):1;
 }
 team.sync();bool ok=detail::seedSstLayer(f.input,f.config,ctx,*w,team);
 team.sync();if(team.leader())cycles[0]=clock64()-start;
 double norm=0;start=clock64();ok=ok&&detail::allResidual(f.input,f.config,ctx,w->state,w->y,w->residual,norm,team);
 team.sync();if(team.leader())cycles[1]=clock64()-start;
 start=clock64();
 for(int i=team.rank();i<n;i+=team.size())for(int row=0;row<B;++row){w->delta[i][row]=-w->residual[i][row];for(int col=0;col<B;++col)w->lower[i][row][col]=w->diagonal[i][row][col]=w->upper[i][row][col]=0;}
 team.sync();for(int j=team.rank();j<n;j+=team.size())if(!detail::assembleJacobianNode(f.input,f.config,ctx,*w,j))ok=false;
 ok=team.all(ok);team.sync();if(team.leader())cycles[2]=clock64()-start;
 start=clock64();bool solved=false;if(team.leader())solved=detail::solveLayerNewtonSystem(f.config,*w);
 solved=team.broadcast(solved);team.sync();if(team.leader()){cycles[3]=clock64()-start;*valid=ok&&solved;}
}
void ck(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::abort();}}
template<class T>T* device(size_t count){T*p;ck(cudaMalloc(&p,count*sizeof(T)));return p;}
int main(){
 for(bool ewm:{false,true}){
  const int faces=1024,workers=32;W* w=device<W>(workers);Result* result=device<Result>(faces);int*cursor=device<int>(1);
  std::vector<Result> reference(faces),actual(faces);
  for(bool dynamic:{false,true}){
   // Warm the module before measuring; every solve still uses its cold seed.
   ck(cudaMemset(cursor,0,sizeof(int)));solveWalls<<<workers,32>>>(dynamic,ewm,faces,workers,cursor,w,result);ck(cudaDeviceSynchronize());
   cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));double total=0;
   for(int repeat=0;repeat<5;++repeat){ck(cudaMemset(cursor,0,sizeof(int)));ck(cudaEventRecord(a));solveWalls<<<workers,32>>>(dynamic,ewm,faces,workers,cursor,w,result);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));total+=ms;}
   ck(cudaMemcpy(actual.data(),result,faces*sizeof(Result),cudaMemcpyDeviceToHost));
   int low=10000,high=0;double residual=0;
   for(int i=0;i<faces;++i){assert(actual[i].ok);low=std::min(low,actual[i].iterations);high=std::max(high,actual[i].iterations);residual=std::max(residual,actual[i].residual);
    if(dynamic)assert(actual[i].heat==reference[i].heat&&actual[i].residual==reference[i].residual&&actual[i].iterations==reference[i].iterations);}
   if(!dynamic)reference=actual;
   std::cout<<"PASS ewm="<<ewm<<" dynamic="<<dynamic<<" faces="<<faces<<" workers="<<workers<<" mean_ms="<<total/5<<" min_iterations="<<low<<" max_iterations="<<high<<" residual="<<residual<<'\n';ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
  }
  auto* cycles=device<unsigned long long>(4);auto* valid=device<bool>(1);unsigned long long sums[4]{},host[4];
  for(int repeat=0;repeat<20;++repeat){phaseCycles<<<1,32>>>(ewm,w,cycles,valid);ck(cudaDeviceSynchronize());bool ok;ck(cudaMemcpy(&ok,valid,sizeof(bool),cudaMemcpyDeviceToHost));assert(ok);ck(cudaMemcpy(host,cycles,sizeof(host),cudaMemcpyDeviceToHost));for(int i=0;i<4;++i)sums[i]+=host[i];}
  std::cout<<"PHASE ewm="<<ewm<<" initialization_seed_cycles="<<sums[0]/20<<" residual_cycles="<<sums[1]/20<<" jacobian_cycles="<<sums[2]/20<<" linear_solve_cycles="<<sums[3]/20<<'\n';
  cudaFree(cycles);cudaFree(valid);cudaFree(w);cudaFree(result);cudaFree(cursor);
 }
}
