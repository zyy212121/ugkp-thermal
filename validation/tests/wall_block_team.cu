#include <cuda_runtime.h>
#include "gasWall/WallLayerTransport.H"
#include "gasWall/WallCudaTeam.cuh"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <stdexcept>
using namespace ugkwp::gaswall;
using W=WallWorkspace<double,2,16>;
struct Input {WallModelConfig<double> config;W work;};
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void run(Input* inputs,int* result){
 auto& item=inputs[blockIdx.x];detail::CudaWallTeam team;
 bool ok=detail::solveLayerNewtonSystem(item.config,item.work,team);
 if(team.leader())result[blockIdx.x]=ok;
}
__global__ void runTiles(Input* inputs,int* result,int count){
 const int k=blockIdx.x*4+threadIdx.x/8;if(k>=count)return;
 auto& item=inputs[k];detail::CudaWallTile8 team;
 const bool ok=detail::solveLayerNewtonSystem(item.config,item.work,team);
 if(team.leader())result[k]=ok;
}
int main(){
 std::vector<Input> inputs;std::vector<int> expected;
 for(int nodes:{2,8,16})for(bool sst:{false,true})for(double scale:{1e-150,1e-12,1.,1e12,1e150})for(int seed=0;seed<4;++seed){
  Input a{};a.config.nodes=nodes;a.config.enableSst=sst;a.config.model=sst?BoundaryLayerModel::ReactingSst:BoundaryLayerModel::EquilibriumOde;
  const int n=sst?6:4;
  for(int i=0;i<nodes;++i)for(int j=0;j<6;++j)a.work.state[i][j]=.25;
  for(int i=0;i<nodes;++i)for(int row=0;row<n;++row){
   for(int col=0;col<n;++col){
    a.work.diagonal[i][row][col]=scale*(row==col?5.:.03*std::sin(seed+i+row*7+col));
    a.work.lower[i][row][col]=i?scale*.02*std::cos(seed+i+row+col*3):0;
    a.work.upper[i][row][col]=i+1<nodes?scale*.04*std::sin(seed+i+row+col*5):0;
    a.work.delta[i][row]+=a.work.diagonal[i][row][col]*(.7+std::sin(i*6+col));
    if(i)a.work.delta[i][row]+=a.work.lower[i][row][col]*(.7+std::sin((i-1)*6+col));
    if(i+1<nodes)a.work.delta[i][row]+=a.work.upper[i][row][col]*(.7+std::sin((i+1)*6+col));
   }
  }
  // Force nontrivial partial pivots, applying the same row permutation to A,b.
  for(int i=0;i<nodes;++i)for(int row=0;row+1<n;row+=2){
   for(int col=0;col<n;++col){std::swap(a.work.diagonal[i][row][col],a.work.diagonal[i][row+1][col]);std::swap(a.work.lower[i][row][col],a.work.lower[i][row+1][col]);std::swap(a.work.upper[i][row][col],a.work.upper[i][row+1][col]);}
   std::swap(a.work.delta[i][row],a.work.delta[i][row+1]);
  }
  inputs.push_back(a);expected.push_back(1);
 }
 Input singular{};singular.config.nodes=8;singular.config.model=BoundaryLayerModel::EquilibriumOde;inputs.push_back(singular);expected.push_back(0);
 auto cpu=inputs;
 for(size_t i=0;i<cpu.size();++i)if(detail::solveLayerNewtonSystem(cpu[i].config,cpu[i].work)!=bool(expected[i]))return 2;
 Input* device;int* result;check(cudaMalloc(&device,inputs.size()*sizeof(Input)));check(cudaMalloc(&result,inputs.size()*sizeof(int)));
 double maximum=0;
 for(int tile:{32,8}){
 check(cudaMemcpy(device,inputs.data(),inputs.size()*sizeof(Input),cudaMemcpyHostToDevice));
 if(tile==32)run<<<inputs.size(),32>>>(device,result);
 else runTiles<<<(inputs.size()+3)/4,32>>>(device,result,int(inputs.size()));
 auto output=inputs;std::vector<int> status(inputs.size());
 check(cudaMemcpy(output.data(),device,inputs.size()*sizeof(Input),cudaMemcpyDeviceToHost));check(cudaMemcpy(status.data(),result,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
 for(size_t k=0;k<inputs.size();++k){
  if(status[k]!=expected[k])return 3;
  if(!status[k])continue;
  for(int i=0;i<inputs[k].config.nodes;++i)for(int j=0;j<6;++j){
   const double x=output[k].work.delta[i][j],y=cpu[k].work.delta[i][j];
   if(!std::isfinite(x))return 4;maximum=fmax(maximum,fabs(x-y));
   if(x!=y){std::printf("mismatch %zu %d %d %.17g %.17g\n",k,i,j,x,y);return 5;}
   if(j<(inputs[k].config.enableSst?6:4)&&fabs(x-(.7+std::sin(i*6+j)))>1e-13)return 6;
  }
 }
 }
 check(cudaFree(device));check(cudaFree(result));
 std::printf("PASS: %zu scaled, pivoted block systems and singular case; CPU/32-lane/8-lane maximum difference %.17g\n",inputs.size(),maximum);
}
