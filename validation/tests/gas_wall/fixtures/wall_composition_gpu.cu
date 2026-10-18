#include "wall_composition_cases.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
using Diag=ugkp::diagnostics::DeviceContext;
static void check(cudaError_t code){if(code!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(code));exit(1);}}
template<class T>T* allocate(int count=1){T*p;check(cudaMallocManaged(&p,count*sizeof(T)));return p;}
__global__ void solve(CompositionWallCase* f,WallWorkspace<double,2,16>* w,WallOutput<double,2>* out,Diag* diag,int* ok){int i=blockIdx.x;detail::CudaWallTeam team;auto& in=f[i].in;if(team.leader()){in.model.thermo.species=f[i].sp;in.model.thermo.coefficients=f[i].coef;in.model.thermo.elementComposition=f[i].elements;in.model.mechanism.reactions=f[i].rx;in.model.mechanism.reactants=f[i].re;in.model.mechanism.products=f[i].pr;in.model.mechanism.stoichiometricBasis=f[i].basis;in.diagnostics=diag;in.model.thermo.diagnostics=diag;}
 team.sync();WallModelConfig<double> c;c.nodes=16;c.maxIterations=80;c.relativeTolerance=1e-8;c.enableSst=false;WallStatus status;bool v=evaluateReactingWallLayerImpl(in,c,w[i],out[i],status,team);if(team.leader())ok[i]=v;}
template<int N>__global__ void multi(int* passed){detail::CudaWallTeam team;__shared__ double x[65][N+4];for(int i=team.rank();i<65;i+=team.size()){for(int j=0;j<N+4;++j)x[i][j]=0;for(int j=0;j<N-1;++j)x[i][3+j]=(i%3==0?-1.:1.)*(j+1)/(N*.2);}
 team.sync();bool good=true;for(int i=team.rank();i<65;i+=team.size())good=detail::feasibleLayerComposition<double,N>(x[i])&&good;team.sync();for(int i=team.rank();i<65;i+=team.size()){double sum=0;for(int j=0;j<N-1;++j){good=good&&x[i][3+j]>=0&&x[i][3+j]<=1;sum+=x[i][3+j];}good=good&&sum<=1;}bool all=team.all(good);if(team.leader())*passed=all;}
int main(){constexpr int count=32;auto f=allocate<CompositionWallCase>(count);auto w=allocate<WallWorkspace<double,2,16>>(count);auto out=allocate<WallOutput<double,2>>(count);auto d=allocate<Diag>();auto ok=allocate<int>(count);*d={};for(int i=0;i<count;++i){f[i]=CompositionWallCase{};configureCompositionCase(f[i],i%2?25:11);f[i].in.integrateOwnerKSource=false;f[i].in.diagnosticEntity=i;w[i]={};out[i]={};ok[i]=0;}
 solve<<<count,32>>>(f,w,out,d,ok);check(cudaGetLastError());check(cudaDeviceSynchronize());assert(!d->fatalClaim);for(int i=0;i<count;++i){assert(ok[i]&&out[i].residual<=1.01e-8);for(int j=0;j<16;++j)assert(w[i].state[j][3]>=0&&w[i].state[j][3]<=1);}printf("PASS real chemistry case11/25 faces=%d residuals=%.17g,%.17g\n",count,out[0].residual,out[1].residual);
 for(int j=0;j<3;++j){*ok=0;if(j==0)multi<3><<<1,32>>>(ok);else if(j==1)multi<10><<<1,32>>>(ok);else multi<32><<<1,32>>>(ok);check(cudaGetLastError());check(cudaDeviceSynchronize());assert(*ok);}puts("PASS Ns3/Ns10/Ns32 cooperative composition mapping, 65 nodes across lanes");check(cudaFree(f));check(cudaFree(w));check(cudaFree(out));check(cudaFree(d));check(cudaFree(ok));}
