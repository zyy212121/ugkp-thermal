#include "validation/tests/gas_wall/fixtures/sst_small_k_column_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
using namespace ugkwp::gaswall;
using In=WallInput<double,2>;using Out=WallOutput<double,2>;using Diag=ugkp::diagnostics::DeviceContext;
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}
template<class T>T* dev(int n){T* p;ck(cudaMalloc(&p,sizeof(T)*n));return p;}
template<int C> __global__ __launch_bounds__(32) void pool(In* input,WallModelConfig<double>* config,WallWorkspace<double,2,C>* work,Out* output,WallStatus* status,int* ok,double* profile,int count,int workers){
 const int worker=blockIdx.x;detail::CudaWallTeam team;
 if(worker>=workers)return;
 for(int slot=worker;slot<count;slot+=workers){
  bool result=ugkwp::evaluateGasBoundaryLayerWorkspace<true>(input[slot],*config,work[worker],output[slot],status[slot],profile+slot*config->nodes*9,team);
  if(team.leader())ok[slot]=result;
 }
}

std::vector<In> cases;
template<int C> void run(int nodes,bool tiny,int repeats){
 constexpr int count=16,workers=16;CapturedSingularWall small;CapturedSstWall fixture;
 fixture.config.nodes=nodes;fixture.config.maxIterations=60;
 std::vector<In> hi(count);for(int i=0;i<count;++i)hi[i]=tiny?small.input:cases[i%cases.size()];
 std::vector<Out> ho(count);std::vector<WallStatus> hs(count);std::vector<int> hk(count);std::vector<Diag> hd(count);
 auto in=dev<In>(count);auto c=dev<WallModelConfig<double>>(1);auto w=dev<WallWorkspace<double,2,C>>(workers);auto o=dev<Out>(count);auto st=dev<WallStatus>(count);auto ok=dev<int>(count);auto profile=dev<double>(count*nodes*9);auto d=dev<Diag>(count);auto species=dev<ugkwp::SpeciesThermoData<double>>(2);auto coef=dev<double>(6);
 ck(cudaMemcpy(species,fixture.species,sizeof(fixture.species),cudaMemcpyHostToDevice));ck(cudaMemcpy(coef,fixture.coefficients,sizeof(fixture.coefficients),cudaMemcpyHostToDevice));
 for(int i=0;i<count;++i){hi[i].integrateOwnerKSource=hi[i].volumeSstClosure=false;hi[i].model.thermo.species=species;hi[i].model.thermo.coefficients=coef;hi[i].diagnostics=d+i;hi[i].model.thermo.diagnostics=d+i;}
 ck(cudaMemcpy(in,hi.data(),sizeof(In)*count,cudaMemcpyHostToDevice));ck(cudaMemcpy(c,&fixture.config,sizeof(fixture.config),cudaMemcpyHostToDevice));ck(cudaMemset(w,0,sizeof(*w)*workers));ck(cudaMemset(d,0,sizeof(Diag)*count));
 for(int repeat=0;repeat<1;++repeat)pool<C><<<workers,32>>>(in,c,w,o,st,ok,profile,count,workers);ck(cudaDeviceSynchronize());
 cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));ck(cudaEventRecord(a));
 for(int repeat=0;repeat<repeats;++repeat)pool<C><<<workers,32>>>(in,c,w,o,st,ok,profile,count,workers);
 ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
 ck(cudaMemcpy(ho.data(),o,sizeof(Out)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hk.data(),ok,sizeof(int)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hd.data(),d,sizeof(Diag)*count,cudaMemcpyDeviceToHost));
 double iter=0,residual=0;int converged=0,fatal=0;for(int i=0;i<count;++i){iter+=ho[i].iterations;residual=std::max(residual,ho[i].residual);converged+=hk[i]&&ho[i].residual<=1.01e-8;fatal+=hd[i].fatalClaim!=0;}
 printf("%d,%d,%d,%d,%d,%.17g,%.17g,%.17g\n",nodes,int(tiny),count,converged,fatal,iter/count,residual,double(ms)/repeats);fflush(stdout);
 for(void* p:{(void*)in,(void*)c,(void*)w,(void*)o,(void*)st,(void*)ok,(void*)profile,(void*)d,(void*)species,(void*)coef})ck(cudaFree(p));ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
}
int main(int argc,char**argv){CapturedSstWall fixture;cases.push_back(fixture.input);
 for(int arg=1;arg<argc;++arg){std::ifstream f(argv[arg]);std::string line;std::getline(f,line);while(std::getline(f,line)){
 std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);int face,owner;double L,y,p,T,u,v,w,k,omega;if(!(row>>face>>owner>>L>>y>>p>>T>>u>>v>>w>>k>>omega))return 2;
 In i=fixture.input;i.matchingDistance=L;i.ownerDistance=y;i.pressure=p;i.matching.temperature=T;i.matching.velocity[0]=u;i.matching.velocity[1]=v;i.matching.velocity[2]=w;i.matching.k=k;i.matching.omega=omega;cases.push_back(i);}}
 puts("nodes,tiny,count,converged,fatal,mean_iterations,max_residual,milliseconds_16_faces");
 for(bool tiny:{false,true}){run<24>(24,tiny,3);run<48>(48,tiny,3);run<128>(128,tiny,3);}return 0;
}
