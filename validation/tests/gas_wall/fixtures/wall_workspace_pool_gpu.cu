#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include "common/gasTransport/GasBoundaryLayerModelState.H"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
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
template<int C> void run(int nodes,int mode){
 constexpr int count=5,workers=2;CapturedSstWall fixture;
 fixture.config.nodes=nodes;fixture.config.maxIterations=mode==0?1:0;
 std::vector<In> hi(count,fixture.input);std::vector<Out> ho(count);std::vector<WallStatus> hs(count);std::vector<int> hk(count);std::vector<Diag> hd(count);std::vector<double> hp(count*nodes*9);
 auto in=dev<In>(count);auto c=dev<WallModelConfig<double>>(1);auto w=dev<WallWorkspace<double,2,C>>(workers);auto o=dev<Out>(count);auto s=dev<WallStatus>(count);auto ok=dev<int>(count);auto profile=dev<double>(count*nodes*9);auto d=dev<Diag>(count);auto species=dev<ugkwp::SpeciesThermoData<double>>(2);auto coef=dev<double>(6);
 ck(cudaMemcpy(species,fixture.species,sizeof(fixture.species),cudaMemcpyHostToDevice));ck(cudaMemcpy(coef,fixture.coefficients,sizeof(fixture.coefficients),cudaMemcpyHostToDevice));
 for(int i=0;i<count;++i){hi[i].temperature+=i;hi[i].matching.velocity[0]+=i*.1;hi[i].model.thermo.species=species;hi[i].model.thermo.coefficients=coef;hi[i].diagnostics=d+i;hi[i].model.thermo.diagnostics=d+i;hi[i].diagnosticEntity=100+i;if(mode==2&&i==1)hi[i].temperature=-1;}
 ck(cudaMemcpy(in,hi.data(),sizeof(In)*count,cudaMemcpyHostToDevice));ck(cudaMemcpy(c,&fixture.config,sizeof(fixture.config),cudaMemcpyHostToDevice));ck(cudaMemset(w,0,sizeof(*w)*workers));ck(cudaMemset(o,0,sizeof(Out)*count));ck(cudaMemset(s,0,sizeof(WallStatus)*count));ck(cudaMemset(d,0,sizeof(Diag)*count));ck(cudaMemset(profile,0,sizeof(double)*hp.size()));
 pool<C><<<workers+1,32>>>(in,c,w,o,s,ok,profile,count,workers);ck(cudaGetLastError());ck(cudaDeviceSynchronize());
 ck(cudaMemcpy(ho.data(),o,sizeof(Out)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hs.data(),s,sizeof(WallStatus)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hk.data(),ok,sizeof(int)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hd.data(),d,sizeof(Diag)*count,cudaMemcpyDeviceToHost));ck(cudaMemcpy(hp.data(),profile,sizeof(double)*hp.size(),cudaMemcpyDeviceToHost));
 for(int i=0;i<count;++i){assert(hk[i]==!(mode==2&&i==1));assert(bool(hd[i].fatalClaim)==(mode==2&&i==1));
 printf("C=%d n=%d mode=%d slot=%d ok=%d iter=%d residual=%a heat=%a fatal=%u site=%u",C,nodes,mode,i,hk[i],ho[i].iterations,ho[i].residual,ho[i].conductiveHeatFlux,hd[i].fatalClaim,hd[i].firstFatal.sourceSiteId);
 for(auto& a:hd[i].warnings)printf(" warning=%u",a.count);
 for(int j=0;j<nodes*9;++j)printf(" %a",hp[i*nodes*9+j]);puts("");}
 for(void* p:{(void*)in,(void*)c,(void*)w,(void*)o,(void*)s,(void*)ok,(void*)profile,(void*)d,(void*)species,(void*)coef})ck(cudaFree(p));
}
int main(){for(int mode=0;mode<3;++mode){run<24>(24,mode);run<48>(40,mode);run<128>(65,mode);}puts("PASS capacities, extra worker, reused scratch, per-face capture, diagnostics");}
