#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/WallCudaTeam.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
using namespace ugkwp::gaswall;
using Input=WallInput<double,2>;using Work=WallWorkspace<double,2,24>;
using Output=WallOutput<double,2>;using Diag=ugkp::diagnostics::DeviceContext;
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(1);}}
template<class T>T* alloc(int n=1){T*p;check(cudaMallocManaged(&p,sizeof(T)*n));return p;}
__global__ void solve(Input* in,WallModelConfig<double>* config,Work* w,Output* out,WallStatus* status,int* ok){int i=blockIdx.x;detail::CudaWallTeam team;bool v=evaluateReactingWallLayerImpl(in[i],*config,w[i],out[i],status[i],team);if(team.leader())ok[i]=v;}
int main(){constexpr int count=32;CapturedSstWall capture;auto in=alloc<Input>(count);auto c=alloc<WallModelConfig<double>>();auto w=alloc<Work>(count);auto out=alloc<Output>(count);auto status=alloc<WallStatus>(count);auto ok=alloc<int>(count);auto d=alloc<Diag>();auto species=alloc<ugkwp::SpeciesThermoData<double>>(2);auto coeff=alloc<double>(6);for(int i=0;i<2;++i)species[i]=capture.species[i];for(int i=0;i<6;++i)coeff[i]=capture.coefficients[i];
for(int mode=0;mode<3;++mode){*d={};*c=capture.config;if(mode==1)c->maxIterations=0;for(int i=0;i<count;++i){in[i]=capture.input;in[i].model.thermo.species=species;in[i].model.thermo.coefficients=coeff;in[i].diagnostics=d;in[i].model.thermo.diagnostics=d;in[i].diagnosticEntity=53683+i;if(mode==2)in[i].temperature=-1;w[i]={};out[i]={};status[i]={};ok[i]=-1;}
solve<<<count,32>>>(in,c,w,out,status,ok);check(cudaGetLastError());check(cudaDeviceSynchronize());for(int i=0;i<count;++i){if(mode==2){assert(!ok[i]);continue;}assert(ok[i]);assert(w[i].state[0][5]==0);for(int j=1;j<c->nodes;++j)assert(w[i].state[j][5]>0);if(mode==0){assert(out[i].residual<=c->absoluteTolerance+c->relativeTolerance);assert(out[i].iterations==7);assert(std::abs(out[i].conductiveHeatFlux-653.76765672560316)<1e-6);}}
if(mode==0)assert(!d->fatalClaim);if(mode==1){assert(!d->fatalClaim&&d->warnings[1].count>=count);assert(d->warnings[1].sample.sourceSiteId==2422);}if(mode==2){assert(d->fatalClaim&&d->firstFatal.observed==-1);assert(d->firstFatal.entityIndex>=53683&&d->firstFatal.entityIndex<53683+count);}
printf("mode=%d faces=%d fatal=%u warnings=%llu iterations=%d residual=%.17g heat=%.17g\n",mode,count,(unsigned)d->fatalClaim,(unsigned long long)d->warnings[1].count,out[0].iterations,out[0].residual,out[0].conductiveHeatFlux);
}check(cudaFree(in));check(cudaFree(c));check(cudaFree(w));check(cudaFree(out));check(cudaFree(status));check(cudaFree(ok));check(cudaFree(d));check(cudaFree(species));check(cudaFree(coeff));puts("PASS cooperative captured wall + warning budget + concurrent physical error");}
