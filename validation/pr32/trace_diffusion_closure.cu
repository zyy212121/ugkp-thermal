#include "gasTransport/SpeciesDiffusion.H"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
UGKWP_GAS_HD bool probe(int index,double& traceFlux,double& expected){
 const int trace=index%2,carrier=1-trace;
 const double tiny=::pow(10.,-double(20+20*(index/2)));
 double y[2],g[2],d[2],f[2],energy=0;
 y[carrier]=::nextafter(1.,0.);y[trace]=tiny;
 g[carrier]=1e-12;g[trace]=2*tiny;
 d[carrier]=2e-5;d[trace]=3e-5;
 ugkwp::SpeciesThermoData<double> sp[2];double coef[6]={1000,0,0,800,0,0};
 for(int j=0;j<2;++j){sp[j].coefficientOffset=3*j;sp[j].molarMass=.028;sp[j].minTemperature=1;sp[j].maxTemperature=5000;}
 ugkwp::SpeciesThermoView<double,2> thermo;thermo.species=sp;thermo.coefficients=coef;thermo.coefficientCount=6;
 bool evaluated=false;
 bool ok=ugkwp::correctedSpeciesDiffusion(y,g,1.,d,0.,1.,300.,thermo,f,energy,false,&evaluated);
 // Compute the trace channel directly, before any carrier cancellation.
 const double rawTrace=-d[trace]*g[trace],rawCarrier=-d[carrier]*g[carrier];
 expected=rawTrace-y[trace]*(rawCarrier+rawTrace);traceFlux=f[trace];
 const double expectedEnergy=(coef[3*trace]-coef[3*carrier])*300.*expected;
 return ok&&evaluated&&f[carrier]==-f[trace]
  &&::fabs((traceFlux-expected)/expected)<1e-13
  &&::fabs((energy-expectedEnergy)/expectedEnergy)<1e-12;
}
#if defined(__CUDACC__)
__global__ void run(int* ok,double* actual,double* expected){int i=threadIdx.x;if(i<28)ok[i]=probe(i,actual[i],expected[i]);}
#endif
int main(){
 bool passed=true;
 for(int i=0;i<28;++i){double a,b;bool ok=probe(i,a,b);printf("host %d actual=%.17g expected=%.17g ok=%d\n",i,a,b,int(ok));passed=passed&&ok;}
 if(!passed)return 1;
#if defined(__CUDACC__)
 int* ok;double* actual;double* expected;
 assert(cudaMallocManaged(&ok,28*sizeof(int))==cudaSuccess);assert(cudaMallocManaged(&actual,28*sizeof(double))==cudaSuccess);assert(cudaMallocManaged(&expected,28*sizeof(double))==cudaSuccess);
 run<<<1,32>>>(ok,actual,expected);assert(cudaGetLastError()==cudaSuccess);assert(cudaDeviceSynchronize()==cudaSuccess);
 for(int i=0;i<28;++i){printf("cuda %d actual=%.17g expected=%.17g ok=%d\n",i,actual[i],expected[i],ok[i]);passed=passed&&ok[i];}
 assert(cudaFree(ok)==cudaSuccess);assert(cudaFree(actual)==cudaSuccess);assert(cudaFree(expected)==cudaSuccess);
#endif
 return passed?0:1;
}
