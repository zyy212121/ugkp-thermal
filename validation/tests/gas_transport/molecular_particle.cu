#include "gasTransport/MolecularTransportDevice.cuh"
#include "gasTransport/GasStateView.H"
#include "../../../applications/gasUGKP/private_backend/GpuDragModels.cuh"
#include "gasNumerics/GpuParticlePhysicsAlgebra.cuh"
#include <cstdio>
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) double(x)
#define GPU_OPERATOR_TINY(x) double(x)
#define GPU_OPERATOR_THERMAL 0
constexpr double OfSmall=1e-15;
__device__ double clampMin(double a,double b){return fmax(a,b);}
__device__ double clampRange(double a,double b,double c){return fmin(fmax(a,b),c);}
__device__ double finiteOr(double a,double b){return isfinite(a)?a:b;}
struct DeviceState:ugkwp::GasStateView<double,double,int> {
 double* couplingTgasOld=nullptr;double rhoSolid=2000,dragParameter0=0,dragParameter1=0,dragParameter2=0,dragParameter3=0;
 double gasPrOneThird=0,TpMin=1,TpMax=5000;double *thetaDragAlpha=nullptr,*pTheta=nullptr;
};
#include "gasTransport/GasMolecularConductivity.cuh"
#include "GpuCellDragRate.cuh"
#include "GpuParticleGasHeatRelaxation.cuh"
__global__ void run(ugkwp::MolecularViscosityView<double> model,double* out){
 int i=threadIdx.x;if(i>=3)return;
 double T=300.+300*i;DeviceState s;s.gasViscosity=model;s.gasMu=1.8e-5;s.gasCp=1000;s.gasPrClamped=.72;s.gasPrOneThird=cbrt(.72);s.couplingTgasOld=&T;
 const double mu=ugkwp::gasCellMolecularViscosityAt(s,0,T),re=1.2*1e-4*3/mu;
 double* row=out+6*i;row[0]=T;row[1]=mu;row[2]=cellDragInverseTime(s,0,1.2,1.,1e-4,3.,ugkwpGpuDrag::SchillerNaumannDeviceDrag{});
 row[3]=particleTemperatureAfterGasRelaxation(s,0,250.,T,re,1e-4,2e6,1e-4);
 row[4]=molecularGasConductivity(s,0,T);s.gasThermalConductivity=.123;row[5]=molecularGasConductivity(s,0,T);
}
int main(){auto cfg=ugkwp::parseMolecularTransport("viscosityModel sutherland; species (gas); sutherland { gas {muRef 1.8e-5;TRef 300;S 110.4;molarMass .028;} }");
 ugkwp::MolecularTransportDevice model;if(model.initialize(cfg)!=cudaSuccess)return 1;
 double *out=nullptr,host[18];if(cudaMalloc(&out,sizeof(host))!=cudaSuccess)return 2;
 run<<<1,32>>>(model.view(),out);if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(host,out,sizeof(host),cudaMemcpyDeviceToHost)!=cudaSuccess)return 3;
 for(int i=0;i<3;++i){for(int j=0;j<6;++j)printf("%.17g ",host[6*i+j]);puts("");}cudaFree(out);
}
