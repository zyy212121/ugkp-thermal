#include "gasTransport/MolecularTransportDevice.cuh"
#include "gasTransport/GasStateView.H"
#include <cstdio>
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_R(x) double(x)
struct State:ugkwp::GasStateView<double,double,int>{ugkwp::GasSpeciesState<double,2> gasSpecies;};
#include "gasTransport/GasMolecularConductivity.cuh"
__global__ void probe(double*out){
 int i=threadIdx.x;if(i>=3)return;double T=300+400*i,rho=1,partial[2]={.3,.7};
 ugkwp::SpeciesThermoData<double> species[2]{};double coefficients[6]={1000,.1,0,1200,.2,0};
 for(int k=0;k<2;++k){species[k].model=ugkwp::SpeciesThermoModel::LinearCp;species[k].molarMass=.028;species[k].minTemperature=100;species[k].maxTemperature=3000;species[k].coefficientOffset=3*k;}
 State s;s.nCells=1;s.rho=&rho;s.gasMu=1.8e-5;s.gasCp=3.5;s.gasPrClamped=.7;s.gasThermalConductivity=-1;
 s.gasSpecies.mode=ugkwp::GasMode::MixtureFrozen;s.gasSpecies.rho=partial;s.gasSpecies.thermo.species=species;s.gasSpecies.thermo.coefficients=coefficients;s.gasSpecies.thermo.coefficientCount=6;
 double*row=out+5*i;row[0]=T;row[1]=molecularGasConductivity(s,0,T);s.gasCp=1004.5;row[2]=molecularGasConductivity(s,0,T);
 s.gasThermalConductivity=.123;row[3]=molecularGasConductivity(s,0,T);s.gasThermalConductivity=-1;s.gasSpecies.mode=static_cast<ugkwp::GasMode>(0);row[4]=molecularGasConductivity(s,0,T);
}
int main(){double*out=nullptr,host[15];if(cudaMalloc(&out,sizeof(host))!=cudaSuccess)return 1;probe<<<1,32>>>(out);if(cudaGetLastError()!=cudaSuccess||cudaMemcpy(host,out,sizeof(host),cudaMemcpyDeviceToHost)!=cudaSuccess)return 2;for(int i=0;i<3;++i){for(int j=0;j<5;++j)printf("%.17g ",host[5*i+j]);puts("");}cudaFree(out);}
