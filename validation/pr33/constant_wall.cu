#include "gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
using namespace ugkwp;using namespace ugkwp::gaswall;
struct Result{bool ok=false;double heat=0,residual=0;};
UGKWP_GAS_HD Result solve(double mass){
 SpeciesThermoData<double> species[2]{};double coefficients[6]={1000,0,0,1000,0,0};
 for(int k=0;k<2;++k){species[k].coefficientOffset=3*k;species[k].molarMass=.028;species[k].minTemperature=1;species[k].maxTemperature=5000;}
 WallInput<double,2> in;in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=6;
 in.model.viscosity=2e-5;in.model.conductivity=2e-5*1000/.72;
 in.model.diffusivity[0]=in.model.diffusivity[1]=1e-4;
 in.normal[1]=1;in.temperature=in.matching.temperature=300;in.pressure=101325;
 in.matchingDistance=.01;in.ownerDistance=.004;in.matching.massFraction[0]=1;
 in.massFlux[0]=mass;in.matching.velocity[0]=100;
 const double rho=in.pressure/(universalGasConstant<double>()/.028*300);
 in.matching.velocity[1]=mass/rho;
 WallModelConfig<double> c;c.model=BoundaryLayerModel::ConstantTransport;
 WallOutput<double,2> out;WallStatus status;
 bool ok=evaluateConstantTransportWall(in,c,out,status);
 return {ok,out.conductiveHeatFlux,out.residual};
}
#ifdef __CUDACC__
__global__ void run(double mass,Result* result){*result=solve(mass);}
#endif
int main(){
 const double mass[]={0,-.1,-.2,-.4,-2,2};
 const double expected[]={-10,-500,-1000,-2000,-10000,-1.14312982636e-309};
 for(int i=0;i<6;++i){auto result=solve(mass[i]);assert(result.ok);
  double tol=i==5?1e-319:1e-10*std::abs(expected[i]);
  assert(std::abs(result.heat-expected[i])<=tol);
#ifdef __CUDACC__
  Result* d;assert(cudaMallocManaged(&d,sizeof(*d))==cudaSuccess);run<<<1,1>>>(mass[i],d);
  assert(cudaDeviceSynchronize()==cudaSuccess&&d->ok);assert(std::abs(d->heat-expected[i])<=tol);assert(cudaFree(d)==cudaSuccess);
#endif
  std::cout<<std::setprecision(17)<<"PASS mass="<<mass[i]<<" heat="<<result.heat<<" reference="<<expected[i]<<'\n';
 }
}
