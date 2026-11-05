#include "gasTransport/MolecularTransportDevice.cuh"
#include "gasWall/ReactingWallLayer.H"
#include <cstdio>
#include <cstdlib>
#define CUDA(call) do{auto e=(call);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 1;}}while(0)
struct Sample {double T,Y[3],mu,constant;};
__global__ void evaluate(ugkwp::MolecularViscosityView<double> view,Sample* samples,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 samples[i].mu=ugkwp::molecularViscosity<3>(view,7.25e-6,samples[i].T,samples[i].Y);
 samples[i].constant=ugkwp::molecularViscosity<3>(ugkwp::MolecularViscosityView<double>{},7.25e-6,samples[i].T,samples[i].Y);
}
// Actual BVP point and finite-interval constitutive operators on CUDA.
struct WallResult {double mu0,mu1,traction,conduction;int ok;};
__global__ void wall(ugkwp::MolecularViscosityView<double> view,WallResult* output){
 ugkwp::SpeciesThermoData<double> data[3];double coefficients[9]={1000,0,0,1400,0,0,1800,0,0};
 for(int i=0;i<3;++i){data[i].molarMass=view.species[i].molarMass;data[i].coefficientOffset=3*i;data[i].minTemperature=100;data[i].maxTemperature=5000;}
 ugkwp::gaswall::WallInput<double,3> in;in.temperature=300;in.pressure=101325;in.matchingDistance=.01;in.normal[1]=1;
 in.model.thermo.species=data;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=9;
 in.model.molecularViscosity=view;in.model.viscosity=7.25e-6;in.model.molecularPrandtl=.72;
 in.matching.massFraction[0]=.2;in.matching.massFraction[1]=.3;in.matching.massFraction[2]=.5;in.matching.temperature=900;
 ugkwp::gaswall::WallModelConfig<double> cfg;ugkwp::gaswall::detail::LayerContext<double,3> ctx;
 bool ok=ugkwp::gaswall::detail::makeContext(in,cfg,ctx);
 double state0[7]={0,0,0,.2,.3,0,0},state1[7]={10,0,600,.6,.1,0,0};
 // Dependent species is index 2 (the largest matching fraction).
 ugkwp::gaswall::detail::LayerPoint<double,3> a,b;
 ok=ugkwp::gaswall::detail::pointState(in,cfg,ctx,state0,a)&&ok;
 ok=ugkwp::gaswall::detail::pointState(in,cfg,ctx,state1,b)&&ok;
 double cache[7][7]{};ugkwp::gaswall::detail::storeSeedThermo(b,cache);
 ugkwp::gaswall::detail::LayerPoint<double,3> restored;
 double values[2][7]{};for(int j=0;j<7;++j)values[1][j]=state1[j];
 double caches[2][7][7]{};ugkwp::gaswall::detail::storeSeedThermo(b,caches[1]);
 ugkwp::gaswall::detail::SeedThermoRows<double,3,decltype(values)> cached(values,caches,true);
 ok=ugkwp::gaswall::detail::loadBasePoint(in,cfg,ctx,cached[1],restored)&&ok;
 ok=ok&&restored.mu==b.mu;
 ugkwp::gaswall::detail::LayerFlux<double,3> flux;
 ok=ugkwp::gaswall::detail::layerFlux(in,cfg,ctx,a,b,.01,flux)&&ok;
 *output={a.mu,b.mu,-flux.momentum[0],flux.thermal,int(ok)};
}
int main(){
 ugkwp::MolecularTransportConfiguration config;config.model=ugkwp::MolecularViscosityModel::SutherlandWilke;
 config.species={{1.8e-5,300,110.4,.028},{2.1e-5,300,127,.032},{8.9e-6,300,72,.002}};
 for(auto a:config.species)for(auto b:config.species){config.massRatioQuarter.push_back(std::sqrt(std::sqrt(b.molarMass/a.molarMass)));config.wilkeDenominator.push_back(std::sqrt(8*(1+a.molarMass/b.molarMass)));}
 ugkwp::MolecularTransportDevice model;CUDA(model.initialize(config));
 std::vector<Sample> host;
 for(double T:{200.,300.,900.,2500.})for(auto Y:std::vector<std::vector<double>>{{1,0,0},{0,1,0},{0,0,1},{.2,.3,.5},{.999999999998,1e-12,1e-12},{.6,.1,.3}})
 host.push_back({T,{Y[0],Y[1],Y[2]},0,0});
 Sample* device=nullptr;CUDA(cudaMalloc(&device,host.size()*sizeof(Sample)));
 CUDA(cudaMemcpy(device,host.data(),host.size()*sizeof(Sample),cudaMemcpyHostToDevice));
 evaluate<<<1,64>>>(model.view(),device,int(host.size()));CUDA(cudaGetLastError());
 CUDA(cudaMemcpy(host.data(),device,host.size()*sizeof(Sample),cudaMemcpyDeviceToHost));
 for(auto s:host)std::printf("bulk %.17g %.17g %.17g %.17g %.17g %.17g\n",s.T,s.Y[0],s.Y[1],s.Y[2],s.mu,s.constant);
 WallResult *deviceWall,result;CUDA(cudaMalloc(&deviceWall,sizeof(result)));
 wall<<<1,1>>>(model.view(),deviceWall);CUDA(cudaGetLastError());CUDA(cudaMemcpy(&result,deviceWall,sizeof(result),cudaMemcpyDeviceToHost));
 std::printf("wall %.17g %.17g %.17g %.17g %d\n",result.mu0,result.mu1,result.traction,result.conduction,result.ok);
 CUDA(cudaFree(device));CUDA(cudaFree(deviceWall));return result.ok?0:2;
}
