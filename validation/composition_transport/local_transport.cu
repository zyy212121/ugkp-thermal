#include "gasTransport/MolecularTransportDevice.cuh"
#include "gasWall/reconstruction/PhysicalPrimitiveFlux.H"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
struct Sample {double T,p,y,mu,k,D,j0,j1,heat,wm,wp,wd,wallHeat,wallJ0,wallJ1;};
#define CUDA(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}while(0)
__global__ void evaluate(ugkwp::MolecularViscosityView<double> model,Sample* s,int count){
 int i=threadIdx.x+blockIdx.x*blockDim.x;if(i>=count)return;auto& v=s[i];
 ugkwp::SpeciesThermoData<double> species[2];double coefficients[]={1004.675,0,0,819.27,0,0};
 for(int j=0;j<2;++j){species[j].model=ugkwp::SpeciesThermoModel::LinearCp;species[j].molarMass=j?.0440095:.02896520682164515;species[j].minTemperature=1;species[j].maxTemperature=5000;species[j].coefficientOffset=3*j;}
 ugkwp::SpeciesThermoView<double,2> thermo;thermo.species=species;thermo.coefficients=coefficients;thermo.coefficientCount=6;
 double Y[]={1-v.y,v.y},grad[]={-.7,.7},diff[2],h=0,D[2];
 v.mu=ugkwp::molecularViscosity<2>(model,9.,v.T,Y);
 const double pr=ugkwp::molecularPrandtl<2>(model,.9,Y),cp=Y[0]*coefficients[0]+Y[1]*coefficients[3];
 v.k=v.mu*cp/pr;
 for(int j=0;j<2;++j)D[j]=ugkwp::molecularDiffusivity(model,j,9.,v.T,v.p);
 v.D=D[0];double R=ugkwp::mixtureGasConstant(Y,thermo),rho=v.p/(R*v.T);
 bool available=false;
 if(!ugkwp::correctedSpeciesDiffusion(Y,grad,rho,D,0.,.7,v.T,thermo,diff,h,true,&available)||!available){v.mu=-1;return;}
 v.j0=diff[0];v.j1=diff[1];v.heat=h;
 ugkwp::gaswall::WallInput<double,2> in;in.pressure=v.p;in.model.thermo=thermo;in.model.viscosity=9;in.model.molecularPrandtl=.9;in.model.molecularViscosity=model;
 v.wm=ugkwp::gaswall::wallMolecularViscosity(in,v.T,Y);v.wp=ugkwp::gaswall::wallMolecularPrandtl(in,Y);v.wd=ugkwp::gaswall::wallMolecularDiffusivity(in,0,v.T,v.p);
 ugkwp::gaswall::physical::Primitive<double,2> state{};state.rho=rho;state.T=v.T;state.Y[0]=Y[0];state.Y[1]=Y[1];state.gradY[0][0]=grad[0];state.gradY[1][0]=grad[1];state.gradT[0]=2.;
 ugkwp::gaswall::WallModelConfig<double> cfg;cfg.enableSst=false;
 ugkwp::gaswall::physical::Face<double,2> flux{};ugkwp::gaswall::physical::Source<double,2> src{};double normal[]={1,0,0};
 if(!ugkwp::gaswall::physical::evaluatePrimitive(in,cfg,state,.01,normal,0.,flux,src)){v.mu=-2;return;}
 v.wallHeat=flux.energy;v.wallJ0=flux.species[0];v.wallJ1=flux.species[1];
}
int main(int argc,char**argv){if(argc!=2)return 1;std::ifstream f(argv[1]);std::string text((std::istreambuf_iterator<char>(f)),{});auto cfg=ugkwp::parseMolecularTransport(text,{"Air","CO2"});ugkwp::MolecularTransportDevice device;CUDA(device.initialize(cfg));
 std::vector<Sample> rows;for(double T:{60.,85.,100.,273.,292.,1000.,3000.})for(double p:{1000.,101325.,500000.})for(double y:{0.,1e-12,.1,.5,.9,1.})rows.push_back({T,p,y});
 Sample* samples;CUDA(cudaMalloc(&samples,rows.size()*sizeof(Sample)));CUDA(cudaMemcpy(samples,rows.data(),rows.size()*sizeof(Sample),cudaMemcpyHostToDevice));
 evaluate<<<(rows.size()+63)/64,64>>>(device.view(),samples,rows.size());CUDA(cudaGetLastError());CUDA(cudaMemcpy(rows.data(),samples,rows.size()*sizeof(Sample),cudaMemcpyDeviceToHost));
 for(auto& v:rows)std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",v.T,v.p,v.y,v.mu,v.k,v.D,v.j0,v.j1,v.heat,v.wm,v.wp,v.wd,v.wallHeat,v.wallJ0,v.wallJ1);
 CUDA(cudaFree(samples));
}
