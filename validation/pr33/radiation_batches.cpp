#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/SpecularReflection.H"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>
using namespace ugkp::gpu;
int main(int argc,char** argv){
 const int cells=argc>1?std::stoi(argv[1]):31;
 const double pi=std::acos(-1.),c=std::cos(.37),s=std::sin(.37);
 std::vector<std::array<double,3>> directions={{{1,0,0}},{{-1,0,0}},{{0,1,0}},{{0,-1,0}},{{0,0,1}},{{0,0,-1}}};
 if(argc>2){
  const int polar=std::stoi(argv[2]),azimuth=2*polar;directions.clear();
  for(int i=0;i<polar;++i)for(int j=0;j<azimuth;++j){
   const double mu=-1+(2.*i+1)/polar,phi=2*pi*(j+.5)/azimuth,r=std::sqrt(1-mu*mu);
   directions.push_back({r*std::cos(phi),r*std::sin(phi),mu});
  }
 }
 const int angles=directions.size();
 const std::vector<std::array<double,3>> normals={{{c,s,0}},{{-c,-s,0}},{{-s,c,0}},{{s,-c,0}},{{0,0,1}},{{0,0,-1}}};
 RadiationGeometry g;g.cells=cells;g.directions=angles;g.volume.assign(cells,1);
 g.cellOffsets.assign(cells+1,0);g.weights.assign(angles,4*pi/angles);
 for(const auto& a:directions)for(const auto& b:directions){double dot=0;for(int j=0;j<3;++j)dot+=a[j]*b[j];g.directionCosines.push_back(dot);}
 for(int cell=0;cell<cells;++cell)for(int f=0;f<6;++f){
  g.boundaryOwner.push_back(cell);g.boundaryPatch.push_back(f);g.boundaryFace.push_back(cell);
  g.boundaryType.push_back(f==5?RadiationBoundary::Black:RadiationBoundary::Specular);
  g.boundaryArea.push_back(1);g.fixedIntensity.push_back(0);double denominator=0;
  for(int a=0;a<angles;++a){double dot=0;for(int j=0;j<3;++j)dot+=directions[a][j]*normals[f][j];denominator+=g.weights[a]*std::max(0.,-dot);}
  g.diffuseDenominator.push_back(denominator);
 }
 std::vector<std::vector<std::vector<ReflectionEntry>>> maps;
 for(const auto& n:normals)maps.push_back(specularReflection(directions,g.weights,n));
 g.reflectionOffsets.push_back(0);
 for(int a=0;a<angles;++a){for(int cell=0;cell<cells;++cell){double diagonal=0;
  for(int f=0;f<6;++f){double dot=0;for(int j=0;j<3;++j)dot+=directions[a][j]*normals[f][j];
   g.boundaryCosine.push_back(dot);g.boundaryFlux.push_back(dot);diagonal+=std::max(0.,dot);
   for(auto e:maps[f][a]){g.reflectionDonors.push_back(e.donor);g.reflectionWeights.push_back(e.weight);}
   g.reflectionOffsets.push_back(g.reflectionDonors.size());
  }g.transportDiagonal.push_back(diagonal);
 }}
 MieOpticalTable table;table.temperature={100,2000};table.diameter={1e-5,1e-3};table.mu={-1,1};table.band={1,1};
 table.absorption.assign(4,1);table.scattering.assign(4,.3);table.extinction.assign(4,1.3);table.phase.assign(8,1);
 RadiationSnapshot snap;snap.epsilon.assign(cells,1e-5);snap.diameter.assign(cells,1e-5);snap.temperature.resize(cells);
 for(int i=0;i<cells;++i)snap.temperature[i]=700+i%17;
 snap.wallTemperature.assign(cells*6,400);snap.emissivity.assign(cells*6,1);
 std::vector<double> reference;int referenceIterations=0;
 for(int limit:{angles,1,2,std::min(8,angles),angles}){
  RadiationControls controls;controls.parallelAngles=true;controls.maximumConcurrentAngles=limit;
  controls.sourceTolerance=1e-12;controls.sourceIterations=500;
  std::size_t before=0,total=0,after=0;cudaMemGetInfo(&before,&total);
  DiscreteOrdinateRadiation solver(g,table,controls);cudaMemGetInfo(&after,&total);
  for(int run=0;run<3;++run){
   const auto start=std::chrono::steady_clock::now();auto result=solver.solve(snap,false);
   const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   if(result.maximumRelativeChange>controls.sourceTolerance)throw std::runtime_error("source convergence");
   std::vector<double> intensity;solver.downloadIntensities(intensity);
   if(reference.empty()){reference=intensity;referenceIterations=result.sourceIterations;}
   if(intensity.size()!=reference.size()||std::memcmp(intensity.data(),reference.data(),intensity.size()*sizeof(double))||referenceIterations!=result.sourceIterations)
    throw std::runtime_error("batching changed frozen-source arithmetic");
   double maximumClosure=0,maximumMirror=0;
   for(int cell=0;cell<cells;++cell){double net=result.power[cell],gross=std::abs(net);
    for(int f=0;f<6;++f){double q=result.boundaryFlux[cell*6+f];net+=q;gross+=std::abs(q);if(f!=5)maximumMirror=std::max(maximumMirror,std::abs(q));}
    maximumClosure=std::max(maximumClosure,std::abs(net)/std::max(1.,gross));
   }
   if(maximumClosure>1e-10||maximumMirror>1e-9)throw std::runtime_error("radiation energy closure");
   std::cout<<std::setprecision(17)<<"PASS cells="<<cells<<" angles="<<angles<<" concurrency="<<limit<<" run="<<run<<" seconds="<<seconds<<" allocated="<<(before-after)<<" iterations="<<result.sourceIterations<<" closure="<<maximumClosure<<" specular="<<maximumMirror<<'\n';
  }
 }
}
