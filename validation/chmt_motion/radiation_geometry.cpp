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
 std::cout<<std::unitbuf;
 const bool instrumented=argc>4&&std::string(argv[4])=="instrumented";
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
 const std::string topology=argc>3?argv[3]:"isolated";
 if(topology!="isolated"&&topology!="chain"&&topology!="cycle")throw std::runtime_error("unknown topology");
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
 // Add real upwind dependencies; a periodic ring exercises SCC iteration.
 if(topology!="isolated"){
  if(cells<3)throw std::runtime_error("connected topology requires at least three cells");
  const bool cyclic=topology=="cycle";
  for(int i=0;i<cells-1;++i){g.owner.push_back(i);g.neighbour.push_back(i+1);}
  if(cyclic){g.owner.push_back(cells-1);g.neighbour.push_back(0);}
  std::vector<std::vector<int>> rows(cells);
  for(int f=0;f<int(g.owner.size());++f){rows[g.owner[f]].push_back(f);rows[g.neighbour[f]].push_back(f);}
  g.cellOffsets={0};for(const auto& row:rows){g.cellFaces.insert(g.cellFaces.end(),row.begin(),row.end());g.cellOffsets.push_back(g.cellFaces.size());}
  for(int cell=0;cell<cells;++cell)for(int f=0;f<2;++f)
   if(cyclic||(f==0?cell<cells-1:cell>0)){
    const int face=6*cell+f;g.boundaryArea[face]=0;
    for(int a=0;a<angles;++a){auto& flux=g.boundaryFlux[a*cells*6+face];g.transportDiagonal[a*cells+cell]-=std::max(0.,flux);flux=0;}
   }
  for(int a=0;a<angles;++a){double flux=0;for(int j=0;j<3;++j)flux+=directions[a][j]*normals[0][j];
   for(int f=0;f<int(g.owner.size());++f){g.internalFlux.push_back(flux);
    g.transportDiagonal[a*cells+g.owner[f]]+=std::max(0.,flux);
    g.transportDiagonal[a*cells+g.neighbour[f]]+=std::max(0.,-flux);}
  }
 }
 MieOpticalTable table;table.temperature={100,2000};table.diameter={1e-5,1e-3};table.mu={-1,1};table.band={1,1};
 table.absorption.assign(4,1);table.scattering.assign(4,.3);table.extinction.assign(4,1.3);table.phase.assign(8,1);
 RadiationSnapshot snap;snap.epsilon.assign(cells,1e-5);snap.diameter.assign(cells,1e-5);snap.temperature.resize(cells);
 for(int i=0;i<cells;++i)snap.temperature[i]=700+i%17;
 snap.wallTemperature.assign(cells*6,400);snap.emissivity.assign(cells*6,1);

 RadiationControls controls;controls.parallelAngles=true;controls.maximumConcurrentAngles=2;controls.sourceTolerance=1e-12;controls.sourceIterations=1000;
 DiscreteOrdinateRadiation reused(g,table,controls);reused.solve(snap,false);
 for(int event=0;event<3;++event){
  auto moved=g;const double area=1.21,volume=1.331;
  for(auto& v:moved.volume)v*=volume;
  for(auto& v:moved.boundaryArea)v*=area;
  for(auto& v:moved.boundaryFlux)v*=area;
  for(auto& v:moved.transportDiagonal)v*=area;
  for(size_t a=0;a<size_t(angles);++a)for(size_t f=0;f<g.owner.size();++f){
   const double old=g.internalFlux[a*g.owner.size()+f]*area;
   const double next=event%2?-old:old;moved.internalFlux[a*g.owner.size()+f]=next;
   moved.transportDiagonal[a*cells+g.owner[f]]+=std::max(0.,next)-std::max(0.,old);
   moved.transportDiagonal[a*cells+g.neighbour[f]]+=std::max(0.,-next)-std::max(0.,-old);
  }
  reused.updateGeometry(moved);auto warm=reused.solve(snap,true);
  DiscreteOrdinateRadiation fresh(moved,table,controls);auto cold=fresh.solve(snap,false);
  std::vector<double> wi,ci;reused.downloadIntensities(wi);fresh.downloadIntensities(ci);double difference=0;
  for(size_t i=0;i<wi.size();++i)difference=std::max(difference,std::abs(wi[i]-ci[i])/std::max(1.,std::abs(ci[i])));
  double net=0,gross=0;for(int c=0;c<cells;++c){const double q=warm.power[c]*moved.volume[c];net+=q;gross+=std::abs(q);}
  for(size_t f=0;f<moved.boundaryArea.size();++f){const double q=warm.boundaryFlux[f]*moved.boundaryArea[f];net+=q;gross+=std::abs(q);}
  if(difference>2e-10||std::abs(net)>1e-9+1e-10*gross)throw std::runtime_error("moved radiation mismatch/closure");
  std::cout<<"MOVED_RADIATION_PASS event="<<event<<" difference="<<difference<<" residual="<<net<<" gross="<<gross<<'\n';g=std::move(moved);
 }
}
