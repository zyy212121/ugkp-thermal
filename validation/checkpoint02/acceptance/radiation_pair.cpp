// Acceptance-only paired workload. Original production and FP64 fixture are unchanged.
#ifndef UGKP_ACCEPTANCE_BITS
#error Compile this fixture with an explicit selected precision
#endif
#include "gpu/DiscreteOrdinateRadiation.H"
#include "gpu/SpecularReflection.H"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <limits>
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
 std::vector<double> reference;int referenceIterations=0;
 const std::vector<int> limits=instrumented?std::vector<int>{1,angles}:std::vector<int>{angles,1,2,std::min(8,angles),angles};
 for(int limit:limits){
  RadiationControls controls;controls.parallelAngles=true;controls.maximumConcurrentAngles=limit;
  controls.sourceTolerance=1e-12;controls.sourceIterations=500;
  std::size_t before=0,total=0,after=0;cudaMemGetInfo(&before,&total);
  DiscreteOrdinateRadiation solver(g,table,controls);cudaMemGetInfo(&after,&total);
  for(int run=0;run<(instrumented?1:3);++run){
   const auto start=std::chrono::steady_clock::now();auto result=solver.solve(snap,false);
   const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
   // Diagnostic output only. No control, stopping condition or assertion is changed.
   std::vector<double> intensity;solver.downloadIntensities(intensity);
   std::cout<<std::setprecision(17);
   int diagnosticWarnings=0;
   for(std::size_t direction=0;direction<result.directionWarnings.size();++direction){
    const auto& warning=result.directionWarnings[direction];
    diagnosticWarnings+=warning.linearIteration>0;
    std::cout<<"DIAG_LINEAR precision="<<UGKP_ACCEPTANCE_BITS<<" direction="<<direction
     <<" warning_residual="<<warning.residual<<" warning_limit="<<warning.limit
     <<" linear_iteration="<<warning.linearIteration<<" source_iteration="<<warning.sourceIteration;
#if UGKP_ACCEPTANCE_BITS == 32
    std::cout<<" publishedResidual="<<warning.publishedResidual<<" publishedTarget="<<warning.publishedTarget;
#else
    std::cout<<" publishedResidual=not_exposed_by_original_fp64_API publishedTarget=not_exposed_by_original_fp64_API";
#endif
    std::cout<<std::endl;
   }
   double diagnosticClosure=std::numeric_limits<double>::quiet_NaN();
   double diagnosticMirror=std::numeric_limits<double>::quiet_NaN();
   std::size_t diagnosticNonfinite=0;
   for(const auto* values:{&intensity,&result.power,&result.boundaryFlux})
    for(double value:*values)diagnosticNonfinite+=!std::isfinite(value);
   // Missing diagnostic entries produce unavailable metrics, not a refusal.
   if(intensity.size()==std::size_t(cells)*angles&&result.power.size()>=std::size_t(cells)
      &&result.boundaryFlux.size()>=std::size_t(cells)*6){
    diagnosticClosure=0;diagnosticMirror=0;
    for(int cell=0;cell<cells;++cell){
     double net=result.power[cell],gross=std::abs(net);
     for(int f=0;f<6;++f){double q=result.boundaryFlux[cell*6+f];net+=q;gross+=std::abs(q);
      if(f!=5)diagnosticMirror=std::max(diagnosticMirror,std::abs(q));}
     for(int at=g.cellOffsets[cell];at<g.cellOffsets[cell+1];++at){
      const int f=g.cellFaces[at];double flux=0;
      for(int direction=0;direction<angles;++direction){
       const double phi=g.internalFlux[direction*g.owner.size()+f];
       flux+=g.weights[direction]*phi*intensity[direction*cells+(phi>=0?g.owner[f]:g.neighbour[f])];
      }
      const double outward=cell==g.owner[f]?flux:-flux;net+=outward;gross+=std::abs(outward);
     }
     diagnosticClosure=std::max(diagnosticClosure,std::abs(net)/std::max(1.,gross));
    }
   }
   std::cout<<"DIAG_PRECHECK precision="<<UGKP_ACCEPTANCE_BITS<<" topology="<<topology
    <<" cells="<<cells<<" angles="<<angles<<" concurrency="<<limit<<" run="<<run
    <<" cold_start=true requested_tolerance="<<controls.sourceTolerance
    <<" linear_tolerance="<<controls.linearTolerance<<" source_metric="<<result.maximumRelativeChange
    <<" iterations="<<result.sourceIterations<<" warnings="<<diagnosticWarnings
    <<" closure="<<diagnosticClosure<<" mirror="<<diagnosticMirror
    <<" nonfinite_fields="<<diagnosticNonfinite
    <<" closure_limit="<<(UGKP_ACCEPTANCE_BITS==32?2e-5:1e-10)<<" mirror_limit="<<1e-9<<std::endl;
   if(!std::isfinite(result.maximumRelativeChange)||result.maximumRelativeChange>controls.sourceTolerance)throw std::runtime_error("source convergence");
   if(reference.empty()){reference=intensity;referenceIterations=result.sourceIterations;}
   if(intensity.size()!=reference.size())throw std::runtime_error("published intensity size mismatch");
   double publishedDifference=0;
   for(std::size_t i=0;i<intensity.size();++i){
    if(!std::isfinite(intensity[i]))throw std::runtime_error("nonfinite published intensity");
    publishedDifference=std::max(publishedDifference,std::abs(intensity[i]-reference[i])/std::max(1.,std::max(std::abs(intensity[i]),std::abs(reference[i]))));
   }
   const double comparisonTolerance=UGKP_ACCEPTANCE_BITS==32?2e-5:1e-12;
   if(publishedDifference>comparisonTolerance)throw std::runtime_error("batching published difference");
   double maximumClosure=0,maximumMirror=0;
   for(int cell=0;cell<cells;++cell){double net=result.power[cell],gross=std::abs(net);
    for(int f=0;f<6;++f){double q=result.boundaryFlux[cell*6+f];net+=q;gross+=std::abs(q);if(f!=5)maximumMirror=std::max(maximumMirror,std::abs(q));}
    for(int at=g.cellOffsets[cell];at<g.cellOffsets[cell+1];++at){const int f=g.cellFaces[at];double flux=0;
     for(int a=0;a<angles;++a){const double phi=g.internalFlux[a*g.owner.size()+f];
      flux+=g.weights[a]*phi*intensity[a*cells+(phi>=0?g.owner[f]:g.neighbour[f])];}
     const double outward=cell==g.owner[f]?flux:-flux;net+=outward;gross+=std::abs(outward);
    }
    maximumClosure=std::max(maximumClosure,std::abs(net)/std::max(1.,gross));
   }
   if(!std::isfinite(maximumClosure)||!std::isfinite(maximumMirror)||maximumClosure>(UGKP_ACCEPTANCE_BITS==32?2e-5:1e-10)||maximumMirror>1e-9)throw std::runtime_error("radiation energy closure");
   std::cout<<std::setprecision(17)<<"PASS precision="<<UGKP_ACCEPTANCE_BITS<<" requested_tolerance="<<controls.sourceTolerance<<" published_delta="<<publishedDifference<<" source_metric="<<result.maximumRelativeChange<<" topology="<<topology<<" cells="<<cells<<" angles="<<angles<<" concurrency="<<limit<<" run="<<run<<" seconds="<<seconds<<" allocated="<<(static_cast<long long>(before)-static_cast<long long>(after))<<" iterations="<<result.sourceIterations<<" closure="<<maximumClosure<<" specular="<<maximumMirror<<'\n';
  }
 }
}
