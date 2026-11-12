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
#include <fstream>
#include <limits>
#include <stdexcept>
struct DeviceOwner {
 std::vector<void*> allocations;
 ~DeviceOwner(){for(auto p:allocations)cudaFree(p);}
 template<class T>T* put(const std::vector<T>& x){T* p=nullptr;cudaMalloc(&p,x.size()*sizeof(T));cudaMemcpy(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice);allocations.push_back(p);return p;}
};
RadiationGeometry geometry(int cells,int polar,const std::string& topology){
 const double pi=std::acos(-1.),c=std::cos(.37),s=std::sin(.37);
 std::vector<std::array<RadiationReal,3>> directions={{{1,0,0}},{{-1,0,0}},{{0,1,0}},{{0,-1,0}},{{0,0,1}},{{0,0,-1}}};
 if(polar>0){
  const int azimuth=2*polar;directions.clear();
  for(int i=0;i<polar;++i)for(int j=0;j<azimuth;++j){
   const double mu=-1+(2.*i+1)/polar,phi=2*pi*(j+.5)/azimuth,r=std::sqrt(1-mu*mu);
   directions.push_back({RadiationReal(r*std::cos(phi)),RadiationReal(r*std::sin(phi)),RadiationReal(mu)});
  }
 }
 const int angles=directions.size();
 if(topology!="isolated"&&topology!="chain"&&topology!="cycle")throw std::runtime_error("unknown topology");
 const std::vector<std::array<RadiationReal,3>> normals={{{RadiationReal(c),RadiationReal(s),0}},{{RadiationReal(-c),RadiationReal(-s),0}},{{RadiationReal(-s),RadiationReal(c),0}},{{RadiationReal(s),RadiationReal(-c),0}},{{0,0,1}},{{0,0,-1}}};
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
 const std::vector<RadiationReal> reflectionWeights(g.weights.begin(),g.weights.end());
 for(const auto& n:normals)maps.push_back(specularReflection(directions,reflectionWeights,n));
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
 return g;
}
MieOpticalTable optical(double scattering){
 MieOpticalTable t;t.temperature={100,2000};t.diameter={1e-5,1e-3};t.mu={-1,1};t.band={1,1};
 t.absorption.assign(4,1);t.scattering.assign(4,scattering);t.extinction.assign(4,1+scattering);t.phase.assign(8,1);return t;
}
RadiationSnapshot snapshot(const RadiationGeometry& g,double temperature=700,double wall=400){
 RadiationSnapshot s;s.epsilon.assign(g.cells,1e-5);s.diameter.assign(g.cells,1e-5);s.temperature.assign(g.cells,temperature);
 s.wallTemperature.assign(g.boundaryOwner.size(),wall);s.emissivity.assign(g.boundaryOwner.size(),1);return s;
}
// Analytic one-cell, black-wall, zero-scattering solution for the actual
// rotated geometry: (D_a+k*V)*I_a = D_a*Bw+k*V*Bp.
double isolatedBlackPower(const RadiationGeometry& g,double tp,double tw,double k){
 const double pi=std::acos(-1.),Bp=5.670374419e-8*tp*tp*tp*tp/pi;
 const double difference=5.670374419e-8/pi*(tw-tp)*(tw+tp)*(tw*tw+tp*tp);
 double weight=0,response=0;
 for(int a=0;a<g.directions;++a){
  const double transport=g.transportDiagonal[a*g.cells];
  weight+=g.weights[a];response+=g.weights[a]*transport/(transport+k*g.volume[0]);
 }
 return k*(response*difference+(weight-4*pi)*Bp);
}
int failures=0;
void require(bool ok,const std::string& reason){if(!ok){++failures;std::cout<<"FAIL "<<reason<<'\n';}}
// Observational FP64 diagnostics do not feed the solver or retain a low part.
double black(double temperature){const double t2=temperature*temperature;return 5.670374419e-8*t2*t2/std::acos(-1.);}
double roundoffCoefficient(const RadiationGeometry& g){
 return sizeof(RadiationReal)==4?8.*(g.directions+8)*std::numeric_limits<float>::epsilon():0.;
}
double grossCellEmission(const RadiationGeometry& g,const RadiationSnapshot& s,
 const RadiationResult& r,const std::vector<double>& intensity,int cell){
 double angular=0;for(int a=0;a<g.directions;++a)angular+=std::abs(g.weights[a]*intensity[a*g.cells+cell]);
 return std::abs(double(r.absorption[cell]))*(angular+4*std::acos(-1.)*std::abs(black(s.temperature[cell])));
}
double observedBoundaryIntensity(const RadiationGeometry& g,const RadiationSnapshot& s,
 const std::vector<double>& intensity,int angle,int face){
 const int cell=g.boundaryOwner[face],row=angle*g.boundaryOwner.size()+face;
 if(g.boundaryCosine[row]>0)return intensity[angle*g.cells+cell];
 switch(g.boundaryType[face]){
 case RadiationBoundary::Black:return black(s.wallTemperature[face]);
 case RadiationBoundary::Zero:return 0;
 case RadiationBoundary::Fixed:return g.fixedIntensity[face];
 case RadiationBoundary::Specular:{double value=0;
  for(int e=g.reflectionOffsets[row];e<g.reflectionOffsets[row+1];++e)
   value+=g.reflectionWeights[e]*intensity[g.reflectionDonors[e]*g.cells+cell];
  return value;}
 default:{double incident=0;
  for(int a=0;a<g.directions;++a){const double cosine=g.boundaryCosine[a*g.boundaryOwner.size()+face];
   if(cosine>0)incident+=g.weights[a]*cosine*intensity[a*g.cells+cell];}
  return s.emissivity[face]*black(s.wallTemperature[face])+(1-s.emissivity[face])*incident/g.diffuseDenominator[face];}
 }
}
void record(const std::string& prefix,const std::string& name,const RadiationGeometry& g,const RadiationSnapshot& s,
 const RadiationResult& r,const std::vector<double>& intensity,double seconds,std::size_t bytes,double requested,bool closure=true){
 double maxClosure=0,total=0,grossTotal=0,maxResidual=0,maxTarget=0,maxCellWatts=0,maxCellBound=0;
 double totalGrossRadiative=0,externalPower=0,externalAngularPower=0;int warningCount=0;bool cellsAccepted=true;
 // FP32 closure allowance only; preserve stopping criteria and FP64.
 const double oldCoefficient=roundoffCoefficient(g);
 const double coefficient=sizeof(RadiationReal)==4?std::max(oldCoefficient,256.*std::numeric_limits<float>::epsilon()):oldCoefficient;
 for(int cell=0;cell<g.cells;++cell){
  double net=r.power[cell]*g.volume[cell],gross=std::abs(net);total+=net;grossTotal+=std::abs(net);
  double radiativeGross=std::abs(g.volume[cell])*grossCellEmission(g,s,r,intensity,cell);
  for(int face=0;face<int(g.boundaryOwner.size());++face)if(g.boundaryOwner[face]==cell){
   const double q=r.boundaryFlux[face]*g.boundaryArea[face];net+=q;gross+=std::abs(q);total+=q;grossTotal+=std::abs(q);externalPower+=std::abs(q);
   for(int a=0;a<g.directions;++a){const double magnitude=std::abs(g.weights[a]*g.boundaryCosine[a*g.boundaryOwner.size()+face]
    *observedBoundaryIntensity(g,s,intensity,a,face)*g.boundaryArea[face]);radiativeGross+=magnitude;externalAngularPower+=magnitude;}
  }
  for(int at=g.cellOffsets[cell];at<g.cellOffsets[cell+1];++at){const int f=g.cellFaces[at];double q=0;
   for(int a=0;a<g.directions;++a){const double flux=g.internalFlux[a*g.owner.size()+f];
    const double angular=g.weights[a]*flux*intensity[a*g.cells+(flux>=0?g.owner[f]:g.neighbour[f])];q+=angular;radiativeGross+=std::abs(angular);}
   net+=(cell==g.owner[f]?q:-q);gross+=std::abs(q);
  }
  const double bound=2e-5*std::max(1.,gross)+coefficient*radiativeGross;
  cellsAccepted=cellsAccepted&&std::abs(net)<=bound;
  maxCellWatts=std::max(maxCellWatts,std::abs(net));maxCellBound=std::max(maxCellBound,bound);
  maxClosure=std::max(maxClosure,std::abs(net)/std::max(1.,gross));totalGrossRadiative+=radiativeGross;
 }
 for(auto w:r.directionWarnings){warningCount+=w.linearIteration>0;
  maxResidual=std::max(maxResidual,double(w.publishedResidual));maxTarget=std::max(maxTarget,double(w.publishedTarget));
  if(w.publishedResidual>w.publishedTarget)require(w.residual>w.limit&&w.linearIteration>0,name+" publication warning is visible");
 }
 for(auto* x:{&r.incident,&r.power,&r.boundaryFlux})for(double value:*x)require(std::isfinite(value),name+" finite output");
 for(double value:intensity)require(std::isfinite(value),name+" finite output");
 require(std::isfinite(r.maximumRelativeChange)&&r.maximumRelativeChange<=radiationConvergenceTolerance(RadiationReal(2e-6)),name+" source convergence");
 const double globalOldBound=2e-5*std::max(1.,grossTotal),globalBound=globalOldBound+coefficient*totalGrossRadiative;
 if(closure)require(cellsAccepted&&std::abs(total)<=globalBound,name+" precision-aware energy closure");
 const auto ratio=[](double numerator,double denominator){return denominator>0?numerator/denominator:(numerator==0?0:std::numeric_limits<double>::infinity());};
 std::cout<<std::setprecision(17)<<"CASE "<<name<<" bits="<<8*sizeof(RadiationReal)<<" seconds="<<seconds<<" allocated="<<bytes
  <<" iterations="<<r.sourceIterations<<" sourceChange="<<r.maximumRelativeChange<<" requestedSource="<<requested
  <<" effectiveSource="<<radiationConvergenceTolerance(RadiationReal(requested))<<" requestedSourceMet="<<(r.maximumRelativeChange<=requested)
  <<" warnings="<<warningCount<<" publishedResidual="<<maxResidual<<" effectiveTarget="<<maxTarget
  <<" legacyCellClosure="<<maxClosure<<" legacyGlobalClosure="<<std::abs(total)/std::max(1.,grossTotal)
  <<" legacyClosureAccepted="<<(maxClosure<2e-5&&std::abs(total)<globalOldBound)
  <<" maxCellClosureW="<<maxCellWatts<<" maxCellBoundW="<<maxCellBound<<" globalClosureW="<<total
  <<" previousPrecisionBoundW="<<(globalOldBound+oldCoefficient*totalGrossRadiative)<<" closureRoundoffCoefficient="<<coefficient
  <<" globalOldBoundW="<<globalOldBound<<" globalBoundW="<<globalBound<<" grossRadiativeW="<<totalGrossRadiative
  <<" externalNetPowerAbsW="<<externalPower<<" externalAngularPowerW="<<externalAngularPower
  <<" relativeExternalPower="<<ratio(std::abs(total),externalPower)
  <<" observationWindowS=1 closureEnergyJ="<<total<<" grossRadiativeEnergyJ="<<totalGrossRadiative
  <<" relativeRadiativeEnergy="<<ratio(std::abs(total),totalGrossRadiative)<<'\n';
 std::ofstream out(prefix+"-"+name+".csv");out<<std::setprecision(17);
 for(double value:intensity)out<<"I,"<<value<<'\n';for(double value:r.power)out<<"P,"<<value<<'\n';for(double value:r.boundaryFlux)out<<"B,"<<value<<'\n';
}
int main(int argc,char**argv)try{
 std::cout<<std::unitbuf;const std::string prefix=argc>1?argv[1]:"radiation";
  for(std::string topology:{"isolated","chain","cycle"})for(bool parallel:{false,true})for(int limit:(parallel?std::vector<int>{1,2,8}:std::vector<int>{1})){
   auto g=geometry(31,2,topology);auto t=optical(.3);auto s=snapshot(g);for(int i=0;i<g.cells;++i)s.temperature[i]+=i%17;
   RadiationControls c;c.parallelAngles=parallel;c.maximumConcurrentAngles=limit;c.sourceTolerance=2e-6;
   DiscreteOrdinateRadiation solver(g,t,c);auto start=std::chrono::steady_clock::now();auto r=solver.solve(s,false);std::vector<double>I;solver.downloadIntensities(I);
   record(prefix,topology+"-"+(parallel?"parallel":"sequential")+"-"+std::to_string(limit),g,s,r,I,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),0,c.sourceTolerance);
  }
  for(double scatter:{0.,30.}){
   auto g=geometry(7,2,"cycle");auto s=snapshot(g);RadiationControls c;c.parallelAngles=true;c.maximumConcurrentAngles=2;c.sourceTolerance=scatter==30.?2e-8:2e-6;c.sourceIterations=800;
   DiscreteOrdinateRadiation solver(g,optical(scatter),c);auto r=solver.solve(s,false);std::vector<double>I;solver.downloadIntensities(I);record(prefix,"scatter-"+std::to_string(scatter),g,s,r,I,0,0,c.sourceTolerance);
  }
  for(double dw:{0.,1e-6,-1e-6,1.,-1.})for(double weightScale:{1.,1.0000001}){
   auto g=geometry(1,0,"isolated");for(auto& b:g.boundaryType)b=RadiationBoundary::Black;for(auto& w:g.weights)w*=weightScale;
   auto s=snapshot(g,1000,1000+dw);RadiationControls c;c.sourceTolerance=1e-12;
   DiscreteOrdinateRadiation solver(g,optical(0),c);auto r=solver.solve(s,false);std::vector<double>I;solver.downloadIntensities(I);
   const double tp=s.temperature[0],tw=s.wallTemperature[0];
   const double expected=isolatedBlackPower(g,tp,tw,1.5);
   const double error=std::abs(r.power[0]-expected),oldBound=1e-8+2e-5*std::abs(expected);
   const double gross=grossCellEmission(g,s,r,I,0),bound=oldBound+roundoffCoefficient(g)*gross;
   require(error<=bound,"near equilibrium power absolute error");
   if(std::abs(expected)>bound)require(double(r.power[0])*expected>0,"resolvable hot/cold direction");
   std::cout<<std::setprecision(17)<<"BALANCE deltaT="<<tw-tp<<" representedDeltaT="<<double(RadiationReal(tw)-RadiationReal(tp))
    <<" weightScale="<<weightScale<<" actual="<<r.power[0]<<" expected="<<expected<<" absError="<<error
    <<" oldBound="<<oldBound<<" precisionBound="<<bound<<" grossPower="<<gross<<" legacyAccepted="<<(error<=oldBound)<<'\n';
   record(prefix,"balance-"+std::to_string(dw)+"-"+(weightScale==1.?"closed":"offset"),g,s,r,I,0,0,c.sourceTolerance,weightScale==1.);
  }
  {
   auto g=geometry(31,2,"cycle");for(auto& b:g.boundaryType)b=RadiationBoundary::Black;auto s=snapshot(g,1000,1000);s.epsilon.assign(g.cells,0);
   RadiationControls c;c.parallelAngles=true;c.maximumConcurrentAngles=2;c.sourceTolerance=2e-6;DiscreteOrdinateRadiation solver(g,optical(0),c);
   auto r=solver.solve(s,false);s.wallTemperature.assign(g.boundaryOwner.size(),0);r=solver.solve(s,true);std::vector<double>I;solver.downloadIntensities(I);
   record(prefix,"warm-hot-to-zero",g,s,r,I,0,0,c.sourceTolerance);for(double x:I)require(x>=0&&x<1e-8,"zero intensity representation");
  }
  {
   auto g=geometry(7,2,"chain");auto s=snapshot(g);RadiationControls c;c.parallelAngles=true;c.maximumConcurrentAngles=2;c.sourceTolerance=2e-6;
   DiscreteOrdinateRadiation solver(g,optical(.3),c);solver.solve(s,false);for(auto& T:s.temperature)T+=127;
   auto r=solver.solve(s,true);std::vector<double>I;solver.downloadIntensities(I);record(prefix,"warm-rebase",g,s,r,I,0,0,c.sourceTolerance);
   solver.uploadIntensities(I);for(auto* x:{&g.volume,&g.internalFlux,&g.transportDiagonal,&g.boundaryFlux,&g.boundaryArea})for(auto& v:*x)v*=1.25;
   solver.updateGeometry(g);r=solver.solve(s,true);solver.downloadIntensities(I);record(prefix,"metric-refresh",g,s,r,I,0,0,c.sourceTolerance);
  }
  {
   auto g=geometry(1,0,"isolated");for(auto& b:g.boundaryType)b=RadiationBoundary::Black;auto s=snapshot(g,1000,0);DeviceOwner memory;
   const float wall=std::nextafter(1000.f,std::numeric_limits<float>::infinity());const double m=1e-5;
   s.resident.enabled=true;s.resident.cells=1;s.resident.boundaryFaces=6;s.resident.materialDensity=1;
   s.resident.mass=memory.put(std::vector<RadiationReal>{RadiationReal(m)});s.resident.temperatureMass=memory.put(std::vector<RadiationReal>{RadiationReal(m*1000)});s.resident.diameterMass=memory.put(std::vector<RadiationReal>{RadiationReal(m*1e-5)});
   s.wallTemperature.assign(6,wall);s.resident.boundaryTemperature=memory.put(std::vector<RadiationReal>(6,wall));s.residentWallTemperature.assign(6,1);s.contactEmissivity.assign(6,0);
   RadiationControls c;c.sourceTolerance=1e-12;DiscreteOrdinateRadiation solver(g,optical(0),c);auto r=solver.solve(s,false);std::vector<double>I;solver.downloadIntensities(I);
   const double tp=1000,tw=double(wall),expected=isolatedBlackPower(g,tp,tw,1.5);
   const double oldBound=1e-8+2e-5*std::abs(expected),error=std::abs(r.power[0]-expected);
   const double bound=oldBound+roundoffCoefficient(g)*grossCellEmission(g,s,r,I,0);
   require(error<=bound,"resident selected-precision temperature input");
   if(std::abs(expected)>bound)require(double(r.power[0])*expected>0,"resident resolvable hot/cold direction");
   std::cout<<std::setprecision(17)<<"RESIDENT wallInput="<<tw<<" deltaT="<<tw-tp<<" actual="<<r.power[0]<<" expected="<<expected<<" absError="<<error<<" oldBound="<<oldBound<<" precisionBound="<<bound<<" legacyAccepted="<<(error<=oldBound)<<'\n';record(prefix,"resident-fp32-temperature",g,s,r,I,0,0,c.sourceTolerance);
  }
  {
   auto g=geometry(7,2,"cycle");auto s=snapshot(g);RadiationControls c;c.parallelAngles=true;c.maximumConcurrentAngles=2;c.sourceTolerance=1e-12;c.sourceIterations=500;
   DiscreteOrdinateRadiation solver(g,optical(.3),c);auto r=solver.solve(s,false);std::vector<double>I;solver.downloadIntensities(I);record(prefix,"strict-request-observation",g,s,r,I,0,0,c.sourceTolerance);
  }
 std::cout<<"SUMMARY failures="<<failures<<'\n';return failures?1:0;
}catch(const std::exception& e){std::cout<<"EXCEPTION "<<e.what()<<'\n';return 2;}
