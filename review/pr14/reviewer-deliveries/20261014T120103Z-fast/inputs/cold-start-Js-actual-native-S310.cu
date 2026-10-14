#include "gpu/Backend.cu"
#include <cassert>
#include <cstring>

#include "common/gasWall/WallGeometryBuilder.H"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
using namespace ugkwp::gaswall;
using V=std::array<double,3>;
V add(V a,V b){return {a[0]+b[0],a[1]+b[1],a[2]+b[2]};}
V sub(V a,V b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
V mul(V a,double b){return {a[0]*b,a[1]*b,a[2]*b};}
double dot(V a,V b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
V cross(V a,V b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<r*(1+std::abs(b));}
struct MeshBuilder {
 WallGeometryMesh<double> m;
 std::map<std::vector<int>,int> seen;
 MeshBuilder(){m.faceOffsets.push_back(0);}
 int point(V p){for(int i=0;i<int(m.points.size());++i)if(m.points[i]==p)return i;m.points.push_back(p);return int(m.points.size())-1;}
 void cell(const std::vector<std::vector<V>>& faces,V centre,double volume){
  int c=int(m.cellCentres.size());m.cellCentres.push_back(centre);m.volumes.push_back(volume);
  for(auto f:faces){
   if(dot(cross(sub(f[1],f[0]),sub(f[2],f[0])),sub(f[0],centre))<0)std::reverse(f.begin(),f.end());
   std::vector<int> ids;for(auto p:f)ids.push_back(point(p));auto key=ids;std::sort(key.begin(),key.end());
   auto old=seen.find(key);if(old!=seen.end()){assert(m.neighbour[old->second]<0);m.neighbour[old->second]=c;continue;}
   int idx=int(m.owner.size());seen[key]=idx;m.owner.push_back(c);m.neighbour.push_back(-1);m.physicalWall.push_back(0);
   m.facePoints.insert(m.facePoints.end(),ids.begin(),ids.end());m.faceOffsets.push_back(int(m.facePoints.size()));
  }
 }
};
WallGeometryMesh<double> cubes(int count=3,double skew=0){
 MeshBuilder b;
 for(int k=0;k<count;++k){
  V a={skew*k,0,double(k)},bb={1+skew*k,0,double(k)},c={1+skew*k,1,double(k)},d={skew*k,1,double(k)};
  V e={skew*(k+1),0,double(k+1)},f={1+skew*(k+1),0,double(k+1)},g={1+skew*(k+1),1,double(k+1)},h={skew*(k+1),1,double(k+1)};
  b.cell({{a,bb,c,d},{e,f,g,h},{a,bb,f,e},{bb,c,g,f},{c,d,h,g},{d,a,e,h}}, {0.5+skew*(k+.5),.5,k+.5},1);
 }
 b.m.physicalWall[0]=1;return b.m;
}
WallGeometryMesh<double> tetra(){
 MeshBuilder b;V a={0,0,0},bb={2,0,0},c={0,2,0},d={.2,.3,1};
 b.cell({{a,bb,c},{a,bb,d},{bb,c,d},{c,a,d}}, {.55,.575,.25},2./3.);
 V e=add(bb,{0,0,3}),f=add(c,{0,0,3}),g=add(d,{0,0,3});
 b.cell({{bb,c,d},{e,f,g},{bb,c,f,e},{c,d,g,f},{d,bb,e,g}}, {2.2/3,2.3/3,11./6},4.5);
 b.m.physicalWall[0]=1;return b.m;
}
void checkMoments(const WallGeometryDescriptor<double>& d,V expected,double volume){
 double v=0,y=0,y2=0;V p={0,0,0};
 assert(d.distance.size()==d.volumeWeight.size());assert(d.distance.size()==d.quadraturePoints.size());
 for(size_t i=0;i<d.distance.size();++i){
  assert(d.volumeWeight[i]>0);assert(d.distance[i]>0);assert(d.distance[i]<d.matchingDistance);
  v+=d.volumeWeight[i];y+=d.volumeWeight[i]*d.distance[i];y2+=d.volumeWeight[i]*d.distance[i]*d.distance[i];
  p=add(p,mul(d.quadraturePoints[i],d.volumeWeight[i]));
 }
 assert(near(v,volume));assert(near(d.volume,volume));assert(near(y,d.firstMoment));
 for(int k=0;k<3;++k){assert(near(p[k],volume*expected[k]));assert(near(d.firstMomentVector[k],p[k]));}
 assert(d.matchingDistance>d.maximumOwnerDistance);assert(d.matchingCells.size()==1);
 assert(d.matchingWeights[0]>0&&near(d.matchingWeights[0],1));
 auto q=d.quadrature();assert(q.count==int(d.distance.size()));assert(q.distance==d.distance.data());
 assert(near(q.volume,volume));assert(near(q.firstMoment,y));
}

#include "gpu/SharedGasDeviceStorage.cuh"
#include "configuration/SharedGasModel.H"
int main(){
 for(int variant=0;variant<2;++variant){
 auto canonical=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model constant;coefficients(0.01 0.01);})");
 chmt::ModelConfig model;std::string error;assert(chmt::bindSharedGasModel(canonical,model,error));
 auto&p=model.physics;p.gasViscosity=.02;p.gasConductivity=1;p.wallModel.family=WallFamily::BoundaryLayer;
 p.wallModel.model=BoundaryLayerModel::ConstantTransport;p.wallWorkspaceSlots=2;
 const auto fixture=cubes();chmt::HostState h;auto&m=h.gasMesh;m.geometryVersion=1;m.volumes=fixture.volumes;
 for(auto v:fixture.points)m.points.push_back({v[0],v[1],v[2]});
 for(auto v:fixture.cellCentres)m.cellCentres.push_back({v[0],v[1],v[2]});
 std::vector<int> order;for(int f=0;f<int(fixture.owner.size());++f)if(fixture.neighbour[f]>=0)order.push_back(f);
 for(int f=0;f<int(fixture.owner.size());++f)if(fixture.neighbour[f]<0)order.push_back(f);
 int wall=-1;m.faceOffsets={0};m.cellFaceOffsets={0};
 for(int f:order){if(f==0)wall=int(m.owner.size());m.owner.push_back(fixture.owner[f]);m.neighbour.push_back(fixture.neighbour[f]);m.periodicPartner.push_back(-1);
  m.boundaryKind.push_back(f==0?chmt::BoundaryKind::Interface:fixture.neighbour[f]>=0?chmt::BoundaryKind::Internal:chmt::BoundaryKind::Slip);
  V area{},centre{};const int start=fixture.faceOffsets[f],end=fixture.faceOffsets[f+1];
  for(int i=start;i<end;++i){m.facePoints.push_back(fixture.facePoints[i]);const auto a=fixture.points[fixture.facePoints[i]],b=fixture.points[fixture.facePoints[i+1<end?i+1:start]];area=add(area,mul(cross(a,b),.5));centre=add(centre,mul(a,1./(end-start)));}
  m.faceOffsets.push_back(int(m.facePoints.size()));m.areaVectors.push_back({area[0],area[1],area[2]});m.faceCentres.push_back({centre[0],centre[1],centre[2]});}
 for(int c=0;c<3;++c){for(int f=0;f<int(m.owner.size());++f)if(m.owner[f]==c||m.neighbour[f]==c)m.cellFaces.push_back(f);m.cellFaceOffsets.push_back(int(m.cellFaces.size()));}
 m.boundaryPrimitive.resize(m.owner.size());h.surface.gasFace={wall};h.surface.area={1};
 for(int c=0;c<3;++c){chmt::GasPrimitive w;w.rho=variant?0.8:1;w.temperature=variant?900:700;w.Y[0]=1;w.velocity={variant?25.:5.,0,0};h.gas.push_back(chmt::conservativeGas(w,1,p));}

 p.enableSst=true;p.minDt=1e-12;p.maxDt=.001;p.cfl=.4;h.sst.assign(3,{1,10});m.wallDistance.assign(3,.5);m.boundarySst.resize(m.owner.size());
 for(int f=0;f<int(m.owner.size());++f)m.faceIds.push_back(f);m.oldVolumes=m.volumes;m.oldPoints=m.referencePoints=m.points;
 h.solid.resize(1);h.solid[0].condensed[0]=1;h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.surface.normal={{0,0,1}};
 chmt::GasExecutionOptions options;options.turbulenceModel=3;
 auto owner=[&](double js){std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> b(chmt::createBackend(model,canonical,{},options,h,error),chmt::destroyBackend);if(!b)throw std::runtime_error(error);
  chmt::WallProgram program;program.interval.end=.01;program.surface=h.surface;
  chmt::WallKnot a,z;a.time=0;z.time=.01;chmt::WallFaceSample w;w.temperature=600;w.speciesRate[0]=js;w.condensedRate[0]=js;a.faces=z.faces={w};program.knots={a,z};
  if(!chmt::beginGasWindow(*b,program,error))throw std::runtime_error(error);b->microBase=b->trial;b->microDt=variant?.00005:.0001;b->microTime=0;if(chmt::prepareCoupledGeometry(b.get())!=0)throw std::runtime_error(b->error);return b;};
 auto read=[&](const double* ptr,int count){std::vector<double> values(count);auto rc=cudaMemcpy(values.data(),ptr,count*sizeof(double),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("sync failed");return values;};
 auto b=owner(.013);const auto&v=b->storage.hostView();const auto oldK=read(v.rhoK,3),oldW=read(v.rhoOmega,3);const double referenceDt=variant?.00005:.0001;
 for(double js:{.013,.021,.034}){for(auto&k:b->program.knots)k.faces[0].speciesRate[0]=js;chmt::transport::GasCourantMaxima co;
  if(!chmt::currentGasCourant(*b,referenceDt,co))throw std::runtime_error(chmt::transport::lastError);const auto phi=read(v.gasPhiRho,int(m.owner.size())),area=read(v.magSf,int(m.owner.size()));
  assert(phi[wall]==-area[wall]*js);assert(read(v.rhoK,3)==oldK&&read(v.rhoOmega,3)==oldW);assert(!b->storage.wallMassPublished());
  auto fresh=owner(js);chmt::transport::GasCourantMaxima ref;if(!chmt::currentGasCourant(*fresh,referenceDt,ref))throw std::runtime_error(chmt::transport::lastError);
  const auto&fv=fresh->storage.hostView();assert(std::memcmp(&co.maximum,&ref.maximum,sizeof(double))==0);assert(read(v.gasFluxPositivityScale,3)==read(fv.gasFluxPositivityScale,3));assert(read(v.gasDiffusionNumber,3)==read(fv.gasDiffusionNumber,3));
  std::printf("R3JS fresh variant=%d Js=%a area=%a phi=%a dt=%a Co=%a flux=%a,%a,%a diffusion=%a,%a,%a\n",variant,js,area[wall],phi[wall],referenceDt,co.maximum,read(v.gasFluxPositivityScale,3)[0],read(v.gasFluxPositivityScale,3)[1],read(v.gasFluxPositivityScale,3)[2],read(v.gasDiffusionNumber,3)[0],read(v.gasDiffusionNumber,3)[1],read(v.gasDiffusionNumber,3)[2]);
 }
 if(chmt::prepareCoupledBoundaryLayer(b.get(),referenceDt,0)!=0)throw std::runtime_error(b->error);
 assert(b->storage.wallMassPublished());auto lagged=read(v.gasBoundaryLayer.speciesFlux,1)[0];assert(lagged==-.034);
 for(auto&k:b->program.knots)k.faces[0].speciesRate[0]=.055;chmt::transport::GasCourantMaxima co;if(!chmt::currentGasCourant(*b,referenceDt,co))throw std::runtime_error(chmt::transport::lastError);
 assert(read(v.gasPhiRho,int(m.owner.size()))[wall]==lagged);
 if(chmt::prepareCoupledBoundaryLayer(b.get(),referenceDt,0)!=0)throw std::runtime_error(b->error);assert(read(v.gasBoundaryLayer.speciesFlux,1)[0]==-.055);
 std::printf("R3JS lagged variant=%d published=%a Co=%a nextactualStage=%a dt=%a\n",variant,lagged,co.maximum,read(v.gasBoundaryLayer.speciesFlux,1)[0],referenceDt);
 }
 std::puts("PASS actual CUDA cold-start Js fresh references and lagged stage consumer two changed flow/dt controls");
}
