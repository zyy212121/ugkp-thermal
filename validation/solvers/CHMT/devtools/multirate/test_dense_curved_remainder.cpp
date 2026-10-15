
#define main original_sweep_main
#include "test_sweep_constraints.cpp"
#undef main
#include "restart/Checkpoint.C"
#include "materials/Reaction.H"
#include <iomanip>
int main(){
 {
  HostMesh bad;bad.faceOffsets={0};
  for(int i=0;i<5;++i){Real angle=i*4*std::acos(-1.)/5;bad.points.push_back({std::cos(angle),std::sin(angle),0});}
  bad.points.push_back({0,0,1});bad.points.push_back({0,0,-1});
  for(int i=0;i<5;++i)for(int side=0;side<2;++side){
   int j=(i+1)%5;bad.owner.push_back(0);bad.neighbour.push_back(-1);bad.boundaryKind.push_back(BoundaryKind::Slip);
   bad.facePoints.push_back(side?6:5);bad.facePoints.push_back(side?j:i);bad.facePoints.push_back(side?i:j);
   bad.faceOffsets.push_back(bad.facePoints.size());
  }
  std::string why;require(rebuildGeometry(bad,why),why);auto end=bad.points;for(auto& x:end)x.x+=.001;
  HostMesh sentinel;sentinel.geometryVersion=987;std::vector<Real> swept={123};
  require(!makeStageGeometry(bad,end,1,sentinel,swept,why),"positive fan must not admit a self-intersecting double cover");
  require(sentinel.geometryVersion==987&&swept==std::vector<Real>({123}),"invalid fan preserves output");
 }

 {
  auto supported=coupledPair();auto film=supported.filmAux;
  for(auto& face:film)face.baseVelocity={.01,0,0};
  HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;std::string why;
  require(moveCoupledMeshesConstrained(supported,film,{-.03,-.03},1,gas,solid,surface,report,why),"prescribed support: "+why);
  for(int face:supported.surface.solidFace)
   for(int k=supported.solidMesh.faceOffsets[face];k<supported.solidMesh.faceOffsets[face+1];++k){
    int point=supported.solidMesh.facePoints[k];
    require(absValue(solid.points[point].x-supported.solidMesh.points[point].x-.01)<1e-14,"support velocity may not be projected away");
   }
 }
 auto base=coupledPair();std::string error;
 for(auto* mesh:{&base.solidMesh,&base.gasMesh}){
  for(auto& x:mesh->points){const Real angle=(.5-x.y)*.08726646259971647;const Real radius=.065+(x.z-1)*.0001;
   x={.01+.0026*x.x,radius*std::cos(angle),radius*std::sin(angle)};}
  require(rebuildGeometry(*mesh,error),error);
 }
 PhysicsConfig physics;
 for(auto& c:physics.condensed){c.rho=1000;c.cp0=1000;c.cp1=0;c.e0=0;c.Tmin=100;c.Tmax=3000;c.conductivity=1;}
 base.solid.resize(2);
 for(int c=0;c<2;++c){base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];base.solid[c].energy=base.solid[c].condensed[0]*300000;}
 long double physicalTotal[2]={},sweepTotal[2]={};Real minimumPhi=1,maximumPhi=0,maximumResidual=0;
 for(int step=0;step<24;++step){
  auto aux=base.filmAux;std::vector<Real> target;
  for(int f=0;f<2;++f){aux[f].solidFront-=1e-9;target.push_back(-1e-9*mag(base.solidMesh.areaVectors[base.surface.solidFace[f]]));}
  HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;
  require(moveCoupledMeshesConstrained(base,aux,target,.001,gas,solid,surface,report,error),"dense wedge: "+error);
  HostMesh measured;std::vector<Real> swept;require(makeStageGeometry(base.solidMesh,solid.points,.001,measured,swept,error),error);
  std::vector<Real> next;require(updateMaterialSweepRemainder(base,target,swept,solid,next,error),error);
  std::vector<Real> geometryLimits;require(materialSweepResidualLimits(solid,base.surface.solidFace,geometryLimits),"geometry limits");
  for(int f=0;f<2;++f){
   int c=base.solidMesh.owner[base.surface.solidFace[f]];const Real dm=1000*target[f];
   base.solid[c].condensed[0]+=dm;base.solid[c].energy+=300000*dm;
   MaterialPrimitive primitive;require(recoverMaterial(base.solid[c],solid.volumes[c],physics,primitive),"dense-material recovery without mass repair step "+std::to_string(step));
   require(primitive.porosity*solid.volumes[c]<=4*geometryLimits[f]+64*std::numeric_limits<Real>::epsilon()*solid.volumes[c],
    "representational porosity must remain within the local geometry roundoff bound");
   minimumPhi=minValue(minimumPhi,primitive.porosity);maximumPhi=maxValue(maximumPhi,primitive.porosity);
   physicalTotal[f]+=target[f];sweepTotal[f]+=swept[base.surface.solidFace[f]];
   const long double residual=physicalTotal[f]-sweepTotal[f]-next[f];
   require(std::abs(residual)<1e-27L,"physical volume equals measured geometry plus retained remainder");
   maximumResidual=maxValue(maximumResidual,absValue(Real(residual)));
  }
  base.solidMesh=std::move(solid);base.gasMesh=std::move(gas);base.surface=std::move(surface);base.filmAux=std::move(aux);base.solidSweepRemainder=std::move(next);
  if(step==11){Encoder enc;enc(base);Decoder dec(enc.data);HostState restarted;dec(restarted);
   require(restarted.solidSweepRemainder==base.solidSweepRemainder,"checkpoint preserves every remainder bit");
   require(validateMaterialSweepRemainder(restarted,error),error);base=std::move(restarted);}
 }
 auto saved=base;HostMesh gas=base.gasMesh,solid=base.solidMesh;SurfaceMesh surface=base.surface;SweepConstraintReport report;
 require(!moveCoupledMeshesConstrained(base,base.filmAux,{-1,-1},1,gas,solid,surface,report,error),"invalid motion rejects");
 require(base.solidSweepRemainder==saved.solidSweepRemainder&&base.solid[0].condensed[0]==saved.solid[0].condensed[0],"rejection preserves physical inventory and remainder");
 std::cout<<std::setprecision(17)<<"PASS dense curved material 24 windows; checkpoint at 12; phi "<<minimumPhi<<" "<<maximumPhi<<" accountedVolumeResidual "<<maximumResidual<<"\n";
}
