#define main inherited_geometry_fixture_main
#include "../solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "restart/Checkpoint.H"
#include "core/Configuration.H"
#include "ablation/InterfaceMath.H"
struct ConvergedColdSeed {
 static bool evaluate(const SurfacePhysicsInput&,const PhysicsConfig&,const MaterialPrimitive&,Real,Real T,bool,SurfacePhysicsResult& out,bool*) {
  if(T!=300)return false;
  out.topTemperature=T;out.thermalResidual=2.74e-108;return true;
 }
};
int main(int argc,char** argv){
 require(argc==2,"checkpoint directory argument");
 ModelConfig m;auto& p=m.physics;p.gasMode=ugkwp::GasMode::MixtureFrozen;
 p.gasViscosity=1.8e-5;p.gasConductivity=.03;p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
 p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;p.wallModel.nodes=8;
 p.enableParticles=true;std::string error;
 require(validateWallModelConfig(p,error),error);
 auto rootPhysics=p;rootPhysics.wallModel.family=ugkwp::gaswall::WallFamily::LowRe;
 for(auto& species:rootPhysics.species){species.R=287;species.cp0=1005;species.Tmin=100;species.Tmax=10000;}
 SurfacePhysicsInput input;input.area=1;input.dt=1;MaterialPrimitive material;material.temperature=300;SurfacePhysicsResult root;
 require((findFeasibleSurfaceRoot<false,ConvergedColdSeed>(input,rootPhysics,material,300,100,10000,root)),"converged cold seed rejected");
 require(root.topTemperature==300&&root.thermalResidual==2.74e-108,"cold root changed temperature or discarded reaction residual");
 auto h=coupledPair();h.surface.ablating={1,0};auto aux=h.filmAux;aux[0].solidFront=-.01;
 HostMesh gas,solid;SurfaceMesh surface;
 require(moveCoupledMeshes(h,aux,1,gas,solid,surface,error),error);
 const int fixed=h.surface.solidFace[1];
 for(int k=solid.faceOffsets[fixed];k<solid.faceOffsets[fixed+1];++k){
  const int v=solid.facePoints[k];require(mag(solid.points[v]-h.solidMesh.points[v])==0,"fixed interface vertex moved");
 }
 require(surface.ablating==h.surface.ablating,"geometry lost ablation selection");
 HostMesh measured;std::vector<Real> sweep;
 require(makeStageGeometry(h.solidMesh,solid.points,1,measured,sweep,error),error);
 require(std::abs(sweep[fixed])<1e-14,"fixed face swept volume");
 require(std::abs(sweep[h.surface.solidFace[0]]+.01)<1e-11,"active face material volume");
 m.particleInlet.faces={3};m.particleInlet.times={0,.1,1};m.particleInlet.volumeFraction={0,.001,.001};
 m.particleInlet.temperature=3500;h.particleSourceFaces={3};h.particleSourceResidualMass={.123e-9};
 require(writeCheckpoint(argv[1],m,h,error),error);HostState restored=h;
 require(readCheckpoint(argv[1],m,restored,error),error);
 require(restored.particleSourceResidualMass==h.particleSourceResidualMass,"inlet mass remainder lost");
 require(restored.surface.ablating==h.surface.ablating,"restart ablation mask lost");
 auto bad=m;bad.particleInlet.volumeFraction.back()=.002;
 require(!readCheckpoint(argv[1],bad,restored,error),"changed inlet program admitted");
 restored=h;restored.surface.ablating={0,1};
 require(!readCheckpoint(argv[1],m,restored,error),"changed ablation selection admitted");
 std::cout<<"PASS ODE admission, fixed-patch geometry, sweep volume, inlet and surface restart\n";
}
