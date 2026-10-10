from pathlib import Path
import subprocess,runpy
APP=Path(__file__).resolve().parents[1];ROOT=APP.parents[1]
def storage_source():
    fixture=runpy.run_path(str(ROOT/'tests/gas_wall/test_wall_geometry.py'))['BASE']
    return fixture+r'''
#include "gpu/SharedGasDeviceStorage.cuh"
#include "configuration/SharedGasModel.H"
int main(){
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
 for(int c=0;c<3;++c){chmt::GasPrimitive w;w.rho=1;w.temperature=700;w.Y[0]=1;w.velocity={5,0,0};h.gas.push_back(chmt::conservativeGas(w,1,p));}
 chmt::SharedGasDeviceStorage storage;assert(storage.configure(model,canonical,{},h,error));auto&v=storage.hostView();
 assert(v.gasBoundaryLayer.enabled&&v.gasBoundaryLayer.count==1);assert(v.gasBoundaryLayerModel.workspaceCount==0);assert(!v.gasBoundaryLayerModel.workspace);
 assert(v.gasBoundaryLayer.faceSlot[wall]==0&&v.gasBoundaryLayer.ownerSlot[0]==0&&v.gasBoundaryLayer.ownerSlot[1]==-1);
 chmt::WallKnot knot;knot.faces.resize(1);knot.faces[0].temperature=600;knot.faces[0].speciesRate[0]=.01;
 assert(storage.prepareWallBoundary(knot,h.surface,error));
 for(int c=0;c<3;++c){v.rho[c]=1;v.p[c]=1e5;v.Tgas[c]=700;v.Ux[c]=5;}
 v.Tgas[1]=900;
 assert(ugkwp::evaluateGasBoundaryLayerSlot(v,0,0));assert(v.gasBoundaryLayer.exchange[0].ready);
 assert(v.gasBoundaryLayerModel.input[0].temperature==600);assert(v.gasBoundaryLayerModel.input[0].matching.temperature==900);
 assert(v.gasBoundaryLayer.speciesFlux[0]==-.01);assert(v.gasBoundaryLayer.exchange[0].mass==-.01);
 assert(v.gasBoundaryLayerModel.output[0].reactionIntegral[0]==0);
 assert(storage.clearTrialStatus());assert(!v.gasBoundaryLayer.exchange[0].ready);
}
'''
def test_bounded_wall_storage_and_current_device_matching(tmp_path):
    stub=runpy.run_path(str(APP/'tests/test_shared_device_storage.py'))['STUB']
    (tmp_path/'cuda_runtime.h').write_text(stub)
    source=storage_source()
    src=tmp_path/'storage.cpp';src.write_text(source);exe=tmp_path/'storage'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(tmp_path),'-I'+str(APP),'-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
