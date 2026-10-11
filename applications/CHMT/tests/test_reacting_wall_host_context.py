from pathlib import Path
import subprocess,runpy
APP=Path(__file__).resolve().parents[1]; ROOT=APP.parents[1]
def test_actual_matching_snapshot_and_shared_workspace(tmp_path):
    fixture=runpy.run_path(str(ROOT/'tests/gas_wall/test_wall_geometry.py'))['BASE']
    body=r'''
#include "ablation/WallClosureHost.H"
#include "tests/TestSupport.H"
int main(){
 const auto m=cubes();chmt::HostMesh mesh;mesh.geometryVersion=7;
 for(auto p:m.points)mesh.points.push_back({p[0],p[1],p[2]});
 for(auto p:m.cellCentres)mesh.cellCentres.push_back({p[0],p[1],p[2]});
 mesh.volumes=m.volumes;mesh.faceOffsets=m.faceOffsets;mesh.facePoints=m.facePoints;mesh.owner=m.owner;mesh.neighbour=m.neighbour;
 mesh.boundaryKind.assign(m.owner.size(),chmt::BoundaryKind::Slip);mesh.boundaryKind[0]=chmt::BoundaryKind::Interface;
 chmt::WallClosureHost host;std::string error;assert(host.prepareGeometry(mesh,{0},error));
 std::vector<chmt::GasPrimitive> gas(3);auto p=chmt_test::physics();for(auto&w:gas)w=chmt_test::gas(p);
 gas[0].pressure=120000;gas[1].pressure=90000;gas[0].temperature=600;gas[1].temperature=800;gas[2].temperature=1000;
 std::vector<chmt::SstPrimitive> sst(3);sst[1].k=.4;sst[1].omega=12;
 std::vector<chmt::GasWallMatchingSample> samples;
 assert(host.sample(gas,sst,true,samples,error));assert(samples.size()==1);assert(samples[0].state.temperature==800);assert(samples[0].state.k==.4);assert(samples[0].pressure==90000);assert(samples[0].mechanicalPressure==120000);
 std::vector<chmt::GasWallClosureContext> contexts;
 p.wallModel.model=BoundaryLayerModel::ConstantTransport;assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(!contexts[0].workspace);
 p.wallModel.model=BoundaryLayerModel::ReactingSst;p.wallModel.nodes=96;
 assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(contexts[0].workspace);assert(contexts[0].workspaceCapacity==128);assert(contexts[0].matchingDistance>1);
 assert(contexts[0].quadrature.volume==1&&contexts[0].quadrature.count==0);assert(contexts[0].matching.temperature==800);assert(contexts[0].matchingPressure==90000);assert(contexts[0].mechanicalPressure==120000);
 auto* workspace=contexts[0].workspace;gas[1].temperature=850;gas[0].pressure=130000;
 assert(host.sample(gas,sst,true,samples,error));assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(contexts[0].workspace==workspace);assert(contexts[0].matching.temperature==850);assert(contexts[0].mechanicalPressure==130000);
 samples[0].mechanicalPressure=0;assert(!host.contexts(samples,{},p.wallModel,contexts,error));
 gas[0].pressure=0;assert(!host.sample(gas,sst,true,samples,error));gas[0].pressure=130000;
 // BVP node changes cannot create or invalidate unused diagnostic geometry.
 assert(host.descriptors()[0].distance.empty());auto config=p.wallModel;
 config.family=WallFamily::BoundaryLayer;config.model=BoundaryLayerModel::ReactingSst;config.enableSst=true;config.nodes=24;config.stretch=2;
 assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance.empty());
 auto invalid=mesh;invalid.boundaryKind.clear();assert(host.prepareGeometry(invalid,{0},error,config)); // unchanged version: cached
 config.nodes=48;config.stretch=3;assert(host.prepareGeometry(invalid,{0},error,config)); // profile controls do not change physical geometry
 ++invalid.geometryVersion;assert(!host.prepareGeometry(invalid,{0},error,config)); // real geometry change still validates
 ++mesh.geometryVersion;for(auto& point:mesh.points)point.z*=1.01;for(auto& centre:mesh.cellCentres)centre.z*=1.01;for(auto& volume:mesh.volumes)volume*=1.01;
 assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance.empty());assert(near(host.descriptors()[0].volume,1.01));
 samples.clear();assert(!host.contexts(samples,{},p.wallModel,contexts,error));
}
'''
    src=tmp_path/'context.cpp';src.write_text(fixture+body);exe=tmp_path/'context'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),'-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
