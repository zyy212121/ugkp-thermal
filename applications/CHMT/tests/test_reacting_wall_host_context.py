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
 gas[0].temperature=600;gas[1].temperature=800;gas[2].temperature=1000;
 std::vector<chmt::SstPrimitive> sst(3);sst[1].k=.4;sst[1].omega=12;
 std::vector<chmt::GasWallMatchingSample> samples;
 assert(host.sample(gas,sst,true,samples,error));assert(samples.size()==1);assert(samples[0].state.temperature==800);assert(samples[0].state.k==.4);
 std::vector<chmt::GasWallClosureContext> contexts;
 p.wallModel.model=BoundaryLayerModel::ConstantTransport;assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(!contexts[0].workspace);
 p.wallModel.model=BoundaryLayerModel::ReactingSst;p.wallModel.nodes=96;
 assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(contexts[0].workspace);assert(contexts[0].workspaceCapacity==128);assert(contexts[0].matchingDistance>1);
 assert(contexts[0].quadrature.volume==1);assert(contexts[0].matching.temperature==800);
 auto* workspace=contexts[0].workspace;gas[1].temperature=850;
 assert(host.sample(gas,sst,true,samples,error));assert(host.contexts(samples,{},p.wallModel,contexts,error));assert(contexts[0].workspace==workspace);assert(contexts[0].matching.temperature==850);
 // SST geometry weights must resolve the actual profile-node source shape.
 const auto legacy=host.descriptors()[0].distance;auto config=p.wallModel;
 config.family=WallFamily::BoundaryLayer;config.model=BoundaryLayerModel::ReactingSst;config.enableSst=true;config.nodes=24;config.stretch=2;
 assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance!=legacy);
 const auto first=host.descriptors()[0].distance;
 auto invalid=mesh;invalid.boundaryKind.clear();assert(host.prepareGeometry(invalid,{0},error,config)); // unchanged version/options: no geometry traversal
 config.nodes=48;assert(!host.prepareGeometry(invalid,{0},error,config)); // options invalidate outer cache
 assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance!=first);
 const auto second=host.descriptors()[0].distance;config.stretch=3;
 assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance!=second);
 config.enableSst=false;assert(host.prepareGeometry(mesh,{0},error,config));assert(host.descriptors()[0].distance==legacy);
 samples.clear();assert(!host.contexts(samples,{},p.wallModel,contexts,error));
}
'''
    src=tmp_path/'context.cpp';src.write_text(fixture+body);exe=tmp_path/'context'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(APP),'-I'+str(ROOT),'-I'+str(ROOT/'common'),str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)
