#define UGKP_INTERFACE_MOTION_FIXTURES_ONLY
#include "interface_motion_cuda.cpp"
#include "local_frame_stage_probe.H"
#include <cstring>
static void near(double a,double b){require(a==b||std::abs(a-b)<=3e-12*std::max({std::abs(a),std::abs(b),1e-300}),"local-frame stage field differs from native");}
static void near(chmt::Vec3 a,g::Point b){near(a.x,b.x);near(a.y,b.y);near(a.z,b.z);}
static void checkStage(const chmt::HostMesh& old,const g::LocalFrameStageDefinition& d,const std::vector<g::Point>& local,
    const g::LocalFrameStageResult& result,g::StageGeometryControls controls){
    chmt::HostMesh native;std::vector<double> sweep;std::string error;chmt::Tolerances tolerance;ugkp::diagnostics::DeviceContext diagnostics{};
    tolerance.diagnostics=&diagnostics;tolerance.absoluteGeometry=controls.absoluteTolerance;tolerance.relativeGeometry=controls.relativeTolerance;
    auto world=old.points;for(std::size_t p=0;p<local.size();++p){const auto displacement=unpack(local[p])-unpack(d.localOld[p]);if(displacement.x!=0||displacement.y!=0||displacement.z!=0)world[p]+=displacement;}
    const bool valid=chmt::makeStageGeometry(old,world,controls.interval,native,sweep,error,tolerance);
    require(valid==(result.status.failure==g::StageGeometryFailure::None),"local-frame trajectory acceptance differs: "+error);
    for(int i=0;i<2;++i){const auto& a=diagnostics.warnings[i];const auto& b=result.diagnostics.warnings[i];require(a.count==b.count,"local-frame warning count differs");
        if(a.count){require(a.sample.sourceSiteId==b.sample.sourceSiteId&&a.sample.entityIndex==b.sample.entityIndex,"local-frame warning sample differs");near(a.sample.observed,b.sample.observed);near(a.sample.reference,b.sample.reference);}}
    if(!valid)return;
    for(std::size_t p=0;p<world.size();++p){const auto expected=pack(std::vector<chmt::Vec3>{world[p]})[0];require(std::memcmp(&expected,&result.points[p],sizeof(expected))==0,"world-coordinate bits differ");}
    for(std::size_t f=0;f<native.owner.size();++f){near(native.areaVectors[f],result.endpoint.faces[f].area);near(native.faceCentres[f],result.endpoint.faces[f].centre);near(native.meshVelocity[f],result.velocity[f]);near(sweep[f],result.sweep[f]);}
    for(std::size_t c=0;c<native.volumes.size();++c){near(native.volumes[c],result.endpoint.cells[c].volume);near(native.oldVolumes[c],result.oldVolume[c]);near(native.cellCentres[c],result.endpoint.cells[c].centre);near(native.wallDistance[c],result.endpoint.distance[c].distance);}
}
int main(){int cases=0;
    for(bool tetra:{false,true})for(double scale:{1e-3,1.,1e3})for(double translation:{0.,1e9})for(bool empty:{false,true}){
        if(tetra&&empty)continue;
        auto mesh=tetra?tetrahedralBulk():twoHexahedra(0);for(auto& p:mesh.points)p={scale*(p.x+translation),scale*(p.y-translation),scale*(p.z+translation)};
        if(translation==0)mesh.points.front()={-0.,-0.,-0.};
        if(empty)for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0){const int lo=mesh.faceOffsets[f],hi=mesh.faceOffsets[f+1];bool layer=true;const double z=mesh.points[mesh.facePoints[lo]].z;
            for(int k=lo;k<hi;++k)layer=layer&&mesh.points[mesh.facePoints[k]].z==z;if(layer)mesh.boundaryKind[f]=chmt::BoundaryKind::Empty;}
        std::string error;require(chmt::rebuildGeometry(mesh,error),error);
        g::LocalFrameStageDefinition d;d.topology=definition(mesh,0).gas.topology;d.topology.cellOffsets=mesh.cellFaceOffsets;d.topology.cellFaces=mesh.cellFaces;d.topology.cellSigns=mesh.cellFaceSigns;
        d.worldOld=pack(mesh.points);d.localOld=d.worldOld;for(auto& p:d.localOld)p=p-d.worldOld.front();d.periodicPartner=mesh.periodicPartner;
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==chmt::BoundaryKind::Slip||mesh.boundaryKind[f]==chmt::BoundaryKind::NoSlip||mesh.boundaryKind[f]==chmt::BoundaryKind::Interface)d.walls.push_back(f);
        if(empty){chmt::empty_detail::Extrusion extrusion;require(chmt::empty_detail::build(mesh,extrusion,error),error);d.emptyAxis=extrusion.axis;d.emptyMate=extrusion.mate;}
        g::LocalFrameStageGeometry owner(d);
        for(int mode=0;mode<8;++mode){auto end=d.localOld;g::StageGeometryControls controls{.001,1e-20,1e-10};
            if(mode==1)for(std::size_t p=0;p<end.size();++p)end[p]+=g::Point{scale*.001*(1+p%2),-scale*.002*(1+p%3),scale*.003};
            if(mode==2)end[0].x=std::numeric_limits<double>::quiet_NaN();if(mode==3)controls.interval=0;if(mode==4)controls.absoluteTolerance=-1;
            if(mode==5)for(auto& p:end)p.z=0;if(mode==6)for(auto& p:end)p.z=-p.z;if(mode==7)for(auto& p:end){p.x=-p.x;p.y=-p.y;}
            try {for(const auto& result:probeLocalFrameStage(owner,d,end,controls))checkStage(mesh,d,end,result,controls);}
            catch(const std::exception& e){std::cerr<<"case="<<cases<<" tetra="<<tetra<<" scale="<<scale<<" translation="<<translation<<" empty="<<empty<<" mode="<<mode<<": "<<e.what()<<std::endl;throw;}
            ++cases;
        }
    }
    std::cout<<"PASS "<<cases<<" frozen local/world coordinate conversions, translated trajectories, collapse/inversion, full native geometry/GCL, resident replay and borrowed publication\n";
}
