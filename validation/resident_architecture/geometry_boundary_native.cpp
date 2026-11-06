// Select the untouched native boundary policies through the frozen include root.
#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "geometry_boundary_probe.H"
#include <tuple>
int cases=0,periodicFailures=0,emptyFailures=0,duplicates=0;
std::vector<device::FaceResult> faceResults(const HostMesh& m){
    std::vector<device::FaceResult> result(m.owner.size());
    for(std::size_t f=0;f<result.size();++f){const auto a=m.areaVectors[f],c=m.faceCentres[f];result[f].area={a.x,a.y,a.z};result[f].centre={c.x,c.y,c.z};}
    return result;
}
void periodicCase(HostMesh m){
    std::string error;const bool expected=chmt::periodicGeometry(m,error);
    const auto result=boundaryProbe::evaluate(topology(m),points(m.points),points(m.points),faceResults(m),m.periodicPartner,-1,{});
    const auto& status=result.periodic;
    assert(expected==(status.failure==device::GeometryBoundaryFailure::None));
    if(!expected){const char* message=status.failure==device::GeometryBoundaryFailure::PeriodicArea?
        "rotational/nonmatching periodic areas unsupported":"rotational/nontranslational periodic vertices unsupported";
        std::string actual;chmt::errorAt(actual,message,status.index);if(actual!=error){std::cerr<<actual<<" != "<<error<<'\n';std::abort();}++periodicFailures;}
    exact(status.scale,chmt::meshScale(m.points));++cases;
}
void emptyCase(HostMesh old,const std::vector<Vec3>& end){
    chmt::empty_detail::Extrusion extrusion;std::string error;assert(chmt::empty_detail::build(old,extrusion,error));
    const bool expected=chmt::empty_detail::trajectory(old,extrusion,end,error);
    const auto result=boundaryProbe::evaluate(topology(old),points(old.points),points(end),faceResults(old),old.periodicPartner,extrusion.axis,extrusion.mate);
    const auto& status=result.trajectory;
    assert(expected==(status.failure==device::GeometryBoundaryFailure::None));
    if(!expected){const char* message=status.failure==device::GeometryBoundaryFailure::EmptyNormal?
        "nonzero normal trajectory displacement":"layer-paired tangential trajectory differs";
        std::string actual;chmt::empty_detail::reject(actual,message);assert(actual==error);++emptyFailures;
        int first=-1;for(std::size_t p=0;p<end.size();++p){
            if(chmt::empty_detail::component(end[p],extrusion.axis)!=chmt::empty_detail::component(old.points[p],extrusion.axis)){first=p;break;}
            for(int d=0;d<3;++d)if(d!=extrusion.axis&&chmt::empty_detail::component(end[p],d)!=chmt::empty_detail::component(end[extrusion.mate[p]],d)){first=p;break;}
            if(first>=0)break;
        }
        assert(status.index==first);
    }else{
        // With a valid fixed extrusion and its unchanged paired trajectory,
        // duplicate tangential coordinates are the remaining coordinate-level
        // constraint. The native implementation re-derives its complete map.
        auto candidate=old;candidate.points=end;chmt::empty_detail::Extrusion next;
        const bool valid=chmt::empty_detail::build(candidate,next,error);
        assert(valid==(result.unique.failure==device::GeometryBoundaryFailure::None));
        if(!valid){assert(error=="unsupported/invalid Empty extrusion: duplicate tangential point in one layer");++duplicates;}
    }
    ++cases;
}
int main(){
    for(int cells:{1,7,129})for(double scale:{.001,1.,1024.}){
        auto original=boxes(cells,scale,true);
        original.periodicPartner.assign(original.owner.size(),-1);
        for(int c=0;c<cells;++c){original.periodicPartner[6*c+2]=6*c+3;original.periodicPartner[6*c+3]=6*c+2;
            original.boundaryKind[6*c+2]=original.boundaryKind[6*c+3]=chmt::BoundaryKind::Periodic;}
        std::string error;assert(chmt::rebuildGeometry(original,error));
        for(int mode=0;mode<8;++mode){auto m=original;
            if(mode==1)m.faceCentres[2].y+=.01*scale;
            if(mode==2)m.areaVectors[3]=m.areaVectors[3]*2;
            if(mode==3)m.points[m.facePoints[m.faceOffsets[2]]].y+=.03*scale;
            if(mode==4)m.points[m.facePoints[m.faceOffsets[2]]].y+=1e-12*scale;
            if(mode==5)m.points[m.facePoints[m.faceOffsets[2]]].y+=1e-8*scale;
            if(mode==6)m.points[m.facePoints[m.faceOffsets[2]+1]]=m.points[m.facePoints[m.faceOffsets[2]]];
            if(mode==7){m.faceCentres[2].y+=.01*scale;m.areaVectors[3]=m.areaVectors[3]*2;}
            periodicCase(m);
        }
        for(int rotate=0;rotate<3;++rotate){auto old=boxes(cells,scale,false);
            for(auto& p:old.points){if(rotate==1)p={p.z,p.x,p.y};if(rotate==2)p={p.y,p.z,p.x};}
            for(int c=0;c<cells;++c)old.boundaryKind[6*c]=old.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
            assert(chmt::rebuildGeometry(old,error));const int axis=(2+rotate)%3;
            for(int mode=0;mode<7;++mode){auto end=old.points;
                const auto set=[&](Vec3& p,int d,double value){if(d==0)p.x=value;else if(d==1)p.y=value;else p.z=value;};
                const auto get=[](Vec3 p,int d){return chmt::empty_detail::component(p,d);};
                if(mode==1)for(auto& p:end)set(p,(axis+1)%3,get(p,(axis+1)%3)+.125*scale);
                if(mode==2)set(end[0],axis,get(end[0],axis)+.125*scale);
                if(mode==3)set(end[0],(axis+1)%3,get(end[0],(axis+1)%3)+.125*scale);
                if(mode==4){end[1]=end[0];end[5]=end[4];}
                if(mode==5&&cells>1)for(int i=0;i<8;++i)end[8+i]=end[i];
                if(mode==6)for(auto& p:end)set(p,(axis+1)%3,-get(p,(axis+1)%3));
                emptyCase(old,end);
            }
        }
    }
    HostMesh empty;empty.faceOffsets={0};empty.cellFaceOffsets={0};
    const auto result=boundaryProbe::evaluate(topology(empty),{},{},{},{},-1,{});
    assert(result.periodic.failure==device::GeometryBoundaryFailure::None&&result.trajectory.failure==device::GeometryBoundaryFailure::None&&result.unique.failure==device::GeometryBoundaryFailure::None);++cases;
    std::cout<<"PASS "<<cases<<" native/GPU boundary policies: periodic failures="<<periodicFailures<<" Empty failures="<<emptyFailures<<" duplicate extrusions="<<duplicates<<"; capture and repeated execution\n";
    return 0;
}
