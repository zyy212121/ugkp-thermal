#define UGKP_INTERFACE_MOTION_FIXTURES_ONLY
#include "interface_motion_cuda.cpp"
#include "interface_stage_probe.H"
static void near(double a,double b){require(a==b||std::abs(a-b)<=3e-12*std::max({std::abs(a),std::abs(b),1e-300}),"stage field differs from native");}
static void near(chmt::Vec3 a,g::Point b){near(a.x,b.x);near(a.y,b.y);near(a.z,b.z);}
static void checkStage(const chmt::HostMesh& old,const g::InterfaceGasPointMotionResult& motion,
    const g::InterfaceGasStageResult& result,g::StageGeometryControls controls){
    require(motion.status.failure==result.motion.status.failure&&motion.status.changed==result.motion.status.changed,"motion stage branch differs");
    if(motion.status.failure!=g::InterfaceGasMotionFailure::None)return;
    chmt::HostMesh native;std::vector<double> sweep;std::string error;chmt::Tolerances tolerance;
    ugkp::diagnostics::DeviceContext diagnostics{};tolerance.diagnostics=&diagnostics;
    tolerance.absoluteGeometry=controls.absoluteTolerance;tolerance.relativeGeometry=controls.relativeTolerance;
    std::vector<chmt::Vec3> endpoint;for(auto p:motion.points)endpoint.push_back(unpack(p));
    const bool valid=chmt::makeStageGeometry(old,endpoint,controls.interval,native,sweep,error,tolerance);
    require(valid==(result.status.failure==g::StageGeometryFailure::None),"trajectory acceptance differs: "+error);
    for(int i=0;i<2;++i){const auto& a=diagnostics.warnings[i];const auto& b=result.diagnostics.warnings[i];
        require(a.count==b.count,"stage diagnostic count differs");if(a.count){require(a.sample.sourceSiteId==b.sample.sourceSiteId&&a.sample.entityIndex==b.sample.entityIndex,"stage diagnostic sample differs");near(a.sample.observed,b.sample.observed);near(a.sample.reference,b.sample.reference);}}
    if(!valid)return;
    require(native.points.size()==result.motion.points.size(),"stage points shape differs");
    for(std::size_t p=0;p<native.points.size();++p)near(native.points[p],result.motion.points[p]);
    for(std::size_t f=0;f<native.owner.size();++f){near(native.areaVectors[f],result.endpoint.faces[f].area);near(native.faceCentres[f],result.endpoint.faces[f].centre);
        near(native.meshVelocity[f],result.velocity[f]);near(sweep[f],result.sweep[f]);}
    for(std::size_t c=0;c<native.volumes.size();++c){near(native.volumes[c],result.endpoint.cells[c].volume);near(native.oldVolumes[c],result.oldVolume[c]);near(native.cellCentres[c],result.endpoint.cells[c].centre);near(native.wallDistance[c],result.endpoint.distance[c].distance);}
}
int main(){int cases=0;
    for(bool tetra:{false,true})for(double scale:{1e-3,1.,1e3})for(double depth:{0.,.001}){
        auto mesh=tetra?tetrahedralBulk():twoHexahedra(0);for(auto& p:mesh.points)p={scale*(p.x+7.1),scale*(p.y-4.3),scale*(p.z+2.2)};
        std::string error;require(chmt::rebuildGeometry(mesh,error),error);const auto areas=pack(mesh.areaVectors);
        g::InterfaceGasStageDefinition sd;auto& d=sd.motion;d=definition(mesh,depth*scale);
        d.gas.topology.cellOffsets=mesh.cellFaceOffsets;d.gas.topology.cellFaces=mesh.cellFaces;d.gas.topology.cellSigns=mesh.cellFaceSigns;
        auto gas=mesh;gas.points.clear();for(auto p:d.gas.points)gas.points.push_back(unpack(p));require(chmt::rebuildGeometry(gas,error),error);
        for(std::size_t f=0;f<gas.owner.size();++f)if(gas.boundaryKind[f]==chmt::BoundaryKind::Slip||gas.boundaryKind[f]==chmt::BoundaryKind::NoSlip||gas.boundaryKind[f]==chmt::BoundaryKind::Interface)sd.wallFaces.push_back(f);
        sd.periodicPartner=gas.periodicPartner;g::InterfaceGasStage owner(sd);
        for(int mode=0;mode<9;++mode){auto end=d.oldSolid;g::StageGeometryControls controls{.001,1e-20,1e-10};
            if(mode)for(std::size_t p=0;p<end.size();++p)end[p]+=g::Point{scale*.001*(1+p%2),-scale*.002*(1+p%3),scale*.003};
            if(mode==2)end[0].x=std::numeric_limits<double>::quiet_NaN();
            if(mode==3)controls.interval=0;if(mode==4)controls.absoluteTolerance=-1;if(mode==5)controls.absoluteTolerance=controls.relativeTolerance=0;
            if(mode==6)for(auto& p:end)p.z=2*d.oldSolid[0].z-p.z;
            if(mode==7)for(auto& p:end){p.x=2*d.oldSolid[0].x-p.x;p.y=2*d.oldSolid[0].y-p.y;}
            if(mode==8)for(auto& p:end)p.z=d.oldSolid[0].z;
            const auto expected=reference(d,end,areas);
            checkStage(gas,expected,owner.solve(end,areas,controls),controls);
            checkStage(gas,expected,probeInterfaceStage(owner,sd,end,areas,controls),controls);
            checkStage(gas,expected,owner.solve(end,areas,controls),controls);++cases;
        }
        const g::StageGeometryControls controls{.001,1e-20,1e-10};checkStage(gas,reference(d,d.oldSolid,areas),owner.solve(d.oldSolid,areas,controls),controls);
    }
    for(double scale:{1e-3,1.,1e3}){
        auto mesh=twoHexahedra(0);for(auto& p:mesh.points)p=p*scale;
        std::string error;require(chmt::rebuildGeometry(mesh,error),error);
        // Identify the two extrusion layers by their original coordinates.
        for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.neighbour[f]<0){bool horizontal=true;const int lo=mesh.faceOffsets[f],hi=mesh.faceOffsets[f+1];
            const double z=mesh.points[mesh.facePoints[lo]].z;for(int k=lo;k<hi;++k)horizontal=horizontal&&mesh.points[mesh.facePoints[k]].z==z;
            if(horizontal)mesh.boundaryKind[f]=chmt::BoundaryKind::Empty;}
        require(chmt::rebuildGeometry(mesh,error),error);chmt::empty_detail::Extrusion empty;require(chmt::empty_detail::build(mesh,empty,error),error);
        g::InterfaceGasStageDefinition sd;auto& d=sd.motion;d=definition(mesh,0);d.emptyAxis=empty.axis;d.emptyMate=empty.mate;
        d.gas.topology.cellOffsets=mesh.cellFaceOffsets;d.gas.topology.cellFaces=mesh.cellFaces;d.gas.topology.cellSigns=mesh.cellFaceSigns;
        sd.periodicPartner=mesh.periodicPartner;for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==chmt::BoundaryKind::Slip)sd.wallFaces.push_back(f);
        g::InterfaceGasStage owner(sd);const auto areas=pack(mesh.areaVectors);const g::StageGeometryControls controls{.001,1e-20,1e-10};
        for(int mode=0;mode<4;++mode){auto end=d.oldSolid;
            if(mode==0)for(auto& p:end)p.x+=.001*scale;
            if(mode==1)end[0].z+=.001*scale;
            if(mode==2)end[0].x+=.001*scale;
            if(mode==3)for(auto& p:end)p.x+=.001*p.y*p.y/scale;
            const auto expected=reference(d,end,areas);checkStage(mesh,expected,owner.solve(end,areas,controls),controls);
            checkStage(mesh,expected,probeInterfaceStage(owner,sd,end,areas,controls),controls);++cases;}
    }
    std::cout<<"PASS "<<cases<<" frozen native interface motion + complete stage transactions, diagnostic order, invalid inputs, borrowed replay and failure recovery\n";
}
