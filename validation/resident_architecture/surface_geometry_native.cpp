#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "surface_geometry_probe.H"
int cases=0,faceChecks=0,edgeChecks=0,failed=0;unsigned long long warnings=0;
surfaceProbe::Mesh packMesh(HostMesh mesh,const std::vector<Vec3>& end){
    surfaceProbe::Mesh result;result.topology=topology(mesh);result.old=points(mesh.points);result.end=points(end);result.partner=mesh.periodicPartner;
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==chmt::BoundaryKind::Slip||mesh.boundaryKind[f]==chmt::BoundaryKind::NoSlip||mesh.boundaryKind[f]==chmt::BoundaryKind::Interface)result.walls.push_back(f);
    chmt::empty_detail::Extrusion extrusion;std::string error;assert(chmt::empty_detail::build(mesh,extrusion,error));result.axis=extrusion.axis;result.mate=extrusion.mate;return result;
}
void compare(HostMesh solid,HostMesh gas,const std::vector<Vec3>& endSolid,const std::vector<Vec3>& endGas,chmt::SurfaceMesh reference,double dt=1e-3){
    std::string error;assert(chmt::rebuildGeometry(solid,error)&&chmt::rebuildGeometry(gas,error));
    auto candidateSolid=solid,candidateGas=gas;candidateSolid.points=endSolid;candidateGas.points=endGas;
    std::vector<chmt::SurfaceEdge> edges;assert(chmt::surfaceEdges(solid,reference,edges,error));
    std::vector<device::SurfaceFaceMap> maps;std::vector<device::SurfaceEdgeMap> edgeMaps;
    for(std::size_t i=0;i<reference.solidFace.size();++i){const int f=reference.solidFace[i],g=reference.gasFace[i];maps.push_back({f,g,solid.owner[f],g>=0?gas.owner[g]:-1});}
    for(const auto e:edges)edgeMaps.push_back({{e.a,e.b,reference.solidFace[e.owner],e.neighbour<0?-1:reference.solidFace[e.neighbour],e.neighbourA,e.neighbourB},e.owner});
    chmt::SurfaceMesh native;ugkp::diagnostics::DeviceContext diagnostic{};chmt::Tolerances tolerance;tolerance.diagnostics=&diagnostic;
    const bool actualExpected=chmt::rebuildTrajectorySurface(gas,solid,reference,candidateGas,candidateSolid,dt,native,error,tolerance);
    const auto actual=surfaceProbe::evaluate(packMesh(solid,endSolid),packMesh(gas,endGas),maps,edgeMaps,dt);
    if(actualExpected!=(actual.status.failure==device::SurfaceGeometryFailure::None)){std::cerr<<"surface "<<cases<<" native="<<error<<" device="<<int(actual.status.failure)<<'\n';std::abort();}
    if(!actualExpected){const char* message=actual.status.failure==device::SurfaceGeometryFailure::InvalidInterval?"invalid surface trajectory interval":
        actual.status.failure==device::SurfaceGeometryFailure::InvalidNormalDistance?"nonpositive interface normal distance":
        actual.status.failure==device::SurfaceGeometryFailure::DegenerateConormal?"degenerate surface edge conormal":"degenerate moving surface hinge";
        std::string actualError;chmt::errorAt(actualError,message,actual.status.index);assert(actualError==error);++failed;
    }else{
        assert(native.area.size()==actual.faces.size()&&native.edgeLength.size()==actual.edges.size());
        for(std::size_t i=0;i<actual.faces.size();++i){const auto a=actual.faces[i];exact(native.oldArea[i],a.oldArea);exact(native.area[i],a.area);
            exact(native.centre[i],a.centre);exact(native.normal[i],a.normal);exact(native.meshVelocity[i],a.velocity);
            exact(native.solidDistance[i],a.solidDistance);exact(native.gasDistance[i],a.gasDistance);++faceChecks;}
        for(std::size_t i=0;i<actual.edges.size();++i){const auto a=actual.edges[i];exact(native.edgeLength[i],a.length);exact(native.edgeConormal[i],a.conormal);
            exact(native.edgeOwnerOffset[i],a.ownerOffset);exact(native.edgeNeighbourOffset[i],a.neighbourOffset);exact(native.sweptEdgeArea[i],a.sweep);assert(a.invalid==0);++edgeChecks;}
    }
    const auto& expectedWarning=diagnostic.warnings[1];const auto& observedWarning=actual.diagnostics.warnings[1];
    assert(expectedWarning.count==observedWarning.count);warnings+=observedWarning.count;
    if(expectedWarning.count){assert(expectedWarning.sample.entityIndex==observedWarning.sample.entityIndex&&expectedWarning.sample.entityKind==observedWarning.sample.entityKind);
        exact(expectedWarning.sample.observed,observedWarning.sample.observed);exact(expectedWarning.sample.reference,observedWarning.sample.reference);}
    ++cases;
}
int main(){
    for(int cells:{1,2,7})for(double scale:{1e-4,1.,1e4})for(bool empty:{false,true})for(bool all:{false,true}){
        auto solid=boxes(cells,scale,false),gas=solid;
        for(auto& p:gas.points){if(empty)p.x+=scale;else p.z+=scale;}
        if(empty)for(int c=0;c<cells;++c){solid.boundaryKind[6*c]=solid.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;
            gas.boundaryKind[6*c]=gas.boundaryKind[6*c+1]=chmt::BoundaryKind::Empty;}
        chmt::SurfaceMesh surface;
        if(all){for(std::size_t f=0;f<solid.owner.size();++f)if(solid.boundaryKind[f]!=chmt::BoundaryKind::Empty){surface.solidFace.push_back(f);surface.gasFace.push_back(-1);}}
        else for(int c=0;c<cells;++c){surface.solidFace.push_back(6*c+(empty?3:1));surface.gasFace.push_back(6*c+(empty?2:0));}
        for(int movement=0;movement<3;++movement){auto endSolid=solid.points,endGas=gas.points;
            const auto move=[&](Vec3 p){const double u=(p.x-11.125)/scale,v=(p.y+3)/scale;
                if(movement==1)p+=Vec3{scale*.013,-scale*.007,empty?0:scale*.003};
                if(movement==2)p+=Vec3{scale*.001*v*v,scale*.002*u,empty?0:scale*.003*u*v};return p;};
            for(auto& p:endSolid)p=move(p);for(auto& p:endGas)p=move(p);compare(solid,gas,endSolid,endGas,surface);
        }
    }
    auto cube=boxes(1,1.,false);chmt::SurfaceMesh surface;surface.solidFace={1};surface.gasFace={-1};
    compare(cube,cube,cube.points,cube.points,surface,0);compare(cube,cube,cube.points,cube.points,surface,std::numeric_limits<double>::quiet_NaN());
    HostMesh concave;concave.faceOffsets={0};const std::vector<Vec3> outline={{0,0,0},{5,0,0},{5,1,0},{1,1,0},{1,2,0},{0,2,0}};
    concave.points=outline;for(auto p:outline){p.z=1;concave.points.push_back(p);}
    std::vector<std::vector<int>> faces={{5,4,3,2,1,0},{6,7,8,9,10,11}};
    for(int p=0;p<6;++p){const int q=(p+1)%6;faces.push_back({p,q,q+6,p+6});}
    chmt::SurfaceMesh boundary;
    for(const auto& face:faces){boundary.solidFace.push_back(concave.owner.size());boundary.gasFace.push_back(-1);
        concave.facePoints.insert(concave.facePoints.end(),face.begin(),face.end());concave.faceOffsets.push_back(concave.facePoints.size());
        concave.owner.push_back(0);concave.neighbour.push_back(-1);concave.boundaryKind.push_back(chmt::BoundaryKind::Slip);}
    compare(concave,concave,concave.points,concave.points,boundary);
    HostMesh none;none.faceOffsets={0};compare(none,none,{},{},{});
    std::cout<<"PASS "<<cases<<" composed gas/solid/surface graphs, "<<faceChecks<<" interface metrics and "<<edgeChecks<<" adaptive edges bitwise; rejected="<<failed<<" edge accuracy warnings="<<warnings<<'\n';return 0;
}
