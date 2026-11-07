#define main unused_polyhedral_geometry_main
#include "../gpu_coupled_pipeline/polyhedral_geometry_cuda.cpp"
#undef main
#include "surface_seam_probe.H"
int configurations=0,valid=0,ambiguous=0,unpaired=0,edgeChecks=0;
void compare(HostMesh mesh,chmt::SurfaceMesh surface,bool distort=false){
    std::string error;assert(chmt::rebuildGeometry(mesh,error));
    std::vector<device::SurfaceTopologyEdge> raw;std::map<std::pair<int,int>,int> map;
    for(int i=0;i<int(surface.solidFace.size());++i){const int f=surface.solidFace[i];
        for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k){const int a=mesh.facePoints[k],b=mesh.facePoints[k+1==mesh.faceOffsets[f+1]?mesh.faceOffsets[f]:k+1];
            const auto key=std::minmax(a,b);const auto hit=map.find(key);
            if(hit==map.end()){map[key]=raw.size();raw.push_back({a,b,i,-1});}
            else{auto& e=raw[hit->second];assert(e.neighbour<0&&e.a==b&&e.b==a);e.neighbour=i;e.neighbourA=a;e.neighbourB=b;}
        }
    }
    std::vector<device::SurfaceSeamPatch> patches(raw.size());
    for(std::size_t i=0;i<raw.size();++i)for(int f=0;f<int(mesh.owner.size());++f){
        if(mesh.boundaryKind[f]!=chmt::BoundaryKind::Periodic)continue;bool a=false,b=false;
        for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k){a=a||mesh.facePoints[k]==raw[i].a;b=b||mesh.facePoints[k]==raw[i].b;}
        if(a&&b){patches[i]={f,mesh.periodicPartner[f]};break;}
    }
    std::vector<seamProbe::Sample> samples;std::vector<HostMesh> nativeMeshes;
    for(int replay=0;replay<3;++replay){auto m=mesh;
        if(replay==1)for(auto& p:m.points){p.x=p.x*1.03125+.125;p.y=p.y*.9375-.25;p.z=p.z*1.0625;}
        assert(chmt::rebuildGeometry(m,error));
        if(distort&&replay==1)for(int f=0;f<int(m.owner.size());++f)if(m.boundaryKind[f]==chmt::BoundaryKind::Periodic){m.faceCentres[f].z+=.25;break;}
        seamProbe::Sample s;s.points=points(m.points);for(std::size_t f=0;f<m.owner.size();++f)
            s.faces.push_back({{m.areaVectors[f].x,m.areaVectors[f].y,m.areaVectors[f].z},{m.faceCentres[f].x,m.faceCentres[f].y,m.faceCentres[f].z},0});
        samples.push_back(std::move(s));nativeMeshes.push_back(std::move(m));
    }
    std::vector<device::SurfaceFaceMap> maps;for(int f:surface.solidFace)maps.push_back({f,-1,mesh.owner[f],-1});
    const auto results=seamProbe::evaluate(topology(mesh),maps,surface.solidFace,raw,patches,samples);
    for(std::size_t r=0;r<results.size();++r){const auto& actual=results[r];std::vector<chmt::SurfaceEdge> expected;
        bool numerical=true;const bool success=chmt::surfaceEdges(nativeMeshes[r],surface,expected,error,&numerical);
        assert(success==(actual.status.failure==device::SurfaceSeamFailure::None));
        if(!success){std::string expectedError;const bool isAmbiguous=actual.status.failure==device::SurfaceSeamFailure::Ambiguous;
            chmt::errorAt(expectedError,isAmbiguous?"ambiguous periodic surface edge":"unpaired periodic surface edge",actual.status.index);
            assert(error==expectedError);assert(numerical==isAmbiguous);
            assert(actual.surfaceStatus.failure==(isAmbiguous?device::SurfaceGeometryFailure::AmbiguousSeam:device::SurfaceGeometryFailure::UnpairedSeam));
            assert(actual.surfaceStatus.index==actual.status.index);if(isAmbiguous)++ambiguous;else++unpaired;continue;}
        ++valid;assert(expected.size()==actual.edges.size());
        assert(actual.surfaceStatus.failure==device::SurfaceGeometryFailure::None&&actual.surfaceStatus.edgeCount==int(expected.size()));
        auto reference=surface;reference.gasFace.assign(reference.solidFace.size(),-1);chmt::SurfaceMesh metrics;
        const auto& native=nativeMeshes[r];assert(chmt::rebuildTrajectorySurface(native,native,reference,native,native,1.,metrics,error));
        for(std::size_t i=0;i<actual.faceMetrics.size();++i){const auto v=actual.faceMetrics[i];
            exact(metrics.oldArea[i],v.oldArea);exact(metrics.area[i],v.area);exact(metrics.normal[i],v.normal);exact(metrics.centre[i],v.centre);exact(metrics.solidDistance[i],v.solidDistance);}
        for(std::size_t i=0;i<actual.edgeMetrics.size();++i){const auto v=actual.edgeMetrics[i];
            exact(metrics.edgeLength[i],v.length);exact(metrics.edgeConormal[i],v.conormal);exact(metrics.edgeOwnerOffset[i],v.ownerOffset);exact(metrics.edgeNeighbourOffset[i],v.neighbourOffset);exact(metrics.sweptEdgeArea[i],v.sweep);}
        for(std::size_t i=0;i<expected.size();++i){const auto e=expected[i];const auto a=actual.edges[i];
            assert(a.ownerSurface==e.owner&&a.geometry.a==e.a&&a.geometry.b==e.b&&a.geometry.ownerFace==surface.solidFace[e.owner]
                &&a.geometry.neighbourFace==(e.neighbour<0?-1:surface.solidFace[e.neighbour])&&a.geometry.neighbourA==e.neighbourA&&a.geometry.neighbourB==e.neighbourB);++edgeChecks;}
    }
    ++configurations;
}
int main(){
    for(int cells:{1,3,37})for(double scale:{1e-4,1.,1e4})for(bool skew:{false,true})for(bool periodic:{false,true}){
        auto m=boxes(cells,scale,skew);m.periodicPartner.assign(m.owner.size(),-1);chmt::SurfaceMesh s;
        for(int c=0;c<cells;++c){s.solidFace.push_back(6*c+1);if(periodic){m.boundaryKind[6*c+2]=m.boundaryKind[6*c+3]=chmt::BoundaryKind::Periodic;m.periodicPartner[6*c+2]=6*c+3;m.periodicPartner[6*c+3]=6*c+2;}}
        compare(m,s,periodic);
    }
    // Two overlapping, individually valid disconnected cells expose ambiguous
    // seam candidates; the native routine does not assume global separation.
    auto overlap=boxes(2,1.,false);for(int p=8;p<16;++p)overlap.points[p]=overlap.points[p-8];
    overlap.periodicPartner.assign(overlap.owner.size(),-1);for(int c=0;c<2;++c){overlap.boundaryKind[6*c+2]=overlap.boundaryKind[6*c+3]=chmt::BoundaryKind::Periodic;overlap.periodicPartner[6*c+2]=6*c+3;overlap.periodicPartner[6*c+3]=6*c+2;}
    chmt::SurfaceMesh s;s.solidFace={1,7};compare(overlap,s);
    auto closed=boxes(1,1.,false);s.solidFace={0,1,2,3,4,5};compare(closed,s);s.solidFace.clear();compare(closed,s);
    HostMesh none;none.faceOffsets={0};compare(none,{});
    assert(ambiguous&&unpaired);std::cout<<"PASS "<<configurations<<" native/GPU seam configurations, three captured replays: valid="<<valid<<" ambiguous="<<ambiguous<<" unpaired="<<unpaired<<" exact edges="<<edgeChecks<<'\n';
}
