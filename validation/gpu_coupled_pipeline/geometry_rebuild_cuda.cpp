#define main unused_geometry_test_main
#include "polyhedral_geometry_cuda.cpp"
#undef main
int main(){device::PolyhedralGeometry gpu;int count=0;
    std::vector<HostMesh> meshes;for(int cells:{1,3,5,128})for(bool skew:{false,true}){auto m=boxes(cells,1.,skew);std::string error;assert(chmt::rebuildGeometry(m,error));meshes.push_back(m);}
    for(int pass=0;pass<3;++pass)for(std::size_t i=0;i<meshes.size();++i){auto m=meshes[(i+pass)%meshes.size()];
        if(pass==1)m.points[0].z=-0.;if(pass==2)for(auto& p:m.points)p.y+=.125;
        std::string error;assert(chmt::rebuildGeometry(m,error));auto t=topology(m);std::vector<int> walls;
        for(std::size_t f=0;f<m.owner.size();++f)if(pass!=2||f%3)walls.push_back(f);
        if(pass==1)std::reverse(walls.begin(),walls.end());
        for(int repeat=0;repeat<2;++repeat){
            const auto r=gpu.rebuild(t,points(m.points),walls);
            for(std::size_t f=0;f<r.faces.size();++f){exact(m.areaVectors[f],r.faces[f].area);exact(m.faceCentres[f],r.faces[f].centre);assert(!r.faces[f].invalid);}
            for(std::size_t c=0;c<r.cells.size();++c){exact(m.volumes[c],r.cells[c].volume);exact(m.cellCentres[c],r.cells[c].centre);assert(!r.cells[c].invalid);
                double expected=std::numeric_limits<double>::max();for(int f:walls)for(int k=m.faceOffsets[f]+1;k+1<m.faceOffsets[f+1];++k)
                    expected=std::min(expected,chmt::triangleDistance(m.cellCentres[c],m.points[m.facePoints[m.faceOffsets[f]]],m.points[m.facePoints[k]],m.points[m.facePoints[k+1]]));
                assert(r.distance[c].invalidFace<0);assert(std::abs(expected-r.distance[c].distance)<=2e-13*std::max(1.,std::abs(expected)));}
            // Interleave a face-only device operation; it cannot stale a full
            // query's topology, coordinates or cached output fields.
            auto other=m.points;for(auto& p:other)p.x+=1;device::Topology only;only.faceOffsets=t.faceOffsets;only.facePoints=t.facePoints;gpu.sweep(only,points(m.points),points(other));++count;
        }
        const auto noWalls=gpu.rebuild(t,points(m.points),{});for(auto r:noWalls.distance)assert(r.invalidFace<0&&r.distance==0);++count;
    }
    // More than 64 MiB of distinct queries forces bounded-cache eviction.
    // The frozen centroid recurrence is evaluated at each translated state:
    // even binary translations can alter its final division roundoff. Outlet
    // metadata disables distance work in this independent eviction oracle.
    const auto base=meshes[6];const auto baseTopology=topology(base);
    for(int iteration=0;iteration<1026;++iteration){const double shift=iteration<1024?double(iteration+1)/8:0;
        auto translated=base.points;for(auto& p:translated)p.x+=shift;
        auto expected=base;expected.points=translated;
        std::fill(expected.boundaryKind.begin(),expected.boundaryKind.end(),chmt::BoundaryKind::Outlet);
        std::string error;assert(chmt::rebuildGeometry(expected,error));
        const auto result=gpu.rebuild(baseTopology,points(translated),{});
        for(std::size_t f=0;f<result.faces.size();++f)exact(expected.areaVectors[f],result.faces[f].area);
        for(std::size_t c=0;c<result.cells.size();++c){exact(expected.volumes[c],result.cells[c].volume);exact(expected.cellCentres[c],result.cells[c].centre);assert(result.distance[c].distance==0);}
        ++count;
    }
    auto mesh=meshes.front();auto p=points(mesh.points),c=points(mesh.cellCentres);p[0].x=std::numeric_limits<double>::quiet_NaN();
    const std::vector<int> wallOrder={5,4,3,2,1,0};const auto invalid=gpu.wallDistance(topology(mesh),p,c,wallOrder);
    for(std::size_t cell=0;cell<c.size();++cell){int expected=-1;for(int f:wallOrder){bool bad=false;for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k)bad=bad||mesh.facePoints[k]==0;if(bad){expected=f;break;}}assert(invalid[cell].invalidFace==expected);}++count;
    std::cout<<"PASS "<<count<<" batched rebuild/reuse queries: topology, endpoint, wall order/subset, signed zero, interleaved operations and first invalid face\n";
}
