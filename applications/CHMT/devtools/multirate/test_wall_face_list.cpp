#include "mesh/Geometry.C"
#define main inherited_sweep_tests_main
#include "test_sweep_constraints.cpp"
#undef main
#include <iomanip>

namespace {
int checked = 0;
void checkAllDistances(HostMesh& mesh) {
    std::string error;
    require(rebuildGeometry(mesh,error),error);
    bool hasWall=false;
    for (auto kind:mesh.boundaryKind)
        hasWall=hasWall || kind==BoundaryKind::NoSlip || kind==BoundaryKind::Slip
            || kind==BoundaryKind::Interface;
    for (std::size_t c=0;c<mesh.cellCentres.size();++c) {
        Real reference=hasWall?std::numeric_limits<Real>::max():0;
        // Preserve the original ascending all-face traversal as the oracle.
        for (std::size_t f=0;f<mesh.owner.size();++f) {
            const auto kind=mesh.boundaryKind[f];
            if (kind!=BoundaryKind::NoSlip && kind!=BoundaryKind::Slip
                && kind!=BoundaryKind::Interface)continue;
            const int begin=mesh.faceOffsets[f],end=mesh.faceOffsets[f+1];
            for (int j=begin+1;j+1<end;++j)
                reference=std::min(reference,triangleDistance(mesh.cellCentres[c],
                    mesh.points[mesh.facePoints[begin]],mesh.points[mesh.facePoints[j]],
                    mesh.points[mesh.facePoints[j+1]]));
        }
        require(mesh.wallDistance[c]==reference,"exact all-face wall distance");
        std::cout << std::hexfloat << mesh.wallDistance[c] << ' ';
    }
    std::cout << mesh.topologyHash << '\n';
    ++checked;
}
void clearWalls(HostMesh& mesh) {
    for (std::size_t f=0;f<mesh.owner.size();++f)
        if (mesh.neighbour[f]<0)mesh.boundaryKind[f]=BoundaryKind::Outlet;
}
void exerciseUpdates(HostMesh mesh) {
    checkAllDistances(mesh);
    for (std::size_t f=0;f<mesh.owner.size();++f)if (mesh.neighbour[f]<0)
        mesh.boundaryKind[f]=(f%3==0?BoundaryKind::NoSlip:
            f%3==1?BoundaryKind::Interface:BoundaryKind::Outlet);
    checkAllDistances(mesh);
    // Change coordinates without changing connectivity or the wall labels.
    // A global affine transform keeps every polygon planar and cell positive.
    for (auto& p:mesh.points)p={1.7*p.x+.2*p.y+11,.9*p.y+.1*p.z-7,1.3*p.z+4};
    checkAllDistances(mesh);
    clearWalls(mesh);
    checkAllDistances(mesh);
    // Re-introduce only the last exterior face, then only the first one. A list
    // cached across rebuilds would retain either no wall or the wrong wall.
    for (int f=int(mesh.owner.size())-1;f>=0;--f)if (mesh.neighbour[f]<0) {
        mesh.boundaryKind[f]=BoundaryKind::Slip;break;
    }
    checkAllDistances(mesh);
    clearWalls(mesh);
    for (std::size_t f=0;f<mesh.owner.size();++f)if (mesh.neighbour[f]<0) {
        mesh.boundaryKind[f]=BoundaryKind::NoSlip;break;
    }
    checkAllDistances(mesh);
}
}
int main() {
    exerciseUpdates(twoHexahedra(0));
    exerciseUpdates(tetrahedralBulk());
    require(checked==12,"all wall-distance update fixtures executed");
    std::cout << "wall-distance fixtures checked: " << checked << '\n';
}
