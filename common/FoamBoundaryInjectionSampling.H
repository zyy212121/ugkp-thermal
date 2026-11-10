#pragma once
#include "BoundaryInjectionApi.H"
#include "fvMesh.H"
#include <vector>
namespace Foam {
// Application adapter only: polygon mathematics and sampling live in common.
struct BoundaryInjectionGeometry {
    std::vector<int> offsets{0},vertices;
    std::vector<ugkp::gpu::geometry::Point> points;
    void append(const fvMesh& mesh,const label faceI) {
        const face& f=mesh.faces()[faceI];
        forAll(f,i) {
            const auto& p=mesh.points()[f[i]];
            vertices.push_back(int(points.size()));points.push_back({p.x(),p.y(),p.z()});
        }
        offsets.push_back(int(vertices.size()));
    }
    int upload(void* owner)const {
        return ugkwpGpuResidentStrictUploadBoundarySourceGeometry(owner,
            int(vertices.size()),offsets.data(),vertices.data(),int(points.size()),points.data());
    }
};
}
