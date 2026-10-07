#ifndef CHMT_MESH_MOTION_H
#define CHMT_MESH_MOTION_H
#include "mesh/Geometry.H"
namespace chmt {
    bool harmonicPointMotion(const HostMesh&, const std::vector<Vec3>& boundaryDisplacement,
        const std::vector<unsigned char>& prescribed, std::vector<Vec3>& endpoint, std::string&);
    bool moveCoupledMeshes(const HostState&, const std::vector<FilmAux>&, double, HostMesh&, HostMesh&,
        SurfaceMesh&, std::string&);
}
#endif
