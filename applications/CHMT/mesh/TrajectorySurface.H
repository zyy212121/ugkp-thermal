#ifndef CHMT_MESH_TRAJECTORY_SURFACE_H
#define CHMT_MESH_TRAJECTORY_SURFACE_H
#include "mesh/Geometry.H"
#include "gpu/ResidentSurfaceSeams.H"
namespace chmt {



bool buildSurfaceSeamTopology(const HostMesh&,const SurfaceMesh&,
    std::vector<ugkp::gpu::geometry::SurfaceTopologyEdge>&,
    std::vector<ugkp::gpu::geometry::SurfaceSeamPatch>&,std::string& error);







bool rebuildTrajectorySurface(const HostMesh& oldGas, const HostMesh& oldSolid,
    const SurfaceMesh& referenceSurface, const HostMesh& gasEndpoint, const HostMesh& solidEndpoint,
    Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
bool rebuildTrajectorySurface(const HostState& base, const HostMesh& gasEndpoint,
    const HostMesh& solidEndpoint, Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
}
#endif
