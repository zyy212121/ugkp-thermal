#ifndef CHMT_MESH_TRAJECTORY_SURFACE_H
#define CHMT_MESH_TRAJECTORY_SURFACE_H
#include "mesh/Geometry.H"
#include "gpu/ResidentSurfaceSeams.H"
namespace chmt {
// Initialization adapter for the common resident seam owner. Only immutable
// surface adjacency and incident cyclic patch identities are prepared here;
// geometric pairing remains a per-update device operation.
bool buildSurfaceSeamTopology(const HostMesh&,const SurfaceMesh&,
    std::vector<ugkp::gpu::geometry::SurfaceTopologyEdge>&,
    std::vector<ugkp::gpu::geometry::SurfaceSeamPatch>&,std::string& error);

// Rebuild every surface metric from the supplied actual linear point trajectory.
// No new motion solve, interpolation of metrics, or inventory remap occurs.
// Non-geometric data (including prescribed support velocity) are preserved from
// referenceSurface. Edge sweep covers oldSolid -> solidEndpoint over interval.
// Endpoint meshes must have the same topology as their respective bases. The
// whole path is certified independently, and output changes only on success.
bool rebuildTrajectorySurface(const HostMesh& oldGas, const HostMesh& oldSolid,
    const SurfaceMesh& referenceSurface, const HostMesh& gasEndpoint, const HostMesh& solidEndpoint,
    Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
bool rebuildTrajectorySurface(const HostState& base, const HostMesh& gasEndpoint,
    const HostMesh& solidEndpoint, Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{},bool* numericalFailure=nullptr);
}
#endif
