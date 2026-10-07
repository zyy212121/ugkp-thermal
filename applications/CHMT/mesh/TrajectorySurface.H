#ifndef CHMT_MESH_TRAJECTORY_SURFACE_H
#define CHMT_MESH_TRAJECTORY_SURFACE_H
#include "mesh/Geometry.H"
namespace chmt {
// Rebuild every surface metric from the supplied actual linear point trajectory.
// No new motion solve, interpolation of metrics, or inventory remap occurs.
// Non-geometric data (including prescribed support velocity) are preserved from
// referenceSurface. Edge sweep covers oldSolid -> solidEndpoint over interval.
// Endpoint meshes must have the same topology as their respective bases. The
// whole path is certified independently, and output changes only on success.
bool rebuildTrajectorySurface(const HostMesh& oldGas, const HostMesh& oldSolid,
    const SurfaceMesh& referenceSurface, const HostMesh& gasEndpoint, const HostMesh& solidEndpoint,
    Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{});
bool rebuildTrajectorySurface(const HostState& base, const HostMesh& gasEndpoint,
    const HostMesh& solidEndpoint, Real interval, SurfaceMesh& output, std::string& error, const Tolerances& tolerances=Tolerances{});
}
#endif
