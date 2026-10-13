#ifndef CHMT_MESH_GEOMETRY_H
#define CHMT_MESH_GEOMETRY_H
#include "core/HostState.H"
namespace chmt {
    // Geometry routines mutate only on success. Vertices follow straight paths
    // over this stage; an arbitrary time trajectory must be split into stages.
    bool rebuildGeometry(HostMesh&, std::string& error);
    // Canonical static/moving interface correspondence. Base-vertex normals and
    // thickness are area weighted over the selected solid faces. Gas vertices
    // must equal X_s + n_vertex*delta_vertex with reversed cyclic face topology.
    // Genuine offsets may have unequal gas/base face areas. Outputs change only
    // on success. Empty gas candidates represent a wholly uncoupled surface.
    bool mapConformalInterface(const HostMesh& gas, const HostMesh& solid,
        const std::vector<int>& solidFaces, const std::vector<Real>& thickness,
        const std::vector<int>& gasCandidates, std::vector<int>& mappedGasFaces,
        std::vector<int>& solidToGas, std::string& error);

    // Moving stages require an analytic orientation/convex-half-space certificate.
    // Uncertifiable concave/folded motion is rejected; no sample-only acceptance.
    // Configured tolerances bound GCL and periodic swept-volume residuals only;
    // topology, orientation and area-closure validity remain independent.
    bool makeStageGeometry(const HostMesh&, const std::vector<Vec3>&, double, HostMesh&, std::vector<double>&,
        std::string&, const Tolerances& tolerances=Tolerances{}, bool* numericalFailure=nullptr);
    bool makeStageAreaVectors(const HostMesh&, const std::vector<Vec3>&, std::vector<Vec3>&, std::string&);
    // Surface diagnostic is physical normal/hinge area change, NOT a source
    // correction and NOT a target that must be forced to zero. Units m^2.
    bool surfaceAreaMetricChange(const SurfaceMesh&, std::vector<Real>&, std::string&);
    bool integrateSurfaceEdgeSweep(const HostMesh&, const std::vector<Vec3>&, int faceOwner, int faceNeighbour,
        int pointA, int pointB, int segments, Real&, std::string&);
    namespace detail {
    bool completeConformalInterface(const HostMesh&,const HostMesh&,const std::vector<int>&,
        const std::vector<Real>&,const std::vector<int>&,std::vector<int>&,std::vector<int>&,std::string&);
    bool completeGeometry(HostMesh&,std::string&,bool* numericalFailure=nullptr);
    bool completeStageAreaVectors(const HostMesh&,const std::vector<Vec3>&,std::vector<Vec3>&,std::string&,bool* numericalFailure=nullptr);
    bool completeStageGeometry(const HostMesh&,const std::vector<Vec3>&,double,HostMesh&,std::vector<double>&,
        std::string&,const Tolerances& =Tolerances{},bool* numericalFailure=nullptr);
    bool completeHarmonicPointMotion(const HostMesh&,const std::vector<Vec3>&,const std::vector<unsigned char>&,
        std::vector<Vec3>&,std::string&,bool* numericalFailure=nullptr);
    }
    Real maximumGclResidual(const HostMesh& oldMesh, const HostMesh& newMesh, const std::vector<Real>& sweeps);
}
#endif
