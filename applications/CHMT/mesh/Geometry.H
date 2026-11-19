#ifndef CHMT_MESH_GEOMETRY_H
#define CHMT_MESH_GEOMETRY_H
#include "core/HostState.H"
#include <cstring>
namespace chmt {
    inline bool identicalPoints(const std::vector<Vec3>& a,const std::vector<Vec3>& b){
        if(a.size()!=b.size())return false;
        for(std::size_t i=0;i<a.size();++i){
            if(std::memcmp(&a[i].x,&b[i].x,sizeof(Real))
                ||std::memcmp(&a[i].y,&b[i].y,sizeof(Real))
                ||std::memcmp(&a[i].z,&b[i].z,sizeof(Real)))return false;
        }
        return true;
    }


    bool rebuildGeometry(HostMesh&, std::string& error,ugkp::diagnostics::DeviceContext* diagnostics=nullptr);





    bool mapConformalInterface(const HostMesh& gas, const HostMesh& solid,
        const std::vector<int>& solidFaces, const std::vector<Real>& thickness,
        const std::vector<int>& gasCandidates, std::vector<int>& mappedGasFaces,
        std::vector<int>& solidToGas, std::string& error);





    bool makeStageGeometry(const HostMesh&, const std::vector<Vec3>&, double, HostMesh&, std::vector<double>&,
        std::string&, const Tolerances& tolerances=Tolerances{}, bool* numericalFailure=nullptr);
    bool makeStageAreaVectors(const HostMesh&, const std::vector<Vec3>&, std::vector<Vec3>&, std::string&);


    bool surfaceAreaMetricChange(const SurfaceMesh&, std::vector<Real>&, std::string&);
    bool integrateSurfaceEdgeSweep(const HostMesh&, const std::vector<Vec3>&, int faceOwner, int faceNeighbour,
        int pointA, int pointB, int segments, Real&, std::string&);
    Real maximumGclResidual(const HostMesh& oldMesh, const HostMesh& newMesh, const std::vector<Real>& sweeps);
}
#endif
