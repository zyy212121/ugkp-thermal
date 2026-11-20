#pragma once
#include "mesh/Geometry.H"

namespace chmt {
class MeshGeometryCache {
    HostMesh mesh_;
    bool ready_=false;
public:
    bool topologyMatches(const HostMesh& mesh) const {
        return ready_ && mesh_.topologyHash==mesh.topologyHash
            && mesh_.owner==mesh.owner && mesh_.neighbour==mesh.neighbour
            && mesh_.faceOffsets==mesh.faceOffsets && mesh_.facePoints==mesh.facePoints
            && mesh_.cellFaceOffsets==mesh.cellFaceOffsets && mesh_.cellFaces==mesh.cellFaces
            && mesh_.cellFaceSigns==mesh.cellFaceSigns && mesh_.boundaryKind==mesh.boundaryKind
            && mesh_.periodicPartner==mesh.periodicPartner;
    }
    bool matches(const HostMesh& mesh) const {
        return topologyMatches(mesh)
            && identicalPoints(mesh_.points,mesh.points)
            && identicalPoints(mesh_.cellCentres,mesh.cellCentres)
            && identicalPoints(mesh_.areaVectors,mesh.areaVectors)
            && identicalPoints(mesh_.faceCentres,mesh.faceCentres)
            && mesh_.volumes==mesh.volumes;
    }
    void remember(const HostMesh& mesh) {
        mesh_.topologyHash=mesh.topologyHash;
        mesh_.owner=mesh.owner;mesh_.neighbour=mesh.neighbour;
        mesh_.faceOffsets=mesh.faceOffsets;mesh_.facePoints=mesh.facePoints;
        mesh_.cellFaceOffsets=mesh.cellFaceOffsets;mesh_.cellFaces=mesh.cellFaces;
        mesh_.cellFaceSigns=mesh.cellFaceSigns;mesh_.boundaryKind=mesh.boundaryKind;
        mesh_.periodicPartner=mesh.periodicPartner;
        mesh_.points=mesh.points;mesh_.cellCentres=mesh.cellCentres;
        mesh_.areaVectors=mesh.areaVectors;mesh_.faceCentres=mesh.faceCentres;mesh_.volumes=mesh.volumes;
        ready_=true;
    }
    void clear() { ready_=false; }
};
}
