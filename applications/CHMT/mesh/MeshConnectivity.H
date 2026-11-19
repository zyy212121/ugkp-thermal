#pragma once
#include "core/HostState.H"
namespace chmt {


inline bool sameMeshConnectivity(const HostMesh& a,const HostMesh& b){
    return a.points.size()==b.points.size()&&a.faceOffsets==b.faceOffsets&&a.facePoints==b.facePoints
        &&a.owner==b.owner&&a.neighbour==b.neighbour&&a.periodicPartner==b.periodicPartner
        &&a.cellFaceOffsets==b.cellFaceOffsets&&a.cellFaces==b.cellFaces&&a.cellFaceSigns==b.cellFaceSigns
        &&a.boundaryKind==b.boundaryKind;
}
}
