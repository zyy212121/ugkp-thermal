#pragma once
#include <array>
namespace ugkwp { namespace gaswall {
struct WallPolygonTriangle {int a,b,c;};
struct WallPolygonMetric {
    std::array<double,3> normal{},centre{};double scale=0;
    int triangles=0,valid=0;
};
struct WallPolygonTopology {
    int faces=0;
    const int *faceOffsets=nullptr,*facePoints=nullptr,*triangleOffsets=nullptr;
};
struct WallPolygonWorkspace {
    // Ring entries share the immutable face CSR offsets. Triangle offsets
    // reserve N-2 entries per face. All storage is allocated at initialization.
    int* ring=nullptr;WallPolygonTriangle* triangles=nullptr;
    WallPolygonMetric* metric=nullptr;int* firstInvalidFace=nullptr;
};
struct WallMatchingTopology {
    WallPolygonTopology polygons{};int cells=0;
    // Sorted by global face index within each cell, as in the native builder.
    const int *cellOffsets=nullptr,*cellFaces=nullptr,*owner=nullptr,*neighbour=nullptr;
    const unsigned char *physicalWall=nullptr,*constrained=nullptr,*gasCell=nullptr;
};
struct WallMatchingGeometry {
    int wallFace=-1,ownerCell=-1,donor=-1,failureFace=-1;
    std::array<double,3> origin{},normal{},tangent1{},tangent2{},matchingPoint{},firstMomentVector{};
    double ownerDistance=0,matchingDistance=0,maximumOwnerDistance=0,matchingDonorDistance=0;
    double volume=0,geometryVolume=0,firstMoment=0;
};
struct WallMatchingStatus {int firstInvalidRow=-1,invalidFace=-1;};
}}
