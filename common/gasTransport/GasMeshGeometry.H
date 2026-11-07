#pragma once
#include "../gpu/PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_GAS_MESH_HD __host__ __device__
#else
#define UGKP_GAS_MESH_HD
#endif
namespace ugkwp {
struct GasFaceGeometry {
    ugkp::gpu::geometry::Point periodicShift{};
    double area=0,delta=0,weight=.5;
};
// The two-point metric policy used by the common CHMT gas adapter. Topology
// distinguishes internal, translational-periodic and ordinary boundary faces;
// a periodic face uses its mapped neighbour centre and paired face centre.
// Keep the original chord/projection arithmetic, including the 5% skew floor.
UGKP_GAS_MESH_HD inline GasFaceGeometry gasFaceGeometry(
    ugkp::gpu::geometry::Point areaVector,ugkp::gpu::geometry::Point faceCentre,
    ugkp::gpu::geometry::Point ownerCentre,bool neighbour,
    ugkp::gpu::geometry::Point neighbourCentre,bool periodic,
    ugkp::gpu::geometry::Point pairedFaceCentre)
{
    using namespace ugkp::gpu::geometry;
    GasFaceGeometry result;result.area=magnitude(areaVector);
    Point chord=faceCentre-ownerCentre;
    if(neighbour)chord=neighbourCentre-ownerCentre;
    if(periodic){result.periodicShift=faceCentre-pairedFaceCentre;
        chord=neighbourCentre+result.periodicShift-ownerCentre;}
    const Point normal=areaVector/result.area;
    const double projection=dot(chord,normal);
    const double axial=projection<0?-projection:projection,skew=.05*magnitude(chord);
    result.delta=1/(axial>skew?axial:skew);
    if(neighbour||periodic){
        const double a=dot(faceCentre-ownerCentre,normal);
        const double own=a<0?-a:a;
        const Point mappedNeighbour=ownerCentre+chord;
        const double b=dot(mappedNeighbour-faceCentre,normal);
        const double other=b<0?-b:b;
        result.weight=(own+other)>0?other/(own+other):.5;
    }
    return result;
}
}
#undef UGKP_GAS_MESH_HD
