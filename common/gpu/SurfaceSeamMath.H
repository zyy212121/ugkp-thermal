#pragma once
#include "PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_SURFACE_SEAM_HD __host__ __device__
#else
#define UGKP_SURFACE_SEAM_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
struct SurfaceTopologyEdge {
    int a=-1,b=-1,owner=-1,neighbour=-1,neighbourA=-1,neighbourB=-1;
};
struct SurfaceSeamMatch {int edge=-1;bool ambiguous=false;};
template<class P>
UGKP_SURFACE_SEAM_HD inline bool surfaceSeamPairMatches(const SurfaceTopologyEdge& edge,
    const SurfaceTopologyEdge& other,const P* points,P translation,double tolerance)
{
    const auto first=points[edge.a]+translation-points[other.b];
    const double da=::sqrt(first.x*first.x+first.y*first.y+first.z*first.z);
    if(!(da<=tolerance))return false;
    const auto second=points[edge.b]+translation-points[other.a];
    const double db=::sqrt(second.x*second.x+second.y*second.y+second.z*second.z);
    return db<=tolerance;
}
// Preserve the native ordered search and both endpoint distance checks. The
// removed mask is updated by the caller after each accepted pair, so a later
// edge cannot reuse an edge consumed by an earlier one.
template<class P>
UGKP_SURFACE_SEAM_HD inline SurfaceSeamMatch matchSurfaceSeam(int index,int count,
    const SurfaceTopologyEdge* edges,const unsigned char* removed,const P* points,
    P translation,double tolerance)
{
    SurfaceSeamMatch result;const auto edge=edges[index];
    for(int j=0;j<count;++j)if(index!=j&&!removed[j]&&edges[j].neighbour<0
        &&surfaceSeamPairMatches(edge,edges[j],points,translation,tolerance)){
        if(result.edge>=0){result.ambiguous=true;return result;}
        result.edge=j;
    }
    return result;
}
}}}

#undef UGKP_SURFACE_SEAM_HD
