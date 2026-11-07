#ifndef UGKP_GPU_GEOMETRY_BOUNDARY_MATH_H
#define UGKP_GPU_GEOMETRY_BOUNDARY_MATH_H
#include "PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_GEOMETRY_BOUNDARY_HD __host__ __device__
#else
#define UGKP_GEOMETRY_BOUNDARY_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
namespace boundaryDetail {
template<class Vec> UGKP_GEOMETRY_BOUNDARY_HD inline double length(Vec p){return ::sqrt(p.x*p.x+p.y*p.y+p.z*p.z);}
template<class Vec> UGKP_GEOMETRY_BOUNDARY_HD inline double coordinate(Vec p,int axis){return axis==0?p.x:axis==1?p.y:p.z;}
}
// Existing translational periodic geometry contract. The first partner within
// tolerance is selected, then checked for duplicate use by the owning face.
// This deliberately is not a nearest-neighbour or best-fit correspondence.
template<class Vec>
UGKP_GEOMETRY_BOUNDARY_HD inline bool periodicAreaMatches(Vec a,Vec b){
    const double first=boundaryDetail::length(a),second=boundaryDetail::length(b);
    return !(boundaryDetail::length(a+b)>1e-10*(first>second?first:second));
}
template<class Vec>
UGKP_GEOMETRY_BOUNDARY_HD inline int periodicVertexMatch(int vertex,int partner,
    TopologyView topology,const Vec* points,Vec shift,double tolerance){
    for(int j=topology.faceOffsets[partner];j<topology.faceOffsets[partner+1];++j)
        if(!(boundaryDetail::length(points[vertex]+shift-points[topology.facePoints[j]])>tolerance))return j;
    return -1;
}
enum class EmptyTrajectoryFailure : int {None,NormalDisplacement,PairedTangentialDisplacement};
template<class Vec>
UGKP_GEOMETRY_BOUNDARY_HD inline EmptyTrajectoryFailure emptyTrajectoryPoint(
    int p,int mate,int axis,const Vec* old,const Vec* end){
    if(boundaryDetail::coordinate(end[p],axis)!=boundaryDetail::coordinate(old[p],axis))
        return EmptyTrajectoryFailure::NormalDisplacement;
    for(int d=0;d<3;++d)if(d!=axis&&boundaryDetail::coordinate(end[p],d)!=boundaryDetail::coordinate(end[mate],d))
        return EmptyTrajectoryFailure::PairedTangentialDisplacement;
    return EmptyTrajectoryFailure::None;
}
struct EmptyCoordinateKey {double u=0,v=0,normal=0;};
struct EmptyCoordinateLess {
    UGKP_GEOMETRY_BOUNDARY_HD bool operator()(EmptyCoordinateKey a,EmptyCoordinateKey b)const{
        return a.u<b.u||(!(b.u<a.u)&&(a.v<b.v||(!(b.v<a.v)&&a.normal<b.normal)));
    }
};
UGKP_GEOMETRY_BOUNDARY_HD inline bool sameEmptyCoordinate(EmptyCoordinateKey a,EmptyCoordinateKey b){
    return a.u==b.u&&a.v==b.v&&a.normal==b.normal;
}
}}}
#undef UGKP_GEOMETRY_BOUNDARY_HD
#endif
