#ifndef UGKP_GPU_NORMAL_HEIGHT_MOTION_MATH_H
#define UGKP_GPU_NORMAL_HEIGHT_MOTION_MATH_H
#include "PointMotionMath.H"
#ifdef __CUDACC__
#define UGKP_HEIGHT_HD __host__ __device__
#else
#define UGKP_HEIGHT_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
struct WallNormalHeightGeometry {
    Point referenceCentre{},currentCentre{},referenceNormal{},currentNormal{};
};
namespace normalHeightDetail {
template<class P> UGKP_HEIGHT_HD inline Point point(P p){return {p.x,p.y,p.z};}
}
// Preserve the arithmetic vertex-centre and oriented polygon normal used by
// the original relative near-wall spacing constraint, including polygon order.
template<class Reference,class Points,class Displacement>
UGKP_HEIGHT_HD inline WallNormalHeightGeometry wallNormalHeightGeometry(TopologyView topology,int face,
    const Reference* reference,const Points* points,const Displacement* displacement){
    using normalHeightDetail::point;WallNormalHeightGeometry result;Point a0{},a1{};
    const int lo=topology.faceOffsets[face],hi=topology.faceOffsets[face+1];
    for(int k=lo;k<hi;++k){const int v=topology.facePoints[k];
        result.referenceCentre+=point(reference[v]);result.currentCentre+=point(points[v])+point(displacement[v]);}
    result.referenceCentre=result.referenceCentre/double(hi-lo);result.currentCentre=result.currentCentre/double(hi-lo);
    for(int k=lo;k<hi;++k){const int v=topology.facePoints[k],w=topology.facePoints[k+1==hi?lo:k+1];
        a0+=cross(point(reference[v])-result.referenceCentre,point(reference[w])-result.referenceCentre);
        a1+=cross(point(points[v])+point(displacement[v])-result.currentCentre,
            point(points[w])+point(displacement[w])-result.currentCentre);}
    result.referenceNormal=normalized(a0)*-1;result.currentNormal=normalized(a1)*-1;return result;
}
// Returns infinity when this point has no positive resolved reference height.
// Every accepted contribution uses its own relative height, never an absolute
// mesh-size allowance shared between low-Re and wall-model grids.
UGKP_HEIGHT_HD inline double addWallNormalHeight(MotionPenalty& penalty,const WallNormalHeightGeometry& wall,
    Point reference,Point point){
    const double height=dot(reference-wall.referenceCentre,wall.referenceNormal);
    if(height>1e-12*magnitude(reference-wall.referenceCentre)){
        const double target=height-dot(point-wall.currentCentre,wall.currentNormal);
        addNormalHeight(penalty,wall.currentNormal,target,height);return height;
    }
    return std::numeric_limits<double>::infinity();
}
UGKP_HEIGHT_HD inline double normalHeightMotionTolerance(double geometricTolerance,double minimumHeight){
    if(minimumHeight>0){const double constrained=1e-10*minimumHeight;
        if(constrained<geometricTolerance)return constrained;}
    return geometricTolerance;
}
}}}
#undef UGKP_HEIGHT_HD
#endif
