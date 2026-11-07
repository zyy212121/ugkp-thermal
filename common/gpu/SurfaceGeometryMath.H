#ifndef UGKP_GPU_SURFACE_GEOMETRY_MATH_H
#define UGKP_GPU_SURFACE_GEOMETRY_MATH_H
#include "ResidentSurfaceGeometry.H"
#ifdef __CUDACC__
#define UGKP_SURFACE_GEOMETRY_HD __host__ __device__
#else
#define UGKP_SURFACE_GEOMETRY_HD
#endif
namespace ugkp {namespace gpu {namespace geometry {
UGKP_SURFACE_GEOMETRY_HD inline SurfaceFaceGeometry surfaceFaceGeometry(
    Point oldSolidArea,Point oldSolidCentre,Point solidArea,Point solidCentre,Point solidCell,
    bool hasGas,Point gasArea,Point gasCentre,Point gasCell,double interval){
    SurfaceFaceGeometry result;result.oldArea=magnitude(oldSolidArea);result.area=magnitude(solidArea);
    result.centre=solidCentre;result.normal=normalized(solidArea);result.velocity=(solidCentre-oldSolidCentre)/interval;
    result.solidDistance=dot(solidCentre-solidCell,result.normal);
    result.gasDistance=hasGas?dot(gasCentre-gasCell,normalized(gasArea)):0;return result;
}
}}}
#undef UGKP_SURFACE_GEOMETRY_HD
#endif
