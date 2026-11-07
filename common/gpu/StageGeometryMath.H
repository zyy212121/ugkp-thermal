#ifndef UGKP_GPU_STAGE_GEOMETRY_MATH_H
#define UGKP_GPU_STAGE_GEOMETRY_MATH_H
#include "PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_STAGE_GEOMETRY_HD __host__ __device__
#else
#define UGKP_STAGE_GEOMETRY_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
constexpr double stageGeometryEpsilon=std::numeric_limits<double>::epsilon();
// Preserve the original three interpolation expressions, including the
// represented Empty-coordinate arithmetic. Algebraic simplification changes
// whether a stationary sample has exactly the accepted coordinates.
template<class Vec>
UGKP_STAGE_GEOMETRY_HD inline Vec stageSample(Vec old,Vec endpoint,int sample,int emptyAxis){
    if(sample==0)return emptyAxis<0?(old+endpoint)*.5:old+(endpoint-old)*.5;
    if(sample==1)return emptyAxis<0?old*(2.0/3)+endpoint/3:old+(endpoint-old)/3;
    return emptyAxis<0?old/3+endpoint*(2.0/3):old+(endpoint-old)*(2.0/3);
}
UGKP_STAGE_GEOMETRY_HD inline bool stageInteriorVolumeInverts(double v0,double v1,double v2,double v3){
    const double a=4.5*(v3-3*v2+3*v1-v0),b=4.5*(v2-2*v1+v0)-a,d=3*(v1-v0)-a/9-b/3;
    const double small=256*stageGeometryEpsilon*(v0>v3?v0:v3);
    double critical[2]={-1,-1};
    if(::fabs(a)<=small){if(::fabs(b)>small)critical[0]=-d/(2*b);}
    else{const double disc=4*b*b-12*a*d;
        if(disc>=0){const double root=::sqrt(disc);critical[0]=(-2*b-root)/(6*a);critical[1]=(-2*b+root)/(6*a);}}
    for(double t:critical)if(t>0&&t<1&&((a*t+b)*t+d)*t+v0<=0)return true;
    return false;
}
struct StageCellConservation {
    double closure=0,closureLimit=0,residual=0,residualLimit=0;
    bool invalid=false;
};
template<class Area,class Sweep>
UGKP_STAGE_GEOMETRY_HD inline StageCellConservation stageCellConservation(
    int cell,TopologyView topology,double oldVolume,double newVolume,Area area,Sweep sweep,
    double absoluteTolerance,double relativeTolerance){
    double sum=0,areaSum=0;decltype(area(0)) closure{};
    for(int k=topology.cellOffsets[cell];k<topology.cellOffsets[cell+1];++k){
        const int face=topology.cellFaces[k],sign=topology.cellSigns[k];
        sum+=sign*sweep(face);closure+=area(face)*sign;const auto a=area(face);
        areaSum+=::sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
    }
    StageCellConservation result;
    result.invalid=!std::isfinite(closure.x)||!std::isfinite(closure.y)||!std::isfinite(closure.z)||!std::isfinite(areaSum);
    result.residualLimit=absoluteTolerance+relativeTolerance*(oldVolume<newVolume?newVolume:oldVolume);
    if(result.invalid)return result;
    result.closure=::sqrt(closure.x*closure.x+closure.y*closure.y+closure.z*closure.z);
    result.closureLimit=1e-10*areaSum;result.residual=newVolume-oldVolume-sum;
    return result;
}
}}}
#undef UGKP_STAGE_GEOMETRY_HD
#endif
