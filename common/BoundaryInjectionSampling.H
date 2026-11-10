#pragma once
#include <cmath>
#include <cstring>
#include "gpu/PolyhedralGeometryMath.H"

#ifdef __CUDACC__
#define UGKP_INJECTION_HD __host__ __device__
#else
#define UGKP_INJECTION_HD
#endif

namespace ugkwp {
// A face-local cumulative area and an affine triangle, in physical 3D space.
struct BoundaryInjectionTriangle {
    double a[3], ab[3], ac[3], cumulativeArea;
};


// Scratch contains face-local indices. Both output arrays have vertexCount
// capacity. Geometry is built once on the same Sf/Cf plane used by tracking;
// ear clipping, collinearity, areas and sampled positions all use that plane.
UGKP_INJECTION_HD inline ugkp::gpu::geometry::Point injectionPlanePoint(
    ugkp::gpu::geometry::Point point,ugkp::gpu::geometry::Point centre,
    ugkp::gpu::geometry::Point normal)
{
    using namespace ugkp::gpu::geometry;
    const Point relative=point-centre;
    return relative-normal*dot(relative,normal);
}

UGKP_INJECTION_HD inline int buildBoundaryInjectionTriangles(
    const ugkp::gpu::geometry::Point* points,const int* vertexIds,
    int vertexCount,int* scratch,BoundaryInjectionTriangle* triangles,
    ugkp::gpu::geometry::Point faceCentre,
    ugkp::gpu::geometry::Point faceArea,
    ugkp::gpu::geometry::Point shift)
{
    using namespace ugkp::gpu::geometry;
    if(vertexCount<3)return 0;
    const double norm=magnitude(faceArea);
    if(!(norm>0)||!std::isfinite(norm))return 0;
    Point normal=faceArea/norm;
    const Point origin=injectionPlanePoint(points[vertexIds[0]],faceCentre,normal);
    double scale2=0,signedArea=0;
    for(int i=0;i<vertexCount;++i){
        scratch[i]=i;
        const Point a=injectionPlanePoint(points[vertexIds[i]],faceCentre,normal)-origin;
        const Point b=injectionPlanePoint(points[vertexIds[(i+1)%vertexCount]],faceCentre,normal)-origin;
        signedArea+=dot(cross(a,b),normal);
        const double d=dot(a,a);if(d>scale2)scale2=d;
    }
    if(!(::fabs(signedArea)>0)||!std::isfinite(signedArea))return 0;
    // Vertex ordering may be opposite the owner-side tracking normal.
    if(signedArea<0)normal=normal*(-1);
    const double tolerance=32*2.2204460492503130808472633361816e-16*scale2;
    int remaining=vertexCount,count=0;
    double area=0;
    while(remaining>=3){
        bool removed=false;
        for(int i=0;i<remaining;++i){
            const int before=(i+remaining-1)%remaining,after=(i+1)%remaining;
            const Point a=injectionPlanePoint(points[vertexIds[scratch[before]]],faceCentre,normal);
            const Point b=injectionPlanePoint(points[vertexIds[scratch[i]]],faceCentre,normal);
            const Point c=injectionPlanePoint(points[vertexIds[scratch[after]]],faceCentre,normal);
            const double turn=dot(cross(b-a,c-a),normal);
            // Collinearity and the between-neighbours test use projected points.
            if(::fabs(turn)<=tolerance && dot(b-a,b-c)<=tolerance){
                for(int k=i;k+1<remaining;++k)scratch[k]=scratch[k+1];
                --remaining;removed=true;break;
            }
            if(turn<=tolerance)continue;
            bool occupied=false;
            for(int k=0;k<remaining;++k){
                if(k==before||k==i||k==after)continue;
                const Point q=injectionPlanePoint(points[vertexIds[scratch[k]]],faceCentre,normal);
                if(dot(cross(b-a,q-a),normal)>=-tolerance
                    &&dot(cross(c-b,q-b),normal)>=-tolerance
                    &&dot(cross(a-c,q-c),normal)>=-tolerance){
                    occupied=true;break;
                }
            }
            if(occupied)continue;
            const Point ab=b-a,ac=c-a,position=faceCentre+a+shift;
            const double triangleArea=.5*magnitude(cross(ab,ac));
            if(!(triangleArea>0)||!std::isfinite(triangleArea))return 0;
            area+=triangleArea;
            auto& t=triangles[count++];
            t.a[0]=position.x;t.a[1]=position.y;t.a[2]=position.z;
            t.ab[0]=ab.x;t.ab[1]=ab.y;t.ab[2]=ab.z;
            t.ac[0]=ac.x;t.ac[1]=ac.y;t.ac[2]=ac.z;
            t.cumulativeArea=area;
            for(int k=i;k+1<remaining;++k)scratch[k]=scratch[k+1];
            --remaining;removed=true;break;
        }
        // Unusable geometry leaves the established point-source fallback.
        if(!removed)return 0;
    }
    return count;
}

UGKP_INJECTION_HD inline unsigned long long injectionMix(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27))*0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Position draws do not consume the diameter/thermal RNG, depend on atomic
// particle-slot allocation, or require new checkpoint state.
UGKP_INJECTION_HD inline unsigned long long injectionPositionSeed
(unsigned long long seed, int face, double time, int ordinal) {
    unsigned long long bits;
#ifdef __CUDA_ARCH__
    bits = static_cast<unsigned long long>(__double_as_longlong(time));
#else
    std::memcpy(&bits, &time, sizeof(bits));
#endif
    return injectionMix(seed ^ injectionMix(static_cast<unsigned long long>(face))
        ^ injectionMix(bits) ^ injectionMix(static_cast<unsigned long long>(ordinal)
        + 0xd1b54a32d192ed03ULL));
}

UGKP_INJECTION_HD inline double injectionUniform(unsigned long long& rng) {
    rng = injectionMix(rng);
    return (static_cast<double>(rng >> 12) + 0.5)*2.2204460492503130808472633361816e-16;
}

UGKP_INJECTION_HD inline bool sampleBoundaryInjection
(const BoundaryInjectionTriangle* triangles, int begin, int end,
 unsigned long long rng, double& x, double& y, double& z) {
    // Callers without a polygon (including existing point-source API users)
    // retain their previous source position. No particle or mass is rejected.
    if (!triangles || begin == end) return false;
    const double target = injectionUniform(rng)*triangles[end - 1].cumulativeArea;
    int lo = begin, hi = end - 1;
    while (lo < hi) {
        const int mid = lo + (hi - lo)/2;
        if (target < triangles[mid].cumulativeArea) hi = mid;
        else lo = mid + 1;
    }
    const auto& t = triangles[lo];
    const double root = ::sqrt(injectionUniform(rng));
    const double u = root*(1.0 - injectionUniform(rng));
    const double v = root - u;
    x = t.a[0] + u*t.ab[0] + v*t.ac[0];
    y = t.a[1] + u*t.ab[1] + v*t.ac[1];
    z = t.a[2] + u*t.ab[2] + v*t.ac[2];
    return true;
}
}
#undef UGKP_INJECTION_HD
