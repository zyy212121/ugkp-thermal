#pragma once
#include "PolyhedralGeometryMath.H"
#ifdef __CUDACC__
#define UGKP_MOTION_HD __host__ __device__
#else
#define UGKP_MOTION_HD
#endif
namespace ugkp { namespace gpu { namespace geometry {
// Soft normal-height constraints augment the harmonic Dirichlet energy.
// The matrix is positive semidefinite; physical boundary points remain exact.
struct MotionPenalty {
    double xx=0,xy=0,xz=0,yy=0,yz=0,zz=0;
    Point rhs{};
};
UGKP_MOTION_HD inline void addNormalHeight(MotionPenalty& p,Point n,double displacement,double height,double strength=1000) {
    const double w=strength/height;
    p.xx+=w*n.x*n.x;p.xy+=w*n.x*n.y;p.xz+=w*n.x*n.z;
    p.yy+=w*n.y*n.y;p.yz+=w*n.y*n.z;p.zz+=w*n.z*n.z;
    p.rhs+=n*(w*displacement);
}
UGKP_MOTION_HD inline Point motionUpdate(Point neighbourSum,double harmonicWeight,const MotionPenalty& p) {
    // Cholesky of harmonicWeight*I + sum(w*n*n^T). No determinant cancellation.
    const double l00=::sqrt(harmonicWeight+p.xx),l10=p.xy/l00,l20=p.xz/l00;
    const double l11=::sqrt(harmonicWeight+p.yy-l10*l10),l21=(p.yz-l20*l10)/l11;
    const double l22=::sqrt(harmonicWeight+p.zz-l20*l20-l21*l21);
    const Point b=neighbourSum+p.rhs;
    const double y0=b.x/l00,y1=(b.y-l10*y0)/l11,y2=(b.z-l20*y0-l21*y1)/l22;
    const double z=y2/l22,y=(y1-l21*z)/l11,x=(y0-l10*y-l20*z)/l00;
    return {x,y,z};
}
}}}

#undef UGKP_MOTION_HD
