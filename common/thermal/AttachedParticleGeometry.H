#pragma once
#include "../gpu/PolyhedralGeometryMath.H"
#if defined(__CUDACC__)
#define UGKP_ATTACHED_HD __host__ __device__
#else
#define UGKP_ATTACHED_HD
#endif
namespace ugkp { namespace thermal {
using ParticleGeometryPoint=gpu::geometry::Point;
// Face addressing may be reordered through oldToNewFace. Point identities and
// each face's vertex set must persist, as in topology-preserving mesh motion.
// All pointers passed to the CUDA API are resident device storage.
struct ParticleSurfaceGeometry {
    int nFaces=0,nPoints=0;
    const int* faceOffsets=nullptr;
    const int* facePoints=nullptr;
    const int* owner=nullptr;
    const ParticleGeometryPoint* points=nullptr;
};
struct AttachedGeometryResult {
    ParticleGeometryPoint position{};
    int face=-1,cell=-1;bool valid=false;
};
namespace attachedGeometry {
using namespace gpu::geometry;
struct Coordinates {double b=0,c=0;Point closest{};bool valid=false;};
// Closest point in the same vertex-zero fan used by the common polyhedral
// geometry implementation. The normal offset is restored after interpolation.
UGKP_ATTACHED_HD inline Coordinates closestTriangle(Point p,Point a,Point b,Point c) {
    const Point ab=b-a,ac=c-a,ap=p-a;
    const double area2=dot(cross(ab,ac),cross(ab,ac));
    if(!(area2>0)||!std::isfinite(area2))return {};
    const double d1=dot(ab,ap),d2=dot(ac,ap);
    double v=0,w=0;
    if(d1<=0 && d2<=0){} // vertex a
    else {
        const Point bp=p-b;const double d3=dot(ab,bp),d4=dot(ac,bp);
        if(d3>=0 && d4<=d3)v=1;
        else {
            const double vc=d1*d4-d3*d2;
            if(vc<=0 && d1>=0 && d3<=0)v=d1/(d1-d3);
            else {
                const Point cp=p-c;const double d5=dot(ab,cp),d6=dot(ac,cp);
                if(d6>=0 && d5<=d6)w=1;
                else {
                    const double vb=d5*d2-d1*d6;
                    if(vb<=0 && d2>=0 && d6<=0)w=d2/(d2-d6);
                    else {
                        const double va=d3*d6-d5*d4;
                        if(va<=0 && d4-d3>=0 && d5-d6>=0) {
                            w=(d4-d3)/((d4-d3)+(d5-d6));v=1-w;
                        } else {const double sum=va+vb+vc;v=vb/sum;w=vc/sum;}
                    }
                }
            }
        }
    }
    const Point q=a+ab*v+ac*w;
    return {v,w,q,std::isfinite(v)&&std::isfinite(w)&&finite(q)};
}

}
UGKP_ATTACHED_HD inline AttachedGeometryResult mapAttachedParticleGeometry(
    const ParticleSurfaceGeometry& old,const ParticleSurfaceGeometry& next,
    int oldFace,int newFace,ParticleGeometryPoint position) {
    using namespace gpu::geometry;using namespace attachedGeometry;
    AttachedGeometryResult out;
    if(oldFace<0||oldFace>=old.nFaces||newFace<0||newFace>=next.nFaces
       ||!old.faceOffsets||!old.facePoints||!old.points
       ||!next.faceOffsets||!next.facePoints||!next.points||!next.owner)return out;
    const int begin=old.faceOffsets[oldFace],end=old.faceOffsets[oldFace+1];
    const int nb=next.faceOffsets[newFace],ne=next.faceOffsets[newFace+1];
    if(end-begin<3||end-begin!=ne-nb)return out;
    for(int j=nb;j<ne;++j)if(next.facePoints[j]<0||next.facePoints[j]>=next.nPoints)return out;
    for(int k=begin;k<end;++k) {
        if(old.facePoints[k]<0||old.facePoints[k]>=old.nPoints||old.facePoints[k]>=next.nPoints)return out;
        bool found=false;for(int j=nb;j<ne;++j)found=found||old.facePoints[k]==next.facePoints[j];
        if(!found)return out;
    }
    TopologyView a{};a.faceOffsets=old.faceOffsets;a.facePoints=old.facePoints;
    TopologyView b{};b.faceOffsets=next.faceOffsets;b.facePoints=next.facePoints;
    const auto oldGeometry=faceGeometry(oldFace,a,old.points),newGeometry=faceGeometry(newFace,b,next.points);
    if(oldGeometry.invalid||newGeometry.invalid)return out;
    const Point oldN=normalized(oldGeometry.area),newN=normalized(newGeometry.area);
    const int ia=old.facePoints[begin];int ib=-1,ic=-1;Coordinates selected;
    double distance=std::numeric_limits<double>::max();
    for(int k=begin+1;k+1<end;++k) {
        const int kb=old.facePoints[k],kc=old.facePoints[k+1];
        const auto candidate=closestTriangle(position,old.points[ia],old.points[kb],old.points[kc]);
        const double d=dot(position-candidate.closest,position-candidate.closest);
        if(candidate.valid&&std::isfinite(d)&&d<distance){distance=d;selected=candidate;ib=kb;ic=kc;}
    }
    if(ib<0)return out;
    const double offset=dot(position-selected.closest,oldN);
    out.position=next.points[ia]+(next.points[ib]-next.points[ia])*selected.b
        +(next.points[ic]-next.points[ia])*selected.c+newN*offset;
    out.face=newFace;out.cell=next.owner[newFace];
    out.valid=finite(out.position)&&out.cell>=0;
    return out;
}
}}
#undef UGKP_ATTACHED_HD
