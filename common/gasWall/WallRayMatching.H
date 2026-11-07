#pragma once
#include "WallPolygonMath.H"
#include <algorithm>
#include <limits>
#ifdef __CUDACC__
#define UGKP_WALL_RAY_HD __host__ __device__
#else
#define UGKP_WALL_RAY_HD
#endif
namespace ugkwp { namespace gaswall { namespace wallGeometryDetail {
template<class R> UGKP_WALL_RAY_HD bool rayTriangle(const Vec<R>&o,const Vec<R>&n,const Vec<R>&a,const Vec<R>&b,const Vec<R>&c,R&time){
    const Vec<R> e1=sub(b,a),e2=sub(c,a),p=cross(n,e2);const R det=dot(e1,p);
    const R eps=R(256)*std::numeric_limits<R>::epsilon();
    // A finite shallow crossing still intersects the represented triangle.
    if(det==R(0))return false;
    const Vec<R> delta=sub(o,a),q=cross(delta,e1);
    const R u=dot(delta,p)/det,v=dot(n,q)/det;
    if(u<-eps||v<-eps||u+v>R(1)+eps)return false;
    time=dot(e2,q)/det;return gasFinite(time);
}
// The ray origin follows represented polygon concavity, before any ray test.
template<class R,class Mesh,class Context,class Descriptor>
UGKP_WALL_RAY_HD void initializeWallRayOrigin(const Mesh& m,const Context& ctx,int wallFace,Descriptor& d){
    d.origin=ctx.faces[wallFace].centre;d.normal=mul(ctx.faces[wallFace].normal,R(-1));
    // Choose the construction from polygon topology before tracing a ray:
    // convex polygons use their area centroid; concave polygons use an
    // interior triangle. Neither an attempted ray nor a failed point test
    // selects a second construction.
    bool concave=false;
    const int begin=m.faceOffsets[wallFace],end=m.faceOffsets[wallFace+1],count=end-begin;
    for(int i=0;i<count;++i)
        if(turn(m.points[m.facePoints[begin+i]],m.points[m.facePoints[begin+(i+1)%count]],
            m.points[m.facePoints[begin+(i+2)%count]],ctx.faces[wallFace].normal)<0)concave=true;
    if(concave){
        R largest=0;
        for(const auto&t:ctx.faces[wallFace].triangles){
            const auto&a=m.points[t.a],&b=m.points[t.b],&c=m.points[t.c];
            const R area=norm(cross(sub(b,a),sub(c,a)));
            if(area>largest){largest=area;d.origin=add(a,mul(add(sub(b,a),sub(c,a)),R(1)/R(3)));}
        }
    }
}
template<class R,class Mesh,class Context,class Descriptor>
UGKP_WALL_RAY_HD void initializeWallRayFrame(const Mesh& m,const Context& ctx,Descriptor& d){
    for(int f:ctx.cellFaces[d.ownerCell])for(int k=m.faceOffsets[f];k<m.faceOffsets[f+1];++k){
        R y=dot(sub(m.points[m.facePoints[k]],d.origin),d.normal);
        d.maximumOwnerDistance=std::max(d.maximumOwnerDistance,y);
    }
    d.ownerDistance=dot(sub(m.cellCentres[d.ownerCell],d.origin),d.normal);
    int axis=0;for(int k=1;k<3;++k)if(std::abs(d.normal[k])<std::abs(d.normal[axis]))axis=k;
    Vec<R> coordinate{};coordinate[axis]=1;d.tangent1=cross(d.normal,coordinate);d.tangent1=mul(d.tangent1,R(1)/norm(d.tangent1));d.tangent2=cross(d.normal,d.tangent1);
}
// Returns the first admissible donor reached along the existing outward
// crossings. No ray-cell iteration cap, geometric snapping, or alternate ray
// is added. Reconstruction of the selected donor is a separate common step.
template<class R,class Mesh,class Context,class Descriptor>
UGKP_WALL_RAY_HD int locateWallRayDonor(const Mesh& m,const Context& ctx,Descriptor& d,
    bool reconstructMatching,int& failureFace){
    R entry=0;int cell=d.ownerCell;



    for(;;){
        if(cell<0||(!m.gasCell.empty()&&!m.gasCell[cell])){failureFace=d.wallFace;return -1;}
        R exit=std::numeric_limits<R>::max();int exitFace=-1;bool ambiguous=false;
        for(int f:ctx.cellFaces[cell])for(const auto&t:ctx.faces[f].triangles){
            // Exit crossings follow the current cell's outward triangle winding.
            // Ingress triangles (including the starting wall) cannot be exits,
            // even when their computed ray position rounds just above entry.
            const auto triangleNormal=cross(sub(m.points[t.b],m.points[t.a]),sub(m.points[t.c],m.points[t.a]));
            const R crossing=dot(d.normal,triangleNormal)*(m.owner[f]==cell?R(1):R(-1));
            if(crossing<=0)continue;
            R hit=0;if(!rayTriangle(d.origin,d.normal,m.points[t.a],m.points[t.b],m.points[t.c],hit)||hit<=entry)continue;
            if(hit<exit){exit=hit;exitFace=f;ambiguous=false;}
            else if(hit==exit&&f!=exitFace){ambiguous=true;}
        }
        if(exitFace<0||ambiguous){failureFace=d.wallFace;return -1;}
        const R lower=std::max(entry,d.maximumOwnerDistance);
        if(cell!=d.ownerCell&&!ctx.constrained[cell]&&exit>lower){
            // Reconstructed gas closure ends at the owner/bulk interface. Extending
            // to a bulk cell centre would model part of that cell while its
            // unchanged macro source still covers its whole volume.
            d.matchingDistance=reconstructMatching?lower:lower+(exit-lower)/R(2);d.matchingPoint=add(d.origin,mul(d.normal,d.matchingDistance));
            d.matchingDonorDistance=norm(sub(m.cellCentres[cell],d.matchingPoint));
            return cell;
        }
        if(m.physicalWall[exitFace]){failureFace=exitFace;return -1;}
        int next=m.owner[exitFace]==cell?m.neighbour[exitFace]:m.owner[exitFace];
        if(next<0){failureFace=exitFace;return -1;}
        entry=exit;cell=next;
    }
}
// Signed owner moments for the production path without optional quadrature.
// Ear order and cell-face order remain the same as the native builder.
template<class R,class Mesh,class Context>
UGKP_WALL_RAY_HD void evaluateWallOwnerMoments(const Mesh& m,const Context& ctx,int cell,
    const Vec<R>& seed,R& volume,Vec<R>& moment){
    volume=0;moment={};
    for(int f:ctx.cellFaces[cell])for(auto t:ctx.faces[f].triangles){
        if(m.owner[f]!=cell){const int temp=t.b;t.b=t.c;t.c=temp;}
        const auto&a=m.points[t.a],&b=m.points[t.b],&c=m.points[t.c];
        const R v=dot(sub(a,seed),cross(sub(b,seed),sub(c,seed)))/R(6);
        const Vec<R> centre=mul(add(add(seed,a),add(b,c)),R(.25));
        volume+=v;moment=add(moment,mul(sub(centre,m.cellCentres[cell]),v));
    }
}
}}}
#undef UGKP_WALL_RAY_HD
