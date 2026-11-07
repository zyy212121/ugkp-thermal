#pragma once
#include "../gasTransport/MixtureThermo.H"
#include <array>
#include <cmath>
#include <cstddef>
#ifdef __CUDACC__
#define UGKP_WALL_GEOMETRY_HD __host__ __device__
#else
#define UGKP_WALL_GEOMETRY_HD
#endif
namespace ugkwp { namespace gaswall { namespace wallGeometryDetail {
template<class R> using Vec=std::array<R,3>;
template<class R> UGKP_WALL_GEOMETRY_HD inline Vec<R> add(const Vec<R>&a,const Vec<R>&b){return {{a[0]+b[0],a[1]+b[1],a[2]+b[2]}};}
template<class R> UGKP_WALL_GEOMETRY_HD inline Vec<R> sub(const Vec<R>&a,const Vec<R>&b){return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};}
template<class R> UGKP_WALL_GEOMETRY_HD inline Vec<R> mul(const Vec<R>&a,R b){return {{a[0]*b,a[1]*b,a[2]*b}};}
template<class R> UGKP_WALL_GEOMETRY_HD inline R dot(const Vec<R>&a,const Vec<R>&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
template<class R> UGKP_WALL_GEOMETRY_HD inline Vec<R> cross(const Vec<R>&a,const Vec<R>&b){return {{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}};}
template<class R> UGKP_WALL_GEOMETRY_HD inline R norm(const Vec<R>&a){return std::hypot(std::hypot(a[0],a[1]),a[2]);}
template<class R> UGKP_WALL_GEOMETRY_HD inline bool finite(const Vec<R>&a){return gasFinite(a[0])&&gasFinite(a[1])&&gasFinite(a[2]);}
template<class R> UGKP_WALL_GEOMETRY_HD inline R turn(const Vec<R>& a,const Vec<R>& b,const Vec<R>&c,const Vec<R>&n){return dot(cross(sub(b,a),sub(c,a)),n);}
// Same ordered ear-clipping construction for host and resident owners.
// Ring and triangle storage are supplied by the caller. A face with N input
// vertices needs N mutable ring entries and at most N-2 output triangles.
// Neither geometry-dependent allocation nor a convex-face assumption is used.
template<class R,class Mesh,class Ring,class Output>
UGKP_WALL_GEOMETRY_HD bool triangulateWallPolygon(const Mesh& m,Ring& ring,Output& out){
    const Vec<R> anchor=m.points[ring[0]];Vec<R> area{};
    for(std::size_t i=0;i<ring.size();++i){
        const Vec<R> a=sub(m.points[ring[i]],anchor),b=sub(m.points[ring[(i+1)%ring.size()]],anchor);
        area=add(area,cross(a,b));const R length=norm(a);out.scale=out.scale<length?length:out.scale;
    }
    const R amag=norm(area);
    if(!(out.scale>0)||!(amag>0))return false;
    out.normal=mul(area,R(1)/amag);
    const R atol=0; // Geometric orientation uses the represented sign, without an area floor.
    // A repeated vertex, crossing edge or self-touching boundary is not a face.
    for(std::size_t i=0;i<ring.size();++i)for(std::size_t j=i+1;j<ring.size();++j){
        if(ring[i]==ring[j])return false;
        if(j==i+1||(i==0&&j+1==ring.size()))continue;
        const auto &a=m.points[ring[i]],&b=m.points[ring[(i+1)%ring.size()]],&c=m.points[ring[j]],&d=m.points[ring[(j+1)%ring.size()]];
        const R u=turn(a,b,c,out.normal),v=turn(a,b,d,out.normal),w=turn(c,d,a,out.normal),z=turn(c,d,b,out.normal);
        if(((u>atol&&v<-atol)||(u<-atol&&v>atol))&&((w>atol&&z<-atol)||(w<-atol&&z>atol)))return false;
    }
    // Collinear edge points carry no triangle area but remain in the mesh's

    bool changed=true;
    while(changed&&ring.size()>3){changed=false;for(std::size_t i=0;i<ring.size();++i){
        const auto&a=m.points[ring[(i+ring.size()-1)%ring.size()]],&b=m.points[ring[i]],&c=m.points[ring[(i+1)%ring.size()]];
        if(std::abs(turn(a,b,c,out.normal))<=atol&&dot(sub(b,a),sub(b,c))<=0){ring.erase(ring.begin()+i);changed=true;break;}
    }}
    while(ring.size()>3){
        bool cut=false;
        for(std::size_t i=0;i<ring.size();++i){
            const int ia=ring[(i+ring.size()-1)%ring.size()],ib=ring[i],ic=ring[(i+1)%ring.size()];
            const auto&a=m.points[ia],&b=m.points[ib],&c=m.points[ic];
            if(turn(a,b,c,out.normal)<=atol)continue;
            bool occupied=false;for(int ip:ring){if(ip==ia||ip==ib||ip==ic)continue;const auto&p=m.points[ip];
                if(turn(a,b,p,out.normal)>=-atol&&turn(b,c,p,out.normal)>=-atol&&turn(c,a,p,out.normal)>=-atol){occupied=true;break;}}
            if(occupied)continue;
            out.triangles.push_back({ia,ib,ic});ring.erase(ring.begin()+i);cut=true;break;
        }
        if(!cut)return false;
    }
    if(turn(m.points[ring[0]],m.points[ring[1]],m.points[ring[2]],out.normal)<=atol)return false;
    out.triangles.push_back({ring[0],ring[1],ring[2]});
    R sum=0;Vec<R> moment{};
    for(const auto&t:out.triangles){const auto a=sub(m.points[t.a],anchor),b=sub(m.points[t.b],anchor),c=sub(m.points[t.c],anchor);
        R w=norm(cross(sub(b,a),sub(c,a)));sum+=w;moment=add(moment,mul(add(add(a,b),c),w/R(3)));}
    if(!(sum>0))return false;
    out.centre=add(anchor,mul(moment,R(1)/sum));return finite(out.centre);
}
}}}
#undef UGKP_WALL_GEOMETRY_HD
