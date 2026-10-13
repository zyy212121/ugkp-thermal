#ifndef CHMT_MESH_EMPTY_EXTRUSION_H
#define CHMT_MESH_EMPTY_EXTRUSION_H
#include "core/HostState.H"
#include <algorithm>
#include <map>
#include <set>

namespace chmt { namespace empty_detail {
inline Real component(Vec3 point,int axis) { return axis==0?point.x:axis==1?point.y:point.z; }
inline bool same(Vec3 a,Vec3 b) { return a.x==b.x && a.y==b.y && a.z==b.z; }
struct Extrusion {
    int axis=-1;
    std::vector<int> mate;
};
inline bool reject(std::string& error,const char* message) {
    error=std::string("unsupported/invalid Empty extrusion: ")+message;return false;
}
// The first supported representation has two fixed axis-aligned layers. Pairing
// is exact and reconstructed from authoritative points/topology, never nearest
// neighbour tolerance or an independently persisted cache.
// Terminal supplied NaNs form one ordered equivalence class after numbers.
// Finite keys, infinities and signed-zero equivalence retain their old ordering.
template<bool Terminal> struct CoordinateLess {
    bool operator()(Real a,Real b) const {
        if(Terminal){if(a!=a)return false;if(b!=b)return true;}
        return a<b;
    }
};
template<bool Terminal> struct TangentialLess {
    bool operator()(const std::pair<Real,Real>& a,const std::pair<Real,Real>& b) const {
        const CoordinateLess<Terminal> less;
        return less(a.first,b.first)||(!less(b.first,a.first)&&less(a.second,b.second));
    }
};
template<bool Terminal=false>
inline bool build(const HostMesh& mesh,Extrusion& output,std::string& error) {
    if(!Terminal && (mesh.boundaryKind.size()!=mesh.owner.size() || mesh.areaVectors.size()!=mesh.owner.size()
        || mesh.faceOffsets.size()!=mesh.owner.size()+1 || mesh.faceOffsets.empty()
        || mesh.faceOffsets.front()!=0 || mesh.faceOffsets.back()<0
        || static_cast<std::size_t>(mesh.faceOffsets.back())!=mesh.facePoints.size()))
        return reject(error,"incomplete authoritative geometry");
    Extrusion result;std::set<Real,CoordinateLess<Terminal>> levels;std::vector<unsigned char> seen(mesh.points.size());
    for(std::size_t f=0;f<mesh.owner.size();++f) {
        const int begin=mesh.faceOffsets[f],end=mesh.faceOffsets[f+1];
        if(!Terminal && (begin<0 || end<begin || static_cast<std::size_t>(end)>mesh.facePoints.size() || end-begin<3))
            return reject(error,"invalid face CSR");
        for(int k=begin;k<end;++k)if(!Terminal && (mesh.facePoints[k]<0 || static_cast<std::size_t>(mesh.facePoints[k])>=mesh.points.size()))
            return reject(error,"invalid face vertex index");
        if(mesh.boundaryKind[f]!=BoundaryKind::Empty)continue;
        int axis=-1;
        for(int d=0;d<3;++d)if(component(mesh.areaVectors[f],d)!=0) {
            if(!Terminal && (axis!=-1))return reject(error,"rotated/non-axis-aligned Empty plane");
            axis=d;
        }
        if(!Terminal && (axis<0 || (result.axis>=0 && axis!=result.axis)))return reject(error,"inconsistent Empty plane axes");
        result.axis=axis;
        const Real level=component(mesh.points[mesh.facePoints[mesh.faceOffsets[f]]],axis);
        levels.insert(level);
        if(!Terminal && (levels.size()>2))return reject(error,"more than two normal layers");
        for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k) {
            const int point=mesh.facePoints[k];seen[point]=1;
            if(!Terminal && (component(mesh.points[point],axis)!=level))return reject(error,"Empty vertices are not in one exact coordinate plane");
        }
    }
    if(result.axis<0){output=std::move(result);return true;}
    if(!Terminal && (levels.size()!=2))return reject(error,"two distinct normal layers are required");
    const Real low=*levels.begin(),high=*levels.rbegin();
    const int u=(result.axis+1)%3,v=(result.axis+2)%3;
    std::map<std::pair<Real,Real>,std::array<int,2>,TangentialLess<Terminal>> pairs;
    for(std::size_t p=0;p<mesh.points.size();++p) {
        if(!Terminal && (!seen[p]))return reject(error,"every point must belong to an Empty layer");
        const Real normal=component(mesh.points[p],result.axis);
        if(!Terminal && (!finite(mesh.points[p])))return reject(error,"nonfinite layer point");
        if(!Terminal && (normal!=low && normal!=high))return reject(error,"point lies outside the two fixed layers");
        const auto key=std::make_pair(component(mesh.points[p],u),component(mesh.points[p],v));
        auto inserted=pairs.emplace(key,std::array<int,2>{{-1,-1}});
        int& slot=inserted.first->second[normal==low?0:1];
        if(!Terminal && (slot>=0))return reject(error,"duplicate tangential point in one layer");
        slot=static_cast<int>(p);
    }
    result.mate.assign(mesh.points.size(),-1);
    for(const auto& pair:pairs) {
        const int a=pair.second[0],b=pair.second[1];
        if(a<0 || b<0)return reject(error,"unpaired layer vertices");
        result.mate[a]=b;result.mate[b]=a;
    }
    if(!Terminal) {
    std::vector<std::array<int,2>> cellFaces(mesh.volumes.size(),{{-1,-1}});
    for(std::size_t f=0;f<mesh.owner.size();++f) {
        if(mesh.boundaryKind[f]==BoundaryKind::Empty) {
            if(!Terminal && (mesh.owner[f]<0 || static_cast<std::size_t>(mesh.owner[f])>=cellFaces.size()))
                return reject(error,"missing Empty owner cell geometry");
            const Real normal=component(mesh.points[mesh.facePoints[mesh.faceOffsets[f]]],result.axis);
            int& slot=cellFaces[mesh.owner[f]][normal==low?0:1];
            if(!Terminal && (slot>=0))return reject(error,"cell has multiple Empty faces on the same layer");
            slot=static_cast<int>(f);
        } else {
            std::set<int> projected;
            for(int k=mesh.faceOffsets[f];k<mesh.faceOffsets[f+1];++k) {
                const int p=mesh.facePoints[k];projected.insert(std::min(p,result.mate[p]));
            }
            if(!Terminal && (projected.size()!=2))return reject(error,"non-Empty face is not an extruded cross-section edge");
        }
    }
    for(const auto& faces:cellFaces) {
        if(!Terminal && (faces[0]<0 || faces[1]<0))return reject(error,"cell does not span both Empty layers");
        std::set<int> mapped,other;
        for(int k=mesh.faceOffsets[faces[0]];k<mesh.faceOffsets[faces[0]+1];++k)mapped.insert(result.mate[mesh.facePoints[k]]);
        for(int k=mesh.faceOffsets[faces[1]];k<mesh.faceOffsets[faces[1]+1];++k)other.insert(mesh.facePoints[k]);
        if(!Terminal && (mapped!=other))return reject(error,"cell Empty faces are not a paired extrusion");
    }
    }
    output=std::move(result);return true;
}
// Terminal geometry consumers require only the actual Empty axis, not a mate map.
inline bool axisGeometry(const HostMesh& mesh,Extrusion& output,std::string&,bool*) {
    Extrusion result;
    for(std::size_t f=0;f<mesh.owner.size();++f)if(mesh.boundaryKind[f]==BoundaryKind::Empty) {
        int axis=-1;
        for(int d=0;d<3;++d)if(component(mesh.areaVectors[f],d)!=0)axis=d;
        if(axis>=0)result.axis=axis;
    }
    output=std::move(result);return true;
}
inline bool trajectory(const HostMesh& old,const Extrusion& extrusion,const std::vector<Vec3>& end,std::string& error) {
    if(extrusion.axis<0)return true;
    if(end.size()!=old.points.size())return reject(error,"trajectory point count mismatch");
    for(std::size_t p=0;p<end.size();++p) {
        if(component(end[p],extrusion.axis)!=component(old.points[p],extrusion.axis))
            return reject(error,"nonzero normal trajectory displacement");
        const int paired=extrusion.mate[p];
        for(int d=0;d<3;++d)if(d!=extrusion.axis && component(end[p],d)!=component(end[paired],d))
            return reject(error,"layer-paired tangential trajectory differs");
    }
    return true;
}
inline bool prescribedMotion(const Extrusion& extrusion,const std::vector<Vec3>& displacement,
    const std::vector<unsigned char>& fixed,std::string& error) {
    if(extrusion.axis<0)return true;
    for(std::size_t p=0;p<displacement.size();++p) {
        if(!fixed[p])return reject(error,"Empty layer vertices require prescribed paired motion");
        if(component(displacement[p],extrusion.axis)!=0)return reject(error,"nonzero normal input displacement");
        const int q=extrusion.mate[p];
        if(!same(displacement[p],displacement[q]))return reject(error,"paired prescribed displacements differ");
    }
    return true;
}
inline bool motionData(const Extrusion& extrusion,const std::vector<Vec3>& displacement,
    const std::vector<unsigned char>& fixed,const std::vector<Vec3>& directions,std::string& error) {
    if(extrusion.axis<0)return true;
    if(!prescribedMotion(extrusion,displacement,fixed,error))return false;
    for(std::size_t p=0;p<displacement.size();++p) {
        if(component(directions[p],extrusion.axis)!=0)return reject(error,"nonzero normal input direction");
        const int q=extrusion.mate[p];
        const Real a=mag(directions[p]),b=mag(directions[q]);
        if(!finite(a) || !finite(b) || ((a>0)!=(b>0)))return reject(error,"paired correction spaces differ");
        if(a>0) {
            const Vec3 u=directions[p]/a,v=directions[q]/b;
            if(!same(u,v) && !same(u,v*(-1)))return reject(error,"paired correction directions are not identical/opposite");
        } else if(!same(directions[p],Vec3{}) || !same(directions[q],Vec3{}))
            return reject(error,"unrepresentably small correction direction");
    }
    return true;
}
inline Real pairedScale(Real a,Real b) {
    const Real low=minValue(a,b),high=maxValue(a,b),ratio=low/high;
    return low/::sqrt(1+ratio*ratio);
}
} }
#endif
