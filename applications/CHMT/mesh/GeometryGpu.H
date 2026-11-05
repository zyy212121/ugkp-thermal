#ifndef CHMT_MESH_GEOMETRY_GPU_H
#define CHMT_MESH_GEOMETRY_GPU_H
#include "core/HostState.H"
#include "mesh/GeometryDiagnostics.H"
#include "gpu/PolyhedralGeometry.H"
#include <algorithm>
#include <map>
#include <set>
namespace chmt { namespace geometry_device {
namespace backend=ugkp::gpu::geometry;
template<class Points> std::vector<backend::Point> pack(const Points& input){
    std::vector<backend::Point> out;out.reserve(input.size());for(const auto& p:input)out.push_back({p.x,p.y,p.z});return out;
}
inline Vec3 unpack(backend::Point p){return {p.x,p.y,p.z};}
template<class Points> void unpack(const std::vector<backend::Point>& input,Points& output){
    output.resize(input.size());for(std::size_t p=0;p<input.size();++p)output[p]=unpack(input[p]);
}
inline backend::Topology faceTopology(const HostMesh& mesh){
    backend::Topology t;t.faceOffsets=mesh.faceOffsets;t.facePoints=mesh.facePoints;return t;
}
inline backend::PolyhedralGeometry& workspace(){thread_local backend::PolyhedralGeometry value;return value;}
inline bool failure(std::string& error,const char* text,int index){error=std::string(text)+" at index "+std::to_string(index);return false;}
// One device transaction computes face, cell and distance fields. Host
// publication still follows the original validation/diagnostic ordering.
class RebuiltGeometry {
    backend::Topology topology_;std::vector<unsigned char> manifold_;backend::RebuildResult result_;
public:
    explicit RebuiltGeometry(const HostMesh& mesh){
    topology_=faceTopology(mesh);auto& t=topology_;t.cellOffsets=mesh.cellFaceOffsets;t.cellFaces=mesh.cellFaces;t.cellSigns=mesh.cellFaceSigns;t.vertexOffsets.push_back(0);
    const int count=mesh.cellFaceOffsets.size()-1;manifold_.assign(count,1);auto& manifold=manifold_;
    for(int c=0;c<count;++c){std::set<int> vertices;std::map<std::pair<int,int>,std::pair<int,int>> edges;
        for(int k=mesh.cellFaceOffsets[c];k<mesh.cellFaceOffsets[c+1];++k){
            const int f=mesh.cellFaces[k],sign=mesh.cellFaceSigns[k],begin=mesh.faceOffsets[f],end=mesh.faceOffsets[f+1];
            for(int j=begin;j<end;++j){const int a=mesh.facePoints[j],b=mesh.facePoints[j+1==end?begin:j+1];vertices.insert(a);
                auto& edge=edges[std::minmax(a,b)];++edge.first;edge.second+=(a<b?1:-1)*sign;}}
        for(const auto& edge:edges)if(edge.second.first!=2||edge.second.second!=0)manifold[c]=0;
        t.vertices.insert(t.vertices.end(),vertices.begin(),vertices.end());t.vertexOffsets.push_back(t.vertices.size());}
        std::vector<int> walls;for(std::size_t f=0;f<mesh.boundaryKind.size();++f){const auto k=mesh.boundaryKind[f];
            if(k==BoundaryKind::NoSlip||k==BoundaryKind::Slip||k==BoundaryKind::Interface)walls.push_back(f);}
        result_=workspace().rebuild(t,pack(mesh.points),walls);
    }
    bool faces(HostMesh& mesh,std::string& error)const{const auto& result=result_.faces;
    mesh.areaVectors.assign(result.size(),{});mesh.faceCentres.assign(result.size(),{});
    for(std::size_t f=0;f<result.size();++f){mesh.areaVectors[f]=unpack(result[f].area);
        if(result[f].invalid)return failure(error,"degenerate face area",f);mesh.faceCentres[f]=unpack(result[f].centre);}
    return true;
    }
    bool cells(HostMesh& mesh,std::string& error,ugkp::diagnostics::DeviceContext* diagnostics)const{
        const auto& result=result_.cells;const auto& t=topology_;const auto& manifold=manifold_;const int count=result.size();
    mesh.volumes.assign(count,0);mesh.cellCentres.assign(count,{});
    // Publish diagnostics and failures in the same cell/policy order as before.
    for(int c=0;c<count;++c){const auto& r=result[c];const Vec3 closure=unpack(r.closure);
        if(t.vertexOffsets[c+1]-t.vertexOffsets[c]<4)return failure(error,"degenerate cell geometry",c);
        if(!finite(closure)||!finite(r.areaSum))return failure(error,"nonfinite cell area closure",c);
        if(mag(closure)>1e-10*r.areaSum)geometryWarning(diagnostics,mag(closure),1e-10*r.areaSum,c,true,ugkp::diagnostics::EntityKind::Cell);
        if(!manifold[c])return failure(error,"nonclosed/nonmanifold cell edges",c);
        if(r.invalid==1)return failure(error,"inverted/nonpositive cell volume",c);
        mesh.volumes[c]=r.volume;mesh.cellCentres[c]=unpack(r.centre);
        if(r.invalid==2)return failure(error,"nonfinite cell centroid",c);}
    return true;
    }
    bool wallDistance(HostMesh& mesh,const std::vector<int>& walls,std::string& error)const{
        if(walls.empty())return true;const auto& result=result_.distance;
    for(std::size_t c=0;c<result.size();++c){if(result[c].invalidFace>=0)return failure(error,"nonfinite wall triangle distance",result[c].invalidFace);
        if(!(result[c].distance>0)||!finite(result[c].distance))return failure(error,"invalid wall distance",c);mesh.wallDistance[c]=result[c].distance;}
    return true;
    }
};
inline void adjacency(const std::vector<std::set<int>>& rows,std::vector<int>& offsets,std::vector<int>& neighbours){
    offsets.assign(1,0);for(const auto& row:rows){neighbours.insert(neighbours.end(),row.begin(),row.end());offsets.push_back(neighbours.size());}
}
}}
#endif
