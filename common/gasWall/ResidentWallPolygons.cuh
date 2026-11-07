#pragma once
#include "WallPolygonMath.H"
#include "../gpu/PolyhedralGeometryMath.H"
namespace ugkwp { namespace gaswall {
struct WallPolygonTriangle {int a,b,c;};
struct WallPolygonMetric {
    std::array<double,3> normal{},centre{};double scale=0;
    int triangles=0,valid=0;
};
struct WallPolygonTopology {
    int faces=0;
    const int *faceOffsets=nullptr,*facePoints=nullptr,*triangleOffsets=nullptr;
};
struct WallPolygonWorkspace {
    // Ring entries share the immutable face CSR offsets. Triangle offsets
    // reserve N-2 entries per face. All storage is allocated at initialization.
    int* ring=nullptr;WallPolygonTriangle* triangles=nullptr;
    WallPolygonMetric* metric=nullptr;int* firstInvalidFace=nullptr;
};
namespace residentWallPolygonDetail {
template<class T> struct Range {
    T* data;std::size_t count;
    __device__ std::size_t size()const{return count;}
    __device__ T* begin()const{return data;}
    __device__ T* end()const{return data+count;}
    __device__ T& operator[](std::size_t i)const{return data[i];}
    __device__ void erase(T* item){const std::size_t i=item-data;for(std::size_t j=i;j+1<count;++j)data[j]=data[j+1];--count;}
    __device__ void push_back(T value){data[count++]=value;}
};
struct Points {
    const ugkp::gpu::geometry::Point* data;
    __device__ std::array<double,3> operator[](int i)const{const auto p=data[i];return {{p.x,p.y,p.z}};}
};
struct Mesh {Points points;};
struct Face {Range<WallPolygonTriangle> triangles;std::array<double,3> normal{},centre{};double scale=0;};
}
__global__ void resetResidentWallPolygonStatusKernel(int faces,int* firstInvalidFace){
    if(!blockIdx.x&&!threadIdx.x)*firstInvalidFace=faces;
}
// Every face retains the native vertex and ear-selection order. Distinct faces
// run independently. No dynamic allocation, host callback or transfer occurs.
__global__ void rebuildResidentWallPolygonsKernel(WallPolygonTopology topology,
    const ugkp::gpu::geometry::Point* points,WallPolygonWorkspace workspace)
{
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=topology.faces)return;
    const int begin=topology.faceOffsets[f],end=topology.faceOffsets[f+1];
    for(int k=begin;k<end;++k)workspace.ring[k]=topology.facePoints[k];
    using namespace residentWallPolygonDetail;
    Range<int> ring{workspace.ring+begin,static_cast<std::size_t>(end-begin)};
    Face face{{workspace.triangles+topology.triangleOffsets[f],0}};
    const bool valid=wallGeometryDetail::triangulateWallPolygon<double>(Mesh{{points}},ring,face);
    workspace.metric[f]={face.normal,face.centre,face.scale,int(face.triangles.size()),valid};
    if(!valid)atomicMin(workspace.firstInvalidFace,f);
}
}}
