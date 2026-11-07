#pragma once
#include "ResidentWallPolygons.cuh"
#include "WallRayMatching.H"
namespace ugkwp { namespace gaswall {
struct WallMatchingTopology {
    WallPolygonTopology polygons{};int cells=0;
    // Sorted by global face index within each cell, as in the native builder.
    const int *cellOffsets=nullptr,*cellFaces=nullptr,*owner=nullptr,*neighbour=nullptr;
    const unsigned char *physicalWall=nullptr,*constrained=nullptr,*gasCell=nullptr;
};
struct WallMatchingGeometry {
    int wallFace=-1,ownerCell=-1,donor=-1,failureFace=-1;
    std::array<double,3> origin{},normal{},tangent1{},tangent2{},matchingPoint{},firstMomentVector{};
    double ownerDistance=0,matchingDistance=0,maximumOwnerDistance=0,matchingDonorDistance=0;
    double volume=0,geometryVolume=0,firstMoment=0;
};
struct WallMatchingStatus {int firstInvalidRow=-1,invalidFace=-1;};
namespace residentWallMatchingDetail {
using residentWallPolygonDetail::Range;
struct Cells {
    const ugkp::gpu::geometry::CellResult* data;
    __device__ std::array<double,3> operator[](int c)const{const auto p=data[c].centre;return {{p.x,p.y,p.z}};}
};
struct Mask {
    const unsigned char* data;
    __device__ bool empty()const{return !data;}
    __device__ unsigned char operator[](int c)const{return data[c];}
};
struct Mesh {
    residentWallPolygonDetail::Points points;Cells cellCentres;
    const int *faceOffsets,*facePoints,*owner,*neighbour;const unsigned char* physicalWall;Mask gasCell;
};
struct Face {
    Range<const WallPolygonTriangle> triangles;std::array<double,3> normal,centre;double scale;
};
struct Faces {
    const WallPolygonMetric* metric;const WallPolygonTriangle* triangles;const int* offsets;
    __device__ Face operator[](int f)const{const auto m=metric[f];return {{triangles+offsets[f],std::size_t(m.triangles)},m.normal,m.centre,m.scale};}
};
struct CellFaces {
    const int *offsets,*faces;
    __device__ Range<const int> operator[](int c)const{return {faces+offsets[c],std::size_t(offsets[c+1]-offsets[c])};}
};
struct Context {Faces faces;CellFaces cellFaces;const unsigned char* constrained;};
}
__global__ void resetResidentWallMatchingStatusKernel(int rows,const int* polygonFailure,
    int faces,WallMatchingStatus* status){
    if(!blockIdx.x&&!threadIdx.x){status->firstInvalidRow=rows;
        status->invalidFace=*polygonFailure<faces?*polygonFailure:-1;}
}
// Produces the common ray frame, donor anchor and signed owner moments.
// FV donor reconstruction and optional positive quadrature are separate
// consumers. A cell-average wall closure uses this anchor with unit weight.
__global__ void rebuildResidentWallMatchingKernel(int count,const int* walls,WallMatchingTopology topology,
    const ugkp::gpu::geometry::Point* points,const ugkp::gpu::geometry::CellResult* cells,
    WallPolygonWorkspace polygons,bool reconstructMatching,WallMatchingGeometry* output,WallMatchingStatus* status)
{
    const int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count||status->invalidFace>=0)return;
    using namespace residentWallMatchingDetail;using namespace wallGeometryDetail;
    const Mesh mesh{{points},{cells},topology.polygons.faceOffsets,topology.polygons.facePoints,
        topology.owner,topology.neighbour,topology.physicalWall,{topology.gasCell}};
    const Context context{{polygons.metric,polygons.triangles,topology.polygons.triangleOffsets},
        {topology.cellOffsets,topology.cellFaces},topology.constrained};
    WallMatchingGeometry result;result.wallFace=walls[row];result.ownerCell=topology.owner[result.wallFace];
    bool valid=mesh.gasCell.empty()||mesh.gasCell[result.ownerCell];
    if(valid){
        initializeWallRayOrigin<double>(mesh,context,result.wallFace,result);
        initializeWallRayFrame<double>(mesh,context,result);
        result.donor=locateWallRayDonor<double>(mesh,context,result,reconstructMatching,result.failureFace);
        valid=result.donor>=0;
    }else result.failureFace=result.wallFace;
    if(valid){
        double volume=0;Vec<double> moment{};const auto seed=mesh.cellCentres[result.ownerCell];
        evaluateWallOwnerMoments<double>(mesh,context,result.ownerCell,seed,volume,moment);
        result.geometryVolume=volume;result.volume=cells[result.ownerCell].volume;
        valid=gasFinite(volume)&&volume!=0&&gasFinite(result.volume)&&finite(moment);
        if(valid){const auto centroid=add(seed,mul(moment,1/volume));
            result.firstMoment=result.volume*(result.ownerDistance+dot(moment,result.normal)/volume);
            result.firstMomentVector=mul(centroid,result.volume);
        }else result.failureFace=result.wallFace;
    }
    output[row]=result;if(!valid)atomicMin(&status->firstInvalidRow,row);
}
__global__ void finishResidentWallMatchingStatusKernel(int count,const WallMatchingGeometry* output,WallMatchingStatus* status){
    if(!blockIdx.x&&!threadIdx.x&&status->invalidFace<0&&status->firstInvalidRow<count)
        status->invalidFace=output[status->firstInvalidRow].failureFace;
}
}}
