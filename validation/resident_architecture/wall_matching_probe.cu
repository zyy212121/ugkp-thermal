#include "wall_matching_probe.H"
#include "common/gasWall/ResidentWallMatching.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
namespace wallMatchingProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* pointer=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;check(cudaMalloc(&pointer,v.size()*sizeof(T)));check(cudaMemcpy(pointer,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}
};
template<class T> std::vector<T> download(const T* p,int n){std::vector<T> out(n);if(n)check(cudaMemcpy(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return out;}
std::vector<Result> evaluate(const Mesh& m,bool reconstructed){
    using namespace ugkwp::gaswall;using namespace ugkp::gpu::geometry;
    const int nf=m.owner.size(),nc=m.centres.size(),nr=m.walls.size();
    std::vector<int> triangleOffsets(1,0),cellOffsets(1,0),cellFaces;
    std::vector<unsigned char> constrained=m.constrained;if(constrained.empty())constrained.resize(nc,0);
    for(int f=0;f<nf;++f){triangleOffsets.push_back(triangleOffsets.back()+m.offsets[f+1]-m.offsets[f]-2);if(m.physicalWall[f])constrained[m.owner[f]]=1;}
    for(int c=0;c<nc;++c){for(int f=0;f<nf;++f)if(m.owner[f]==c||m.neighbour[f]==c)cellFaces.push_back(f);cellOffsets.push_back(cellFaces.size());}
    std::vector<Point> points;for(auto p:m.points)points.push_back({p[0],p[1],p[2]});
    std::vector<CellResult> cells(nc);for(int c=0;c<nc;++c){const auto p=m.centres[c];cells[c].centre={p[0],p[1],p[2]};cells[c].volume=m.volumes[c];}
    Device<Point> dPoints(points);Device<CellResult> dCells(cells);
    Device<int> offsets(m.offsets),indices(m.indices),triOffsets(triangleOffsets),co(cellOffsets),cf(cellFaces),owner(m.owner),neighbour(m.neighbour),walls(m.walls),ring(std::vector<int>(m.indices.size())),first(std::vector<int>(1));
    Device<unsigned char> physical(m.physicalWall),mask(constrained),gas(m.gasCell);
    Device<WallPolygonTriangle> triangles{std::vector<WallPolygonTriangle>(triangleOffsets.back())};Device<WallPolygonMetric> metrics{std::vector<WallPolygonMetric>(nf)};
    Device<WallMatchingGeometry> output{std::vector<WallMatchingGeometry>(nr)};Device<WallMatchingStatus> status{std::vector<WallMatchingStatus>(1)};
    WallPolygonTopology ptop{nf,offsets.pointer,indices.pointer,triOffsets.pointer};WallPolygonWorkspace workspace{ring.pointer,triangles.pointer,metrics.pointer,first.pointer};
    WallMatchingTopology topology{ptop,nc,co.pointer,cf.pointer,owner.pointer,neighbour.pointer,physical.pointer,mask.pointer,gas.pointer};
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    resetResidentWallPolygonStatusKernel<<<1,1,0,stream>>>(nf,first.pointer);
    if(nf)rebuildResidentWallPolygonsKernel<<<(nf+127)/128,128,0,stream>>>(ptop,dPoints.pointer,workspace);
    resetResidentWallMatchingStatusKernel<<<1,1,0,stream>>>(nr,first.pointer,nf,status.pointer);
    if(nr)rebuildResidentWallMatchingKernel<<<(nr+127)/128,128,0,stream>>>(nr,walls.pointer,topology,dPoints.pointer,dCells.pointer,workspace,reconstructed,output.pointer,status.pointer);
    finishResidentWallMatchingStatusKernel<<<1,1,0,stream>>>(nr,output.pointer,status.pointer);
    check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeHost||type==cudaGraphNodeTypeMemcpy)throw std::runtime_error("wall matching graph contains host communication");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));std::vector<Result> result(3);
    for(int replay=0;replay<3;++replay){auto active=constrained;if(replay==1)std::fill(active.begin(),active.end(),1);
        if(nc)check(cudaMemcpyAsync(mask.pointer,active.data(),nc,cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));const auto state=download(status.pointer,1)[0];
        auto& r=result[replay];r.firstInvalidRow=state.firstInvalidRow;r.invalidFace=state.invalidFace;
        for(const auto& d:download(output.pointer,nr))r.row.push_back({d.wallFace,d.ownerCell,d.donor,d.failureFace,d.origin,d.normal,d.tangent1,d.tangent2,d.matchingPoint,d.firstMomentVector,
            d.ownerDistance,d.matchingDistance,d.maximumOwnerDistance,d.matchingDonorDistance,d.volume,d.geometryVolume,d.firstMoment});
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
