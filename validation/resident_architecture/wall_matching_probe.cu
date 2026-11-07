#include "wall_matching_probe.H"
#include "common/gpu/ResidentWallMatching.H"
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
    std::vector<unsigned char> constrained=m.constrained;if(constrained.empty())constrained.resize(nc,0);
    for(int f=0;f<nf;++f)if(m.physicalWall[f])constrained[m.owner[f]]=1;
    std::vector<Point> points;for(auto p:m.points)points.push_back({p[0],p[1],p[2]});
    std::vector<CellResult> cells(nc);for(int c=0;c<nc;++c){const auto p=m.centres[c];cells[c].centre={p[0],p[1],p[2]};cells[c].volume=m.volumes[c];}
    Device<Point> dPoints(points);Device<CellResult> dCells(cells);
    Device<int> offsets(m.offsets),indices(m.indices);Device<unsigned char> mask(constrained),gasMask(m.gasCell.empty()?std::vector<unsigned char>(nc,1):m.gasCell);
    WallMatchingDefinition definition{nc,m.offsets,m.owner,m.neighbour,m.walls,m.physicalWall,m.constrained,m.gasCell};
    TopologyView topology{};topology.faceOffsets=offsets.pointer;topology.facePoints=indices.pointer;
    ResidentWallMatching owner(definition,topology,reconstructed);const auto view=owner.view();
    cudaStream_t stream;cudaGraph_t graphs[2]{};cudaGraphExec_t executables[2]{};check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int supplied=0;supplied<2;++supplied){
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        owner.prepare(dPoints.pointer,dCells.pointer,supplied?mask.pointer:nullptr,supplied?gasMask.pointer:nullptr,stream);
        check(cudaStreamEndCapture(stream,&graphs[supplied]));
        std::size_t count=0;check(cudaGraphGetNodes(graphs[supplied],nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graphs[supplied],nodes.data(),&count));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeHost||type==cudaGraphNodeTypeMemcpy)throw std::runtime_error("wall matching graph contains host communication");}
        check(cudaGraphInstantiateWithFlags(&executables[supplied],graphs[supplied],0));
    }
    std::vector<Result> result(3);
    for(int replay=0;replay<3;++replay){auto active=constrained;if(replay==1)std::fill(active.begin(),active.end(),1);
        if(nc)check(cudaMemcpyAsync(mask.pointer,active.data(),nc,cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executables[replay==1],stream));check(cudaStreamSynchronize(stream));const auto state=download(view.status,1)[0];
        auto& r=result[replay];r.firstInvalidRow=state.firstInvalidRow;r.invalidFace=state.invalidFace;
        for(const auto& d:download(view.geometry,nr))r.row.push_back({d.wallFace,d.ownerCell,d.donor,d.failureFace,d.origin,d.normal,d.tangent1,d.tangent2,d.matchingPoint,d.firstMomentVector,
            d.ownerDistance,d.matchingDistance,d.maximumOwnerDistance,d.matchingDonorDistance,d.volume,d.geometryVolume,d.firstMoment});
    }
    for(int i=0;i<2;++i){check(cudaGraphExecDestroy(executables[i]));check(cudaGraphDestroy(graphs[i]));}check(cudaStreamDestroy(stream));return result;
}
}
