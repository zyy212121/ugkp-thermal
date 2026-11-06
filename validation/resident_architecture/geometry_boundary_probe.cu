#include "geometry_boundary_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace boundaryProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* pointer=nullptr;
    explicit Device(const std::vector<T>& values){if(values.empty())return;
        check(cudaMalloc(&pointer,values.size()*sizeof(T)));check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}
};
Result evaluate(const Topology& topology,const std::vector<Point>& old,const std::vector<Point>& endpoint,
    const std::vector<FaceResult>& faces,const std::vector<int>& partner,int axis,const std::vector<int>& mate){
    ResidentPolyhedralGeometry geometry(topology,{});
    ResidentGeometryBoundary boundary(topology,geometry.view().topology,old.size(),partner,axis,mate);
    Device<Point> points(old),end(endpoint);Device<FaceResult> face(faces);
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    boundary.periodic(end.pointer,face.pointer,stream);boundary.emptyTrajectory(points.pointer,end.pointer,stream);boundary.emptyUniqueness(end.pointer,stream);
    check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeHost)throw std::runtime_error("geometry boundary graph transfers or host work");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<3;++replay){const auto& target=replay==1?old:endpoint;
        if(!target.empty())check(cudaMemcpyAsync(end.pointer,target.data(),target.size()*sizeof(Point),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        if(replay==1){GeometryBoundaryStatus status;check(cudaMemcpy(&status,boundary.emptyTrajectoryStatus(),sizeof(status),cudaMemcpyDeviceToHost));
            if(status.failure!=GeometryBoundaryFailure::None)throw std::runtime_error("empty trajectory retained a previous failure");}
    }
    Result result;check(cudaMemcpy(&result.periodic,boundary.periodicStatus(),sizeof(GeometryBoundaryStatus),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.trajectory,boundary.emptyTrajectoryStatus(),sizeof(GeometryBoundaryStatus),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.unique,boundary.emptyUniquenessStatus(),sizeof(GeometryBoundaryStatus),cudaMemcpyDeviceToHost));
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
