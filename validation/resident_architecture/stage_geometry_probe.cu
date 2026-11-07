#include "stage_geometry_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace stageProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {
    T* pointer=nullptr;explicit Device(const std::vector<T>& values){if(values.empty())return;
        check(cudaMalloc(&pointer,values.size()*sizeof(T)));check(cudaMemcpy(pointer,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}
};
template<class T> std::vector<T> download(const T* source,int count){std::vector<T> out(count);if(count)check(cudaMemcpy(out.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost));return out;}
Result evaluate(const Topology& topology,const std::vector<Point>& old,const std::vector<Point>& end,const std::vector<int>& walls,
    const std::vector<int>& partner,int axis,const std::vector<int>& mate,StageGeometryControls controls){
    ResidentStageGeometry geometry(topology,old.size(),walls,partner,axis,mate);
    Device<Point> points(old),endpoint(end);Device<StageGeometryControls> control(std::vector<StageGeometryControls>{controls});
    Device<ugkp::diagnostics::DeviceContext> diagnostic(std::vector<ugkp::diagnostics::DeviceContext>(1));
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));geometry.prepare(points.pointer,endpoint.pointer,control.pointer,diagnostic.pointer,stream);
    check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeHost)throw std::runtime_error("stage graph contains host communication or a host node");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));const auto view=geometry.view();
    for(int replay=0;replay<3;++replay){const auto& target=replay==1?old:end;
        if(!target.empty())check(cudaMemcpyAsync(endpoint.pointer,target.data(),target.size()*sizeof(Point),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(diagnostic.pointer,0,sizeof(ugkp::diagnostics::DeviceContext),stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        if(replay==1&&controls.interval>0&&std::isfinite(controls.interval)&&controls.absoluteTolerance>=0&&std::isfinite(controls.absoluteTolerance)
            &&controls.relativeTolerance>=0&&std::isfinite(controls.relativeTolerance)){
            StageGeometryStatus status;check(cudaMemcpy(&status,view.status,sizeof(status),cudaMemcpyDeviceToHost));
            if(status.failure!=StageGeometryFailure::None)throw std::runtime_error("stationary replay did not clear failed stage");}
    }
    Result result;check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.diagnostics,diagnostic.pointer,sizeof(result.diagnostics),cudaMemcpyDeviceToHost));
    if(result.status.failure==StageGeometryFailure::None){const auto end=view.endpointGeometry;
        result.faces=download(end.face,end.faces);result.cells=download(end.cell,end.cells);result.distances=download(end.distance,end.cells);
        result.sweep=download(view.oldGeometry.sweptFaces,end.faces);result.velocity=download(view.meshVelocity,end.faces);
        const auto oldCells=download(view.oldGeometry.cell,end.cells);
        const auto oldVolume=download(view.oldVolume,end.cells),newVolume=download(view.newVolume,end.cells);
        const auto sweptVolume=download(view.sweptVolume,end.faces);
        for(int c=0;c<end.cells;++c)if(oldCells[c].volume!=oldVolume[c]||result.cells[c].volume!=newVolume[c])throw std::runtime_error("packed stage volume differs");
        for(int f=0;f<end.faces;++f)if(result.sweep[f].volume!=sweptVolume[f])throw std::runtime_error("packed stage sweep differs");}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
