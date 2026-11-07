#include "surface_seam_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace seamProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {T* p=nullptr;explicit Device(std::size_t n){if(n)check(cudaMalloc(&p,n*sizeof(T)));}~Device(){if(p)cudaFree(p);}};
std::vector<Result> evaluate(const Topology& topology,const std::vector<SurfaceFaceMap>& maps,const std::vector<int>& surfaces,const std::vector<SurfaceTopologyEdge>& edges,
    const std::vector<SurfaceSeamPatch>& patches,const std::vector<Sample>& samples){
    if(samples.empty())return {};
    const auto& first=samples.front();ResidentSurfaceSeams owner(first.points.size(),first.faces.size(),surfaces,edges,patches);
    Device<Point> points(first.points.size());Device<FaceResult> faces(first.faces.size());Device<double> interval(1);
    const double dt=1;check(cudaMemcpy(interval.p,&dt,sizeof(dt),cudaMemcpyHostToDevice));
    ResidentPolyhedralGeometry poly(topology,{});ResidentSurfaceGeometry surface(maps,edges.size(),first.points.size(),false);
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    poly.rebuild(points.p,stream);owner.prepare(points.p,faces.p,stream);
    surface.prepareWithSeams(poly.view(),poly.view(),poly.view(),points.p,points.p,owner.view(),interval.p,nullptr,stream);
    check(cudaStreamEndCapture(stream,&graph));
    std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeHost)throw std::runtime_error("surface seam graph contains host exchange");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));std::vector<Result> results;
    for(const auto& input:samples){
        if(input.points.size()!=first.points.size()||input.faces.size()!=first.faces.size())throw std::runtime_error("fixture shape changed");
        if(points.p)check(cudaMemcpyAsync(points.p,input.points.data(),input.points.size()*sizeof(Point),cudaMemcpyHostToDevice,stream));
        if(faces.p)check(cudaMemcpyAsync(faces.p,input.faces.data(),input.faces.size()*sizeof(FaceResult),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));Result r;
        const auto view=owner.view();check(cudaMemcpy(&r.status,view.status,sizeof(r.status),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&r.surfaceStatus,surface.view().status,sizeof(r.surfaceStatus),cudaMemcpyDeviceToHost));
        if(r.status.failure==SurfaceSeamFailure::None){r.edges.resize(r.status.count);
            r.faceMetrics.resize(maps.size());r.edgeMetrics.resize(r.surfaceStatus.edgeCount);
            if(!maps.empty())check(cudaMemcpy(r.faceMetrics.data(),surface.view().face,maps.size()*sizeof(SurfaceFaceGeometry),cudaMemcpyDeviceToHost));
            if(r.surfaceStatus.edgeCount)check(cudaMemcpy(r.edgeMetrics.data(),surface.view().edge,r.surfaceStatus.edgeCount*sizeof(EdgeResult),cudaMemcpyDeviceToHost));
            if(r.status.count)check(cudaMemcpy(r.edges.data(),view.edges,r.status.count*sizeof(SurfaceEdgeMap),cudaMemcpyDeviceToHost));}
        results.push_back(std::move(r));
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return results;
}
}
