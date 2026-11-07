#include "surface_geometry_probe.H"
#include <cuda_runtime.h>
#include <stdexcept>
namespace surfaceProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> struct Device {T* pointer=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;
    check(cudaMalloc(&pointer,v.size()*sizeof(T)));check(cudaMemcpy(pointer,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(pointer)cudaFree(pointer);}};
Result evaluate(const Mesh& solid,const Mesh& gas,const std::vector<SurfaceFaceMap>& faces,const std::vector<SurfaceEdgeMap>& edges,double interval){
    ResidentStageGeometry solidStage(solid.topology,solid.old.size(),solid.walls,solid.partner,solid.axis,solid.mate);
    ResidentStageGeometry gasStage(gas.topology,gas.old.size(),gas.walls,gas.partner,gas.axis,gas.mate);
    ResidentSurfaceGeometry surface(faces,edges.size(),solid.old.size(),solid.axis>=0);
    Device<Point> oldSolid(solid.old),endSolid(solid.end),oldGas(gas.old),endGas(gas.end);Device<SurfaceEdgeMap> edge(edges);
    // Keep region geometry valid when testing the surface operator's separate
    // invalid-interval contract; this is fixture setup, not a solver recovery.
    Device<StageGeometryControls> controls(std::vector<StageGeometryControls>{{1e-3,1e-12,1e-10}});
    Device<double> dt(std::vector<double>{interval});Device<ugkp::diagnostics::DeviceContext> diagnostic(std::vector<ugkp::diagnostics::DeviceContext>(1));
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    solidStage.prepare(oldSolid.pointer,endSolid.pointer,controls.pointer,nullptr,stream);gasStage.prepare(oldGas.pointer,endGas.pointer,controls.pointer,nullptr,stream);
    surface.prepare(solidStage.view().oldGeometry,solidStage.view().endpointGeometry,gasStage.view().endpointGeometry,
        oldSolid.pointer,endSolid.pointer,edge.pointer,dt.pointer,diagnostic.pointer,stream);
    check(cudaStreamEndCapture(stream,&graph));std::size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));if(type==cudaGraphNodeTypeMemcpy||type==cudaGraphNodeTypeHost)throw std::runtime_error("surface graph contains host exchange");}
    check(cudaGraphInstantiateWithFlags(&executable,graph,0));
    for(int replay=0;replay<3;++replay){const double value=replay==1?0:interval;
        check(cudaMemcpyAsync(dt.pointer,&value,sizeof(value),cudaMemcpyHostToDevice,stream));check(cudaMemsetAsync(diagnostic.pointer,0,sizeof(ugkp::diagnostics::DeviceContext),stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        if(replay==1){SurfaceGeometryStatus status;check(cudaMemcpy(&status,surface.view().status,sizeof(status),cudaMemcpyDeviceToHost));
            if(status.failure!=SurfaceGeometryFailure::InvalidInterval)throw std::runtime_error("surface interval is not read from device");}}
    for(const auto* status:{solidStage.view().status,gasStage.view().status}){StageGeometryStatus state;check(cudaMemcpy(&state,status,sizeof(state),cudaMemcpyDeviceToHost));
        if(state.failure!=StageGeometryFailure::None)throw std::runtime_error("surface fixture did not supply certified region geometry");}
    Result result;const auto view=surface.view();check(cudaMemcpy(&result.status,view.status,sizeof(result.status),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&result.diagnostics,diagnostic.pointer,sizeof(result.diagnostics),cudaMemcpyDeviceToHost));
    if(result.status.failure==SurfaceGeometryFailure::None){result.faces.resize(view.faces);result.edges.resize(view.edges);
        if(view.faces)check(cudaMemcpy(result.faces.data(),view.face,view.faces*sizeof(SurfaceFaceGeometry),cudaMemcpyDeviceToHost));
        if(view.edges)check(cudaMemcpy(result.edges.data(),view.edge,view.edges*sizeof(EdgeResult),cudaMemcpyDeviceToHost));}
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
