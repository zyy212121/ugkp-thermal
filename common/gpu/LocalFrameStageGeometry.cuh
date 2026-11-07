// Included after ResidentStageGeometry.cuh.
#include "LocalFrameStageGeometry.H"
#include "ExecutionGraph.cuh"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void localFrameStageCoordinates(int count,const LocalFrameStageInput* input,Point* old,Point* endpoint){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    const Point origin=input->worldOld[p],displacement=input->localEndpoint[p]-input->localOld[p];
    old[p]=origin;
    // Preserve all three additions, including signed-zero rounding, when any
    // component moves. An unchanged point retains its exact original bits.
    endpoint[p]=(displacement.x!=0||displacement.y!=0||displacement.z!=0)?origin+displacement:origin;
}
__global__ void initializeLocalFrameStage(const LocalFrameStageInput* input,StageGeometryControls* controls,ugkp::diagnostics::DeviceContext* diagnostics){
    *controls=input->controls;*diagnostics={};
}
__global__ void copyLocalFrameStageInput(const LocalFrameStageInput* input,LocalFrameStageInput* output){*output=*input;}
__global__ void bindLocalFrameStage(const Point* world,const Point* local,const Point* endpoint,StageGeometryControls controls,LocalFrameStageInput* output){
    *output={world,local,endpoint,controls};
}
}
struct LocalFrameStageGeometry::Implementation {
    int points=0;std::unique_ptr<ResidentStageGeometry> stage;
    Array<Point> referenceWorld,referenceLocal,old,endpoint;
    Array<LocalFrameStageInput> input;Array<StageGeometryControls> controls;Array<ugkp::diagnostics::DeviceContext> diagnostics;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;PinnedTransferQueue transfer;
    explicit Implementation(const LocalFrameStageDefinition& d){
        if(d.worldOld.size()!=d.localOld.size()||d.worldOld.size()>std::size_t(INT_MAX))throw std::invalid_argument("local-frame stage point shape mismatch");
        points=d.worldOld.size();stage.reset(new ResidentStageGeometry(d.topology,points,d.walls,d.periodicPartner,d.emptyAxis,d.emptyMate));
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);PinnedTransferQueue::Operation pending(transfer);
        referenceWorld.upload(d.worldOld,transfer);referenceLocal.upload(d.localOld,transfer);
        old.reserve(points);endpoint.reserve(points);input.reserve(1);controls.reserve(1);diagnostics.reserve(1);transfer.finish();
    }
    ~Implementation(){transfer.discard();if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
LocalFrameStageGeometry::LocalFrameStageGeometry(const LocalFrameStageDefinition& definition):data_(new Implementation(definition)){
    auto& d=*data_;checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph,nullptr,d.input.pointer);checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
LocalFrameStageGeometry::~LocalFrameStageGeometry()=default;
LocalFrameStageView LocalFrameStageGeometry::view()const{const auto& d=*data_;return {d.endpoint.pointer,d.stage->view(),d.diagnostics.pointer};}
ExecutionGraphNode LocalFrameStageGeometry::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,const LocalFrameStageInput* input){
    auto& d=*data_;return executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        initializeLocalFrameStage<<<1,1,0,d.stream>>>(input,d.controls.pointer,d.diagnostics.pointer);
        if(d.points)localFrameStageCoordinates<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,input,d.old.pointer,d.endpoint.pointer);
        d.stage->prepare(d.old.pointer,d.endpoint.pointer,d.controls.pointer,d.diagnostics.pointer,d.stream);
    });
}
void LocalFrameStageGeometry::prepare(const LocalFrameStageInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copyLocalFrameStageInput<<<1,1,0,stream>>>(input,d.input.pointer);checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,stream));
}
LocalFrameStageResult LocalFrameStageGeometry::solveBorrowed(const Point* endpoint,StageGeometryControls controls){
    auto& d=*data_;PinnedTransferQueue::Operation pending(d.transfer);
    bindLocalFrameStage<<<1,1,0,d.stream>>>(d.referenceWorld.pointer,d.referenceLocal.pointer,endpoint,controls,d.input.pointer);
    checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,d.stream));
    const auto v=view();const int faces=v.geometry.endpointGeometry.faces,cells=v.geometry.endpointGeometry.cells;
    LocalFrameStageResult result;const auto read=[&](auto& values,const auto* source,int count){values.resize(count);d.transfer.download(values.data(),source,std::size_t(count)*sizeof(*source));};
    d.transfer.download(&result.status,v.geometry.status,sizeof(result.status));d.transfer.download(&result.diagnostics,v.diagnostics,sizeof(result.diagnostics));
    read(result.points,v.points,d.points);read(result.endpoint.faces,v.geometry.endpointGeometry.face,faces);read(result.endpoint.cells,v.geometry.endpointGeometry.cell,cells);
    read(result.endpoint.distance,v.geometry.endpointGeometry.distance,cells);read(result.oldVolume,v.geometry.oldVolume,cells);read(result.sweep,v.geometry.sweptVolume,faces);read(result.velocity,v.geometry.meshVelocity,faces);
    d.transfer.finish();if(result.status.failure!=StageGeometryFailure::None){result.points.clear();result.endpoint={};result.oldVolume.clear();result.sweep.clear();result.velocity.clear();}
    return result;
}
}}}
