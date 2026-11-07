// Included after InterfaceGasPointMotion.cuh and ResidentStageGeometry.cuh.
#include "InterfaceGasStage.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void beginInterfaceGasStage(const InterfaceGasStageInput* input,
    StageGeometryControls* controls,ugkp::diagnostics::DeviceContext* diagnostics,
    const InterfaceGasMotionStatus* motion,StageGeometryStatus* status){
    *controls=input->controls;*diagnostics={};*status={};
    if(motion->changed&&motion->failure==InterfaceGasMotionFailure::None
        &&!(motion->harmonic.residual<=motion->constraints.harmonicTolerance))
        ugkp::diagnostics::recordWarning(diagnostics,ugkp::diagnostics::makeRecord(
            ugkp::diagnostics::Code::AccuracyResidual,3402,34,0,ugkp::diagnostics::EntityKind::Face,-1,-1,
            motion->harmonic.residual,motion->constraints.harmonicTolerance,motion->harmonic.residual,
            ugkp::diagnostics::Quantity::Accuracy));
}
__global__ void selectInterfaceGasStagePoints(int count,const InterfaceGasStageInput* input,Point* old){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)old[p]=input->motion.interface.oldGas[p];
}
__global__ void publishInterfaceGasStageStatus(const StageGeometryStatus* input,StageGeometryStatus* output,
    const InterfaceGasMotionStatus* motion,ugkp::diagnostics::DeviceContext* diagnostics){
    // Failed motion has already published the immutable old coordinates as
    // nonphysical scratch. Keep its failure authoritative and discard the
    // speculative certification, including warnings. It is never accepted.
    if(motion->failure!=InterfaceGasMotionFailure::None){*output={};*diagnostics={};}
    else *output=*input;
}
__global__ void copyInterfaceGasStageInput(const InterfaceGasStageInput* input,InterfaceGasStageInput* output){*output=*input;}
}
struct InterfaceGasStage::Implementation {
    std::unique_ptr<InterfaceGasPointMotion> motion;
    std::unique_ptr<ResidentStageGeometry> stage;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;PinnedTransferQueue transfer;
    Array<Point> old;
    Array<InterfaceGasStageInput> input;
    Array<StageGeometryControls> controls;
    Array<StageGeometryStatus> status;
    Array<ugkp::diagnostics::DeviceContext> diagnostics;
    int points=0;
    explicit Implementation(const InterfaceGasStageDefinition& d):points(d.motion.gas.points.size()){
        motion.reset(new InterfaceGasPointMotion(d.motion));
        stage.reset(new ResidentStageGeometry(d.motion.gas.topology,points,d.wallFaces,d.periodicPartner,
            d.motion.emptyAxis,d.motion.emptyMate));
        old.reserve(points);input.reserve(1);controls.reserve(1);status.reserve(1);diagnostics.reserve(1);
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);
    }
    ~Implementation(){transfer.discard();if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
InterfaceGasStage::InterfaceGasStage(const InterfaceGasStageDefinition& definition):data_(new Implementation(definition)){
    auto& d=*data_;checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph,nullptr,d.input.pointer);
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
InterfaceGasStage::~InterfaceGasStage()=default;
InterfaceGasStageView InterfaceGasStage::view()const{
    const auto& d=*data_;auto geometry=d.stage->view();geometry.status=d.status.pointer;
    return {d.motion->view(),geometry,d.diagnostics.pointer};
}
ExecutionGraphNode InterfaceGasStage::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,const InterfaceGasStageInput* input){
    auto& d=*data_;using executionGraphDetail::appendCaptured;
    auto previous=d.motion->appendTo(graph,dependency,&input->motion);
    return appendCaptured(graph,{previous},d.stream,[&]{
        beginInterfaceGasStage<<<1,1,0,d.stream>>>(input,d.controls.pointer,d.diagnostics.pointer,d.motion->view().status,d.status.pointer);
        if(d.points)selectInterfaceGasStagePoints<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,input,d.old.pointer);
        d.stage->prepare(d.old.pointer,d.motion->view().points,d.controls.pointer,d.diagnostics.pointer,d.stream);
        publishInterfaceGasStageStatus<<<1,1,0,d.stream>>>(d.stage->view().status,d.status.pointer,d.motion->view().status,d.diagnostics.pointer);
    });
}

void InterfaceGasStage::prepare(const InterfaceGasStageInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copyInterfaceGasStageInput<<<1,1,0,stream>>>(input,d.input.pointer);checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,stream));
}
InterfaceGasStageResult InterfaceGasStage::solve(const std::vector<Point>& endpoint,const std::vector<Point>& area,StageGeometryControls controls){
    auto& d=*data_;auto& m=*d.motion->data_;
    if(endpoint.size()!=std::size_t(m.solidPoints)||area.size()!=std::size_t(m.solidFaces))
        throw std::invalid_argument("interface gas stage input shape mismatch");
    PinnedTransferQueue::Operation pending(d.transfer);
    m.endpoint.upload(endpoint,d.transfer);m.area.upload(area,d.transfer);
    // Immutable host-adapter bindings are restored independently of the last
    // externally composed replay; no foreign borrowed pointer survives here.
    InterfaceGasStageInput input{{{m.oldSolid.pointer,m.endpoint.pointer,m.area.pointer,m.motion->initialPoints(),
        {m.thickness.pointer,sizeof(double)},{m.previousFilm.pointer,sizeof(double)},{m.film.pointer,sizeof(double)}},
        m.motion->referencePoints(),0},controls};
    // Geometric tolerance belongs to the initialized host descriptor, not an
    // output from the previous solve. Copy that descriptor on the device.
    d.transfer.upload(d.input.pointer,&input,sizeof(input));
    copyInterfaceGasMotionInput<<<1,1,0,d.stream>>>(m.hostInput.pointer,&d.input.pointer->motion);
    checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,d.stream));
    InterfaceGasStageResult result;const auto v=view();const int faces=v.geometry.endpointGeometry.faces,cells=v.geometry.endpointGeometry.cells;
    const auto read=[&](auto& values,const auto* source,int count){values.resize(count);
        d.transfer.download(values.data(),source,std::size_t(count)*sizeof(*source));};
    d.transfer.download(&result.motion.status,v.motion.status,sizeof(result.motion.status));
    d.transfer.download(&result.status,v.geometry.status,sizeof(result.status));
    d.transfer.download(&result.diagnostics,v.diagnostics,sizeof(result.diagnostics));
    read(result.motion.points,v.motion.points,d.points);read(result.endpoint.faces,v.geometry.endpointGeometry.face,faces);
    read(result.endpoint.cells,v.geometry.endpointGeometry.cell,cells);read(result.endpoint.distance,v.geometry.endpointGeometry.distance,cells);
    read(result.oldVolume,v.geometry.oldVolume,cells);read(result.sweep,v.geometry.sweptVolume,faces);read(result.velocity,v.geometry.meshVelocity,faces);
    d.transfer.finish();
    if(result.motion.status.failure!=InterfaceGasMotionFailure::None||result.status.failure!=StageGeometryFailure::None){
        result.motion.points.clear();result.endpoint={};result.oldVolume.clear();result.sweep.clear();result.velocity.clear();}
    return result;
}
}}}
