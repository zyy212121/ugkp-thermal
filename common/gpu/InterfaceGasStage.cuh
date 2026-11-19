
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



    if(motion->failure!=InterfaceGasMotionFailure::None){*output={};*diagnostics={};}
    else *output=*input;
}
__global__ void bindBorrowedInterfaceStage(const InterfaceGasPointMotionInput* definition,const Point* endpoint,
    const Point* area,StageGeometryControls controls,InterfaceGasStageInput* output){
    output->motion=*definition;output->motion.interface.solidEndpoint=endpoint;output->motion.interface.solidArea=area;output->controls=controls;
}
__global__ void copyInterfaceGasStageInput(const InterfaceGasStageInput* input,InterfaceGasStageInput* output){*output=*input;}
}
struct InterfaceGasStage::Implementation {
    std::unique_ptr<InterfaceGasPointMotion> motion;
    std::unique_ptr<ResidentStageGeometry> stage;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;PinnedTransferQueue transfer;
    Array<WetOffsetFace> wetFaces;
    Array<WetOffsetVertex> wetVertices;
    Array<int> invalidWet;
    int wetCount=0;
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
        wetCount=d.wetFaces.size();
        if(wetCount){
            wetFaces.upload(d.wetFaces,transfer);wetVertices.upload(d.wetVertices,transfer);
            invalidWet.reserve(1);transfer.finish();
        }
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
        if(d.points)selectInterfaceGasStagePoints<<<(d.points-1)/workPlan().faceBlockThreads+1,workPlan().faceBlockThreads,0,d.stream>>>(d.points,input,d.old.pointer);
        d.stage->prepare(d.old.pointer,d.motion->view().points,d.controls.pointer,d.diagnostics.pointer,d.stream);
        publishInterfaceGasStageStatus<<<1,1,0,d.stream>>>(d.stage->view().status,d.status.pointer,d.motion->view().status,d.diagnostics.pointer);
    });
}

void InterfaceGasStage::prepare(const InterfaceGasStageInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copyInterfaceGasStageInput<<<1,1,0,stream>>>(input,d.input.pointer);checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,stream));
}
InterfaceGasStageResult InterfaceGasStage::collect(){
    auto& d=*data_;
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
namespace {
__global__ void initializeWetOffset(int* first){*first=INT_MAX;}
__global__ void checkWetOffset(int count,const WetOffsetFace* faces,const WetOffsetVertex* vertices,
    LocalFrameStageView solid,InterfaceGasStageView gas,int* invalid){
    if(!gas.motion.status->changed||gas.motion.status->failure!=InterfaceGasMotionFailure::None||
        solid.geometry.status->failure!=StageGeometryFailure::None||gas.geometry.status->failure!=StageGeometryFailure::None)return;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=gridDim.x*blockDim.x){
        const auto face=faces[i];bool moving=false;
        for(int k=face.first;k<face.end&&!moving;++k){
            const auto v=vertices[k];const auto s=solid.points[v.solid];
            moving=s.x!=v.oldSolid.x||s.y!=v.oldSolid.y||s.z!=v.oldSolid.z;
            if(v.gas>=0){const auto p=gas.motion.points[v.gas];moving|=p.x!=v.oldGas.x||p.y!=v.oldGas.y||p.z!=v.oldGas.z;}
        }
        if(!moving)continue;
        const auto a=gas.geometry.endpointGeometry.area[face.gas],b=solid.geometry.endpointGeometry.area[face.solid];
        const double ga=::sqrt(a.x*a.x+a.y*a.y+a.z*a.z),sa=::sqrt(b.x*b.x+b.y*b.y+b.z*b.z);
        const double x=a.x/ga+b.x/sa,y=a.y/ga+b.y/sa,z=a.z/ga+b.z/sa;
        if(face.oldMismatch||::fabs(ga-sa)>1e-10*::fmax(ga,sa)||::sqrt(x*x+y*y+z*z)>1e-10)
            atomicMin(invalid,face.interfaceIndex);
    }
}
}
const int* InterfaceGasStage::invalidWetOffset()const{return data_->invalidWet.pointer;}
ExecutionGraphNode InterfaceGasStage::appendBorrowed(ExecutionGraph graph,ExecutionGraphNode dependency,
    LocalFrameStageView solid,StageGeometryControls controls){
    auto& d=*data_;auto& motion=*d.motion->data_;
    auto bound=executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        bindBorrowedInterfaceStage<<<1,1,0,d.stream>>>(motion.hostInput.pointer,solid.points,solid.geometry.endpointGeometry.area,controls,d.input.pointer);
    });
    auto checkedStage=appendTo(graph,bound,d.input.pointer);
    if(!d.wetCount)return checkedStage;
    return executionGraphDetail::appendCaptured(graph,{checkedStage},d.stream,[&]{
        initializeWetOffset<<<1,1,0,d.stream>>>(d.invalidWet.pointer);
        checkWetOffset<<<(d.wetCount+127)/128,128,0,d.stream>>>(d.wetCount,d.wetFaces.pointer,d.wetVertices.pointer,solid,view(),d.invalidWet.pointer);
    });
}
InterfaceGasStageResult InterfaceGasStage::solveBorrowed(const Point* endpoint,const Point* area,StageGeometryControls controls){
    auto& d=*data_;auto& m=*d.motion->data_;PinnedTransferQueue::Operation pending(d.transfer);
    bindBorrowedInterfaceStage<<<1,1,0,d.stream>>>(m.hostInput.pointer,endpoint,area,controls,d.input.pointer);
    checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,d.stream));return collect();
}
InterfaceGasStageResult InterfaceGasStage::solve(const std::vector<Point>& endpoint,const std::vector<Point>& area,StageGeometryControls controls){
    auto& d=*data_;auto& m=*d.motion->data_;
    if(endpoint.size()!=std::size_t(m.solidPoints)||area.size()!=std::size_t(m.solidFaces))throw std::invalid_argument("interface gas stage input shape mismatch");
    PinnedTransferQueue::Operation pending(d.transfer);m.endpoint.upload(endpoint,d.transfer);m.area.upload(area,d.transfer);
    bindBorrowedInterfaceStage<<<1,1,0,d.stream>>>(m.hostInput.pointer,m.endpoint.pointer,m.area.pointer,controls,d.input.pointer);
    checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,d.stream));return collect();
}
}}}
