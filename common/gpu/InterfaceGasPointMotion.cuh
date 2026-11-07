// Included by PolyhedralGeometry.cu; owns no application-specific constitutive law.
#include "InterfaceGasPointMotion.H"
#include "ExecutionGraph.cuh"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void beginInterfaceGasMotion(const InterfaceGasPointMotionInput* input,
    NormalHeightPointMotionInput* height,InterfaceGasMotionStatus* status,const Point* displacement,
    unsigned long long* failure){
    *status={};*height={input->gasReference,input->interface.oldGas,displacement,input->geometricTolerance};*failure=ULLONG_MAX;
}
__global__ void validateInterfaceGasMotion(int count,int axis,const int* mate,const int* valence,
    const unsigned char* fixed,const Point* displacement,const int* changed,unsigned long long* first){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count||!*changed)return;
    using Failure=InterfaceGasMotionFailure;
    const auto claim=[&](int phase,Failure failure){atomicMin(first,(static_cast<unsigned long long>(phase)*count+p)*8+int(failure));};
    const auto d=displacement[p];
    if(axis>=0){
        if(!fixed[p])claim(0,Failure::EmptyFreePoint);
        else if(boundaryDetail::coordinate(d,axis)!=0)claim(0,Failure::EmptyNormal);
        else {const auto paired=displacement[mate[p]];
            if(d.x!=paired.x||d.y!=paired.y||d.z!=paired.z)claim(0,Failure::EmptyPair);}
    }
    if(fixed[p]&&!finite(d))claim(1,Failure::NonfiniteFixed);
    else if(!valence[p])claim(1,Failure::IsolatedPoint);
}
__global__ void selectInterfaceGasMotion(int count,bool anchored,const unsigned long long* first,
    const int* changed,InterfaceGasMotionStatus* status,cudaGraphConditionalHandle handle){
    status->changed=*changed;
    if(*changed){
        if(*first!=ULLONG_MAX){status->failure=InterfaceGasMotionFailure(*first%8);status->index=int((*first/8)%count);}
        else if(count&&!anchored)status->failure=InterfaceGasMotionFailure::NoAnchor;
    }
    cudaGraphSetConditional(handle,*changed&&status->failure==InterfaceGasMotionFailure::None);
}
__global__ void acceptInterfaceGasMotion(const HarmonicDeviceStatus* motion,const NormalHeightMotionControl* constraints,
    InterfaceGasMotionStatus* status){
    status->harmonic=*motion;status->constraints=*constraints;
    if(motion->errorPoint>=0){status->failure=InterfaceGasMotionFailure::Harmonic;status->index=motion->errorPoint;}
}
__global__ void publishInterfaceGasMotion(int count,const InterfaceGasPointMotionInput* input,
    const InterfaceGasMotionStatus* status,const Point* moved,Point* result){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count)
        result[p]=status->changed&&status->failure==InterfaceGasMotionFailure::None?moved[p]:input->interface.oldGas[p];
}
__global__ void copyInterfaceGasMotionInput(const InterfaceGasPointMotionInput* input,InterfaceGasPointMotionInput* output){*output=*input;}
}
struct InterfaceGasPointMotion::Implementation {
    int points=0,solidPoints=0,solidFaces=0,axis=-1;bool anchored=false;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;PinnedTransferQueue transfer;
    std::unique_ptr<ResidentInterfacePointDisplacement> displacement;
    std::unique_ptr<NormalHeightPointMotion> motion;
    Array<Point> oldSolid,endpoint,area,result;
    Array<double> thickness,previousFilm,film;
    Array<int> mate,valence;
    Array<unsigned long long> failure;
    Array<InterfaceGasPointMotionInput> input,hostInput;
    Array<NormalHeightPointMotionInput> heightInput;
    Array<InterfaceGasMotionStatus> status;
    explicit Implementation(const InterfaceGasPointMotionDefinition& d){
        points=d.gas.points.size();solidPoints=d.oldSolid.size();solidFaces=d.solidTopology.faceOffsets.size()-1;axis=d.emptyAxis;
        if(axis< -1||axis>2||(axis>=0&&d.emptyMate.size()!=std::size_t(points))
            ||d.pointThickness.size()!=std::size_t(solidPoints)||d.previousFilmThickness.size()!=d.solidFaces.size()
            ||d.filmThickness.size()!=d.solidFaces.size()||d.gas.offsets.size()!=std::size_t(points)+1||d.gas.fixed.size()!=std::size_t(points))
            throw std::invalid_argument("interface gas point motion definition shape mismatch");
        displacement.reset(new ResidentInterfacePointDisplacement(d.solidTopology,solidPoints,points,d.solidFaces,d.solidToGas));
        auto gas=d.gas;for(int g:d.solidToGas)if(g>=0)gas.fixed[g]=1;
        std::vector<int> lengths(points);for(int p=0;p<points;++p){lengths[p]=gas.offsets[p+1]-gas.offsets[p];anchored=anchored||gas.fixed[p];}
        motion.reset(new NormalHeightPointMotion(gas));
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);PinnedTransferQueue::Operation pending(transfer);
        oldSolid.upload(d.oldSolid,transfer);thickness.upload(d.pointThickness,transfer);
        previousFilm.upload(d.previousFilmThickness,transfer);film.upload(d.filmThickness,transfer);
        mate.upload(d.emptyMate,transfer);valence.upload(lengths,transfer);endpoint.reserve(solidPoints);area.reserve(solidFaces);result.reserve(points);
        input.reserve(1);hostInput.reserve(1);heightInput.reserve(1);status.reserve(1);failure.reserve(1);
        const InterfaceGasPointMotionInput initial{{oldSolid.pointer,endpoint.pointer,area.pointer,motion->initialPoints(),
            {thickness.pointer,sizeof(double)},{previousFilm.pointer,sizeof(double)},{film.pointer,sizeof(double)}},
            motion->referencePoints(),d.gas.geometricTolerance};
        transfer.upload(hostInput.pointer,&initial,sizeof(initial));transfer.finish();
    }
    ~Implementation(){transfer.discard();if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
InterfaceGasPointMotion::InterfaceGasPointMotion(const InterfaceGasPointMotionDefinition& definition):data_(new Implementation(definition)){
    auto& d=*data_;checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph,nullptr,d.input.pointer);checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
InterfaceGasPointMotion::~InterfaceGasPointMotion()=default;
InterfaceGasPointMotionView InterfaceGasPointMotion::view()const{return {data_->result.pointer,data_->status.pointer};}
ExecutionGraphNode InterfaceGasPointMotion::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,const InterfaceGasPointMotionInput* input){
    auto& d=*data_;using executionGraphDetail::appendCaptured;using executionGraphDetail::addIf;
    cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
    auto previous=appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        d.displacement->evaluate(&input->interface,d.stream);const auto displacement=d.displacement->view();
        beginInterfaceGasMotion<<<1,1,0,d.stream>>>(input,d.heightInput.pointer,d.status.pointer,displacement.displacement,d.failure.pointer);
        if(d.points)validateInterfaceGasMotion<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,d.axis,d.mate.pointer,d.valence.pointer,
            d.motion->fixedPoints(),displacement.displacement,displacement.changed,d.failure.pointer);
        selectInterfaceGasMotion<<<1,1,0,d.stream>>>(d.points,d.anchored,d.failure.pointer,displacement.changed,d.status.pointer,handle);
    });
    cudaGraph_t body=nullptr;previous=addIf(graph,handle,{previous},body);
    const auto moved=appendCaptured(body,{},d.stream,[&]{
        const auto result=d.motion->solveDevice(d.heightInput.pointer,d.stream);
        acceptInterfaceGasMotion<<<1,1,0,d.stream>>>(result.status,d.motion->constraints().control,d.status.pointer);
    });(void)moved;
    return appendCaptured(graph,{previous},d.stream,[&]{
        // The selected harmonic result has a stable owner address even when
        // this replay took the unchanged branch and did not execute the solve.
        if(d.points)publishInterfaceGasMotion<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,input,d.status.pointer,
            d.motion->deviceView().points,d.result.pointer);
    });
}
void InterfaceGasPointMotion::solveDevice(const InterfaceGasPointMotionInput* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copyInterfaceGasMotionInput<<<1,1,0,stream>>>(input,d.input.pointer);checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,stream));
}
InterfaceGasPointMotionResult InterfaceGasPointMotion::solve(const std::vector<Point>& endpoint,const std::vector<Point>& area){
    auto& d=*data_;if(endpoint.size()!=std::size_t(d.solidPoints)||area.size()!=std::size_t(d.solidFaces))
        throw std::invalid_argument("interface gas point motion input shape mismatch");
    PinnedTransferQueue::Operation pending(d.transfer);d.endpoint.upload(endpoint,d.transfer);d.area.upload(area,d.transfer);
    copyInterfaceGasMotionInput<<<1,1,0,d.stream>>>(d.hostInput.pointer,d.input.pointer);checked(cudaGetLastError());
    checked(cudaGraphLaunch(d.executable,d.stream));InterfaceGasPointMotionResult result;result.points.resize(d.points);
    d.transfer.download(&result.status,d.status.pointer,sizeof(result.status));d.transfer.download(result.points.data(),d.result.pointer,std::size_t(d.points)*sizeof(Point));d.transfer.finish();
    if(result.status.failure!=InterfaceGasMotionFailure::None)result.points.clear();return result;
}
}}}
