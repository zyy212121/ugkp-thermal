// Implementation is included by PolyhedralGeometry.cu to share its CUDA
// ownership utilities and preserve the existing application link surface.
#include "HarmonicSweepCorrection.H"
#include "ExecutionGraph.cuh"
namespace ugkp { namespace gpu { namespace geometry { namespace {
__global__ void beginHarmonicSweep(const HarmonicSweepDeviceProblem* input,
    HarmonicSweepDeviceProblem* operation,int points,int rows,int columns,int planar,
    HarmonicSweepStatus* state){
    *state={};*operation=*input;auto& p=*operation;
    if(p.planarCount==-1)p.planarCount=planar;
    const int faces=rows-3*planar;
    if(p.planarCount<0||p.planarCount>planar){state->error=1;return;}
    state->rows=faces+3*p.planarCount;
    if(p.columns<0||p.columns>columns||(points&&(!p.old||!p.endpoint))
        ||(p.columns&&!p.modes)||(state->rows&&!p.rhs)||(faces&&!p.scales)
        ||(p.planarCount&&!p.planar))state->error=1;
}
__global__ void copyHarmonicSweepCoordinates(int count,const HarmonicSweepDeviceProblem* input,
    const HarmonicSweepStatus* state,Point* old){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)old[i]=state->error?Point{}:input->old[i];
}
__global__ void initializeHarmonicPass(HarmonicSweepStatus* state,const HarmonicSweepDeviceProblem* input,
    cudaGraphConditionalHandle handle){state->mode=0;cudaGraphSetConditional(handle,!state->error&&input->columns>0);}
__global__ void clearHarmonicBoundary(int count,Point* boundary){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)boundary[i]={};
}
__global__ void seedHarmonicBoundary(int count,const HarmonicSweepDeviceProblem* input,
    HarmonicSweepStatus* state,Point* boundary){
    if(state->error)return;const auto mode=input->modes[state->mode];
    if(mode.point<0||mode.point>=count||mode.partner< -1||mode.partner>=count
        ||!isfinite(mode.displacement.x)||!isfinite(mode.displacement.y)||!isfinite(mode.displacement.z)){
        state->error=1;state->failedMode=state->mode;state->errorPoint=mode.point;return;}
    boundary[mode.point]=mode.displacement;if(mode.partner>=0)boundary[mode.partner]=mode.displacement;
    ++state->modeSolves;
}
__global__ void acceptHarmonicMode(int pass,const HarmonicSweepDeviceProblem* input,
    HarmonicSweepStatus* state,const HarmonicDeviceStatus* report,HarmonicDeviceStatus* history,
    const Point* old,const Point* mode,const double* coefficients,SweepLinearizationDeviceInput* sweep){
    *sweep={old,input->endpoint,mode,input->scales,input->planar,0,false,input->planarCount};
    if(state->error)return;history[pass*input->columns+state->mode]=*report;
    if(report->errorPoint>=0){state->error=2;state->failedMode=state->mode;state->errorPoint=report->errorPoint;return;}
    sweep->coefficient=pass?coefficients[state->mode]:0;sweep->active=true;
}
__global__ void scatterHarmonicColumn(int rows,const HarmonicSweepDeviceProblem* input,
    const HarmonicSweepStatus* state,const double* column,double* matrix){
    const int r=blockIdx.x*blockDim.x+threadIdx.x;if(r<state->rows&&!state->error)
        matrix[std::size_t(r)*input->columns+state->mode]=column[r];
}
__global__ void nextHarmonicMode(const HarmonicSweepDeviceProblem* input,HarmonicSweepStatus* state,
    cudaGraphConditionalHandle handle){++state->mode;cudaGraphSetConditional(handle,!state->error&&state->mode<input->columns);}
__global__ void bindHarmonicQr(int rows,const HarmonicSweepDeviceProblem* input,
    const HarmonicSweepStatus* state,const double* matrix,MinimumNormDeviceProblem* qr){
    *qr={state->error?0:state->rows,state->error?0:input->columns,input->rankTolerance,matrix,input->rhs};
}
__global__ void acceptHarmonicQr(int rows,const HarmonicSweepDeviceProblem* input,HarmonicSweepStatus* state,
    MinimumNormDeviceView qr){
    if(state->error)return;state->rank=qr.status->rank;state->linearInitialized=qr.status->error!=1;
    if(qr.status->error){state->error=3;return;}
    for(int r=state->rank;r<state->rows;++r)if(!isfinite(qr.rhs[r])){state->error=3;return;}
    for(int c=0;c<input->columns;++c)if(!isfinite(qr.delta[c])){state->error=3;return;}
    state->linearValid=1;
}
__global__ void bindBorrowedHarmonicProblem(HarmonicSweepDeviceProblem definition,const Point* endpoint,
    const double* rhs,const double* tolerance,double rankTolerance,SweepPlanarityView layout,HarmonicSweepDeviceProblem* output){
    definition.endpoint=endpoint;definition.rhs=rhs;definition.consistencyTolerance=*tolerance;
    definition.rankTolerance=rankTolerance;
    if(layout.count){definition.planar=layout.planar;definition.planarCount=*layout.count;}
    *output=definition;
}
__global__ void copyHarmonicProblem(const HarmonicSweepDeviceProblem* input,HarmonicSweepDeviceProblem* out){*out=*input;}
}
struct HarmonicSweepCorrection::Implementation {
    int points=0,rows=0,columns=0,planarCount=0,hostRows=0;
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    PinnedTransferQueue transfer;
    std::unique_ptr<HarmonicPointMotion> harmonic;
    std::unique_ptr<ResidentSweepLinearization> sweep;
    std::unique_ptr<ResidentDenseMinimumNorm> qr;
    Array<Point> old,initial,endpoint,boundary;
    Array<double> scales,rhs,matrix;
    Array<PlanarConstraint> planar;
    Array<HarmonicBoundaryMode> modes;
    Array<HarmonicSweepDeviceProblem> input,operation;
    Array<SweepLinearizationDeviceInput> sweepInput;
    Array<MinimumNormDeviceProblem> qrInput;
    Array<HarmonicSweepStatus> state;
    Array<HarmonicDeviceStatus> reports;
    HarmonicSweepDeviceProblem hostInput;
    explicit Implementation(const HarmonicSweepDefinition& definition,int capacity){
        if(definition.faces.size()!=definition.scales.size()||definition.points.size()>std::size_t(INT_MAX)
            ||definition.modes.size()>std::size_t(INT_MAX)/2)
            throw std::invalid_argument("harmonic sweep definition shape mismatch");
        if(capacity<0||std::size_t(capacity)<definition.planar.size())throw std::invalid_argument("harmonic sweep plane capacity mismatch");
        points=definition.points.size();columns=definition.modes.size();planarCount=capacity;hostRows=definition.faces.size()+3*definition.planar.size();
        sweep.reset(new ResidentSweepLinearization(definition.topology,points,definition.faces,planarCount));
        rows=definition.faces.size()+3*std::size_t(planarCount);
        harmonic.reset(new HarmonicPointMotion(definition.points,definition.offsets,definition.neighbours,
            definition.fixed,definition.harmonicTolerance));
        qr.reset(new ResidentDenseMinimumNorm(rows,columns));
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);
        PinnedTransferQueue::Operation pending(transfer);
        initial.upload(definition.points,transfer);scales.upload(definition.scales,transfer);
        planar.upload(definition.planar,transfer);modes.upload(definition.modes,transfer);
        old.reserve(points);endpoint.reserve(points);boundary.reserve(points);rhs.reserve(rows);
        matrix.reserve(std::size_t(rows)*columns);reports.reserve(std::size_t(columns)*2);
        input.reserve(1);operation.reserve(1);sweepInput.reserve(1);qrInput.reserve(1);state.reserve(1);
        hostInput={initial.pointer,endpoint.pointer,scales.pointer,planar.pointer,modes.pointer,rhs.pointer,
            columns,definition.harmonicTolerance,0,0,int(definition.planar.size())};transfer.finish();
    }
    HarmonicSweepResult collect(bool publishCorrection=true){
        HarmonicSweepResult result;transfer.download(&result.status,state.pointer,sizeof(result.status));transfer.finish();
        result.modes.resize(result.status.modeSolves);
        transfer.download(result.modes.data(),reports.pointer,result.modes.size()*sizeof(HarmonicDeviceStatus));
        if(result.status.linearValid){result.rhs.resize(result.status.rows);transfer.download(result.rhs.data(),qr->view().rhs,std::size_t(result.status.rows)*sizeof(double));}
        if(!result.status.error&&publishCorrection){result.correction.resize(points);transfer.download(result.correction.data(),sweep->view().correction,std::size_t(points)*sizeof(Point));}
        transfer.finish();return result;
    }
    ~Implementation(){transfer.discard();if(stream)cudaStreamSynchronize(stream);
        if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);}
};
HarmonicSweepCorrection::HarmonicSweepCorrection(const HarmonicSweepDefinition& definition):HarmonicSweepCorrection(definition,definition.planar.size()){}
HarmonicSweepCorrection::HarmonicSweepCorrection(const HarmonicSweepDefinition& definition,int capacity):data_(new Implementation(definition,capacity)){
    auto& d=*data_;checked(cudaGraphCreate(&d.graph,0));appendTo(d.graph,nullptr,d.input.pointer);
    checked(cudaGraphInstantiateWithFlags(&d.executable,d.graph,0));
}
HarmonicSweepCorrection::~HarmonicSweepCorrection()=default;
HarmonicSweepDeviceView HarmonicSweepCorrection::view()const{
    const auto& d=*data_;return {d.sweep->view().correction,d.state.pointer,d.reports.pointer,d.qr->view()};
}
ExecutionGraphNode HarmonicSweepCorrection::appendTo(ExecutionGraph graph,ExecutionGraphNode dependency,
    const HarmonicSweepDeviceProblem* problem){
    auto& d=*data_;using executionGraphDetail::appendCaptured;using executionGraphDetail::addWhile;
    auto previous=appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        beginHarmonicSweep<<<1,1,0,d.stream>>>(problem,d.operation.pointer,d.points,d.rows,d.columns,d.planarCount,d.state.pointer);
        if(d.points)copyHarmonicSweepCoordinates<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,d.operation.pointer,d.state.pointer,d.old.pointer);
        d.harmonic->prepareDeviceGeometry(d.old.pointer,d.stream);
        d.sweep->resetCorrection(d.stream);
    });
    for(int pass=0;pass<2;++pass){
        if(pass){
            previous=appendCaptured(graph,{previous},d.stream,[&]{
                bindHarmonicQr<<<1,1,0,d.stream>>>(d.rows,d.operation.pointer,d.state.pointer,d.matrix.pointer,d.qrInput.pointer);
                d.qr->solve(d.qrInput.pointer,d.stream);
                acceptHarmonicQr<<<1,1,0,d.stream>>>(d.rows,d.operation.pointer,d.state.pointer,d.qr->view());
            });
        }
        cudaGraphConditionalHandle handle;checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        previous=appendCaptured(graph,{previous},d.stream,[&]{
            initializeHarmonicPass<<<1,1,0,d.stream>>>(d.state.pointer,d.operation.pointer,handle);
        });
        cudaGraph_t body=nullptr;previous=addWhile(graph,handle,{previous},body);
        appendCaptured(body,{},d.stream,[&]{
            if(d.points)clearHarmonicBoundary<<<(d.points-1)/128+1,128,0,d.stream>>>(d.points,d.boundary.pointer);
            seedHarmonicBoundary<<<1,1,0,d.stream>>>(d.points,d.operation.pointer,d.state.pointer,d.boundary.pointer);
            const auto mode=d.harmonic->solvePreparedDevice(d.old.pointer,d.boundary.pointer,nullptr,
                &d.operation.pointer->harmonicTolerance,true,d.stream);
            acceptHarmonicMode<<<1,1,0,d.stream>>>(pass,d.operation.pointer,d.state.pointer,mode.status,d.reports.pointer,
                d.old.pointer,mode.points,d.qr->view().delta,d.sweepInput.pointer);
            if(!pass){
                d.sweep->column(d.sweepInput.pointer,d.stream);
                if(d.rows)scatterHarmonicColumn<<<(d.rows-1)/128+1,128,0,d.stream>>>(d.rows,d.operation.pointer,
                    d.state.pointer,d.sweep->view().column,d.matrix.pointer);
            }else d.sweep->accumulate(d.sweepInput.pointer,d.stream);
            nextHarmonicMode<<<1,1,0,d.stream>>>(d.operation.pointer,d.state.pointer,handle);
        });
    }
    return previous;
}
void HarmonicSweepCorrection::solveDevice(const HarmonicSweepDeviceProblem* problem,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    copyHarmonicProblem<<<1,1,0,stream>>>(problem,d.input.pointer);checked(cudaGetLastError());
    checked(cudaGraphLaunch(d.executable,stream));
}
HarmonicSweepResult HarmonicSweepCorrection::solveBorrowed(const Point* endpoint,const double* rhs,
    const double* consistencyTolerance,double rankTolerance,SweepPlanarityView layout,bool publishCorrection){
    auto& d=*data_;PinnedTransferQueue::Operation pending(d.transfer);
    bindBorrowedHarmonicProblem<<<1,1,0,d.stream>>>(d.hostInput,endpoint,rhs,consistencyTolerance,rankTolerance,layout,d.input.pointer);
    checked(cudaGetLastError());checked(cudaGraphLaunch(d.executable,d.stream));return d.collect(publishCorrection);
}
HarmonicSweepResult HarmonicSweepCorrection::solve(const std::vector<Point>& endpoint,const std::vector<double>& rhs,
    double rankTolerance,double consistencyTolerance){
    auto& d=*data_;if(endpoint.size()!=std::size_t(d.points)||rhs.size()!=std::size_t(d.hostRows))
        throw std::invalid_argument("harmonic sweep solve shape mismatch");
    PinnedTransferQueue::Operation pending(d.transfer);d.endpoint.upload(endpoint,d.transfer);d.rhs.upload(rhs,d.transfer);
    d.hostInput.rankTolerance=rankTolerance;d.hostInput.consistencyTolerance=consistencyTolerance;
    d.transfer.upload(d.input.pointer,&d.hostInput,sizeof(d.hostInput));checked(cudaGraphLaunch(d.executable,d.stream));
    return d.collect();
}
}}}
