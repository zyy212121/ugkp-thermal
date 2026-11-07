// Included by PolyhedralGeometry.cu; uses its allocation and transfer owner.
#include "SweepResidual.H"
#include "SweepLineSearch.H"
namespace ugkp { namespace gpu { namespace geometry { namespace {
struct SweepResidualSummary {
    double metric=0,sweep=0,plane=0,rhs=0;
    int count=0,firstBad=INT_MAX;
    bool hadNaN=false,hadRhsNaN=false;
};
__device__ SweepResidualSummary joinSweepSummary(SweepResidualSummary a,SweepResidualSummary b){
    if(!a.count)return b;if(!b.count)return a;
    // The original ordered max replaces a prior NaN with the next finite row.
    // Preserve that order even for invalid controls while reducing in parallel.
    a.metric=b.hadNaN?b.metric:(a.metric>b.metric?a.metric:b.metric);
    a.sweep=a.sweep>b.sweep?a.sweep:b.sweep;a.plane=a.plane>b.plane?a.plane:b.plane;
    a.rhs=b.hadRhsNaN?b.rhs:(a.rhs>b.rhs?a.rhs:b.rhs);a.firstBad=a.firstBad<b.firstBad?a.firstBad:b.firstBad;
    a.count+=b.count;a.hadNaN=a.hadNaN||b.hadNaN;a.hadRhsNaN=a.hadRhsNaN||b.hadRhsNaN;return a;
}
__global__ void sweepResidualPoints(int count,int axis,const int* mate,const SweepResidualProblem* input,
    int* endpointFailure,int* emptyFailure){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;const auto in=*input;
    endpointFailure[p]=finite(in.endpoint[p])?0:1;
    emptyFailure[p]=axis<0?0:int(emptyTrajectoryPoint(p,mate[p],axis,in.old,in.endpoint));
}
__global__ void sweepResidualFaces(int count,TopologyView topology,const SweepResidualProblem* input,SweepResult* out){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f<count)out[f]=faceSweep(f,topology,input->old,input->endpoint);
}
__global__ void sweepResidualPlanes(int count,const SweepResidualProblem* input,Polynomial* values){
    const int selected=input->planarCount,active=selected==-1?count:selected;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(active>=0&&active<=count&&i<active)values[i]=planarPolynomial(input->old,input->endpoint,input->planar[i]);
}
__global__ void sweepResidualRows(int faces,int planes,const int* faceIds,const SweepResidualProblem* input,
    const SweepResult* sweeps,const Polynomial* polynomials,double* residual,double* rhs,SweepResidualSummary* partial){
    const int row=blockIdx.x*blockDim.x+threadIdx.x;const auto in=*input;SweepResidualSummary value;
    const int selected=in.planarCount==-1?planes:in.planarCount;
    if(selected>=0&&selected<=planes&&row<faces+3*selected){double magnitude=0,tolerance=0,r=0;
        if(row<faces){const double actual=sweeps[faceIds[row]].volume;const double raw=actual-in.targets[row];
            r=raw/in.scales[row];magnitude=sweepResidualAbsolute(raw);value.sweep=magnitude;
            if(!std::isfinite(raw))value.firstBad=row;
            tolerance=sweepResidualTolerance(in.targets[row],actual,in.scales[row],in.absoluteTolerance,in.relativeTolerance,in.faceCeiling,row);
        }else{const int i=(row-faces)/3,k=(row-faces)%3;const auto p=in.planar[i];
            r=polynomials[i].value[k]/p.scale;magnitude=sweepResidualAbsolute(r);value.plane=magnitude;
            if(!std::isfinite(r))value.firstBad=row;tolerance=in.planeTolerance[i];
        }
        residual[row]=r;value.metric=sweepResidualRatio(magnitude,tolerance);value.hadNaN=std::isnan(value.metric);value.count=1;
        if(row>=faces&&sweepResidualAbsolute(r)<=tolerance)r=0;
        rhs[row]=-r;value.rhs=sweepResidualAbsolute(r);value.hadRhsNaN=std::isnan(value.rhs);
    }
    __shared__ SweepResidualSummary values[128];values[threadIdx.x]=value;__syncthreads();
    for(int stride=1;stride<128;stride*=2){if(threadIdx.x%(2*stride)==0)values[threadIdx.x]=joinSweepSummary(values[threadIdx.x],values[threadIdx.x+stride]);__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void finishSweepResidual(int points,int faces,int constrained,int planes,int blocks,const int* faceIds,
    const int* endpointFailure,const int* emptyFailure,const SweepResult* sweeps,
    const SweepResidualSummary* partial,const SweepResidualProblem* input,SweepResidualStatus* output){
    // Ordered reductions preserve the native first-failure phase and index.
    __shared__ int badPoint[128],badEmpty[128],badFace[128];
    int p=INT_MAX,e=INT_MAX,f=INT_MAX;
    for(int i=threadIdx.x;i<points;i+=128){if(endpointFailure[i]&&i<p)p=i;if(emptyFailure[i]&&i<e)e=i;}
    for(int i=threadIdx.x;i<faces;i+=128)if((!finite(sweeps[i].area)||!std::isfinite(sweeps[i].volume))&&i<f)f=i;
    badPoint[threadIdx.x]=p;badEmpty[threadIdx.x]=e;badFace[threadIdx.x]=f;__syncthreads();
    for(int stride=64;stride;stride/=2){if(threadIdx.x<stride){
        if(badPoint[threadIdx.x+stride]<badPoint[threadIdx.x])badPoint[threadIdx.x]=badPoint[threadIdx.x+stride];
        if(badEmpty[threadIdx.x+stride]<badEmpty[threadIdx.x])badEmpty[threadIdx.x]=badEmpty[threadIdx.x+stride];
        if(badFace[threadIdx.x+stride]<badFace[threadIdx.x])badFace[threadIdx.x]=badFace[threadIdx.x+stride];}__syncthreads();}
    if(threadIdx.x)return;SweepResidualStatus status;SweepResidualSummary summary;
    for(int i=0;i<blocks;++i)summary=joinSweepSummary(summary,partial[i]);
    const int selected=input->planarCount==-1?planes:input->planarCount;
    status.rows=selected>=0&&selected<=planes?constrained+3*selected:0;
    status.metric=summary.metric;status.maximumSweep=summary.sweep;status.maximumPlanarity=summary.plane;
    const double floor=128*std::numeric_limits<double>::epsilon(),scaled=input->rankTolerance*summary.rhs;
    status.consistencyTolerance=floor>scaled?floor:scaled;
    if(selected<0||selected>planes){status.failure=SweepResidualFailure::Layout;status.index=selected;}
    else if(badPoint[0]!=INT_MAX){status.failure=SweepResidualFailure::Endpoint;status.index=badPoint[0];}
    else if(badEmpty[0]!=INT_MAX){status.index=badEmpty[0];status.failure=emptyFailure[status.index]==int(EmptyTrajectoryFailure::NormalDisplacement)?SweepResidualFailure::EmptyNormal:SweepResidualFailure::EmptyTangential;}
    else if(badFace[0]!=INT_MAX){status.failure=SweepResidualFailure::Sweep;status.index=badFace[0];}
    else if(summary.firstBad!=INT_MAX){const int row=summary.firstBad;
        status.failure=row<constrained?SweepResidualFailure::FaceResidual:SweepResidualFailure::PlanarResidual;
        status.index=row<constrained?faceIds[row]:input->planar[(row-constrained)/3].a;
    }else if(!std::isfinite(summary.metric))status.failure=SweepResidualFailure::Metric;
    *output=status;
}
}
struct ResidentSweepResidual::Implementation {
    int points=0,faces=0,constrained=0,planes=0,blocks=0,axis=-1;
    Array<int> offsets,vertices,faceIds,mate,endpointFailure,emptyFailure;
    Array<SweepResult> sweeps;Array<Polynomial> polynomials;Array<double> residual,rhs;
    Array<SweepResidualSummary> partial;Array<SweepResidualStatus> status;
    TopologyView topology()const{return {offsets.pointer,vertices.pointer,nullptr,nullptr,nullptr,nullptr,nullptr};}
};
ResidentSweepResidual::ResidentSweepResidual(const Topology& topology,int points,const std::vector<int>& faces,
    int planes,int axis,const std::vector<int>& mate):data_(new Implementation){
    if(points<0||planes<0||topology.faceOffsets.empty()||faces.size()>std::size_t(INT_MAX)
        ||std::size_t(planes)>(std::size_t(INT_MAX)-faces.size())/3||axis< -1||axis>2||(axis>=0&&mate.size()!=std::size_t(points)))
        throw std::invalid_argument("resident sweep residual topology shape mismatch");
    auto& d=*data_;d.points=points;d.faces=topology.faceOffsets.size()-1;d.constrained=faces.size();d.planes=planes;d.axis=axis;
    const int rows=d.constrained+3*planes;d.blocks=rows?(rows-1)/128+1:0;
    PinnedTransferQueue transfer;PinnedTransferQueue::Operation pending(transfer);
    d.offsets.upload(topology.faceOffsets,transfer);d.vertices.upload(topology.facePoints,transfer);d.faceIds.upload(faces,transfer);d.mate.upload(mate,transfer);
    d.endpointFailure.reserve(points);d.emptyFailure.reserve(points);d.sweeps.reserve(d.faces);d.residual.reserve(rows);d.rhs.reserve(rows);
    d.polynomials.reserve(planes);d.partial.reserve(d.blocks);d.status.reserve(1);transfer.finish();
}
ResidentSweepResidual::~ResidentSweepResidual()=default;
SweepResidualView ResidentSweepResidual::view()const{const auto& d=*data_;return {d.sweeps.pointer,d.residual.pointer,d.rhs.pointer,d.status.pointer};}
void ResidentSweepResidual::evaluate(const SweepResidualProblem* input,void* rawStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(rawStream);
    if(d.points)sweepResidualPoints<<<(d.points-1)/128+1,128,0,stream>>>(d.points,d.axis,d.mate.pointer,input,d.endpointFailure.pointer,d.emptyFailure.pointer);
    if(d.faces)sweepResidualFaces<<<(d.faces-1)/128+1,128,0,stream>>>(d.faces,d.topology(),input,d.sweeps.pointer);
    if(d.planes)sweepResidualPlanes<<<(d.planes-1)/128+1,128,0,stream>>>(d.planes,input,d.polynomials.pointer);
    if(d.blocks)sweepResidualRows<<<d.blocks,128,0,stream>>>(d.constrained,d.planes,d.faceIds.pointer,input,d.sweeps.pointer,d.polynomials.pointer,d.residual.pointer,d.rhs.pointer,d.partial.pointer);
    finishSweepResidual<<<1,128,0,stream>>>(d.points,d.faces,d.constrained,d.planes,d.blocks,d.faceIds.pointer,d.endpointFailure.pointer,d.emptyFailure.pointer,d.sweeps.pointer,d.partial.pointer,input,d.status.pointer);
    checked(cudaGetLastError());
}
namespace {
__global__ void bindResidualEndpoint(const Point* endpoint,SweepResidualProblem* input){input->endpoint=endpoint;}
__global__ void seedSweepLineSearch(const SweepResidualStatus* status,double* metric){*metric=status->metric;}
__global__ void bindSweepLineSearch(const SweepResidualProblem* residual,const Point* origin,const Point* correction,
    const double* initialMetric,int maximumTrials,SweepLineSearchProblem* output){
    *output={*residual,origin,correction,*initialMetric,maximumTrials};
}
__global__ void bindResidualPlanarity(SweepPlanarityView layout,SweepResidualProblem* input){
    input->planar=layout.planar;input->planeTolerance=layout.tolerance;input->planarCount=*layout.count;
}
}
struct SweepResidualEvaluation::Implementation {
    int points=0,faces=0,rows=0,planeCapacity=0;cudaStream_t stream=nullptr;PinnedTransferQueue transfer;
    std::unique_ptr<ResidentSweepResidual> owner;
    std::unique_ptr<ResidentSweepLineSearch> search;
    Array<Point> lineOrigin;Array<double> lineMetric;Array<SweepLineSearchProblem> searchInput;
    const Point* currentEndpoint=nullptr;
    std::unique_ptr<ResidentSweepPlanarity> generated;Array<SweepPlanarityInput> layoutInput;Array<int> planeCount;
    Array<Point> old,endpoint;Array<double> targets,scales,ceiling,planeTolerance;
    Array<PlanarConstraint> planar;Array<SweepResidualProblem> problem;
    explicit Implementation(const SweepResidualDefinition& d,const double* uncertainty=nullptr){
        if(d.targets.size()!=d.faces.size()||d.scales.size()!=d.faces.size()
            ||(!d.faceCeiling.empty()&&d.faceCeiling.size()!=d.faces.size())||d.planeTolerance.size()!=d.planar.size())
            throw std::invalid_argument("sweep residual definition shape mismatch");
        points=d.old.size();faces=d.topology.faceOffsets.size()-1;
        if(uncertainty)generated.reset(new ResidentSweepPlanarity(d.topology));
        planeCapacity=generated?generated->capacity():int(d.planar.size());
        owner.reset(new ResidentSweepResidual(d.topology,points,d.faces,planeCapacity,d.emptyAxis,d.emptyMate));
        rows=d.faces.size()+3*std::size_t(planeCapacity);
        search.reset(new ResidentSweepLineSearch(points,*owner));lineOrigin.reserve(points);lineMetric.reserve(1);searchInput.reserve(1);
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);PinnedTransferQueue::Operation pending(transfer);
        old.upload(d.old,transfer);endpoint.reserve(points);currentEndpoint=endpoint.pointer;targets.upload(d.targets,transfer);scales.upload(d.scales,transfer);
        ceiling.upload(d.faceCeiling,transfer);planeTolerance.upload(d.planeTolerance,transfer);planar.upload(d.planar,transfer);problem.reserve(1);
        const SweepResidualProblem input{old.pointer,endpoint.pointer,targets.pointer,scales.pointer,ceiling.pointer,planeTolerance.pointer,
            planar.pointer,d.absoluteTolerance,d.relativeTolerance,d.rankTolerance};transfer.upload(problem.pointer,&input,sizeof(input));
        planeCount.upload(std::vector<int>{int(d.planar.size())},transfer);
        if(generated){layoutInput.reserve(1);const SweepPlanarityInput layout{old.pointer,*uncertainty};transfer.upload(layoutInput.pointer,&layout,sizeof(layout));
            generated->evaluate(layoutInput.pointer,stream);bindResidualPlanarity<<<1,1,0,stream>>>(generated->view(),problem.pointer);checked(cudaGetLastError());}
        transfer.finish();
    }
    ~Implementation(){transfer.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
SweepResidualEvaluation::SweepResidualEvaluation(const SweepResidualDefinition& d):data_(new Implementation(d)){}
SweepResidualEvaluation::SweepResidualEvaluation(const SweepResidualDefinition& d,double uncertainty):data_(new Implementation(d,&uncertainty)){}
SweepResidualEvaluation::~SweepResidualEvaluation()=default;
SweepPlanarityView SweepResidualEvaluation::planarity()const{const auto& d=*data_;return d.generated?d.generated->view():SweepPlanarityView{d.planar.pointer,d.planeTolerance.pointer,d.planeCount.pointer};}
int SweepResidualEvaluation::planeCapacity()const{return data_->planeCapacity;}
SweepResidualView SweepResidualEvaluation::view()const{return data_->owner->view();}
const Point* SweepResidualEvaluation::endpoint()const{return data_->currentEndpoint;}
SweepResidualStatus SweepResidualEvaluation::evaluate(const std::vector<Point>& endpoint){
    auto& d=*data_;if(endpoint.size()!=std::size_t(d.points))throw std::invalid_argument("sweep residual endpoint shape mismatch");
    PinnedTransferQueue::Operation pending(d.transfer);d.endpoint.upload(endpoint,d.transfer);d.currentEndpoint=d.endpoint.pointer;
    bindResidualEndpoint<<<1,1,0,d.stream>>>(d.currentEndpoint,d.problem.pointer);d.owner->evaluate(d.problem.pointer,d.stream);
    SweepResidualStatus result;d.transfer.download(&result,d.owner->view().status,sizeof(result));d.transfer.finish();return result;
}
void SweepResidualEvaluation::beginLineSearch(){
    auto& d=*data_;
    if(d.points)checked(cudaMemcpyAsync(d.lineOrigin.pointer,d.currentEndpoint,std::size_t(d.points)*sizeof(Point),cudaMemcpyDeviceToDevice,d.stream));
    seedSweepLineSearch<<<1,1,0,d.stream>>>(d.owner->view().status,d.lineMetric.pointer);checked(cudaGetLastError());
}
SweepLineSearchStatus SweepResidualEvaluation::lineSearch(const Point* correction,int maximumTrials,bool restart){
    auto& d=*data_;PinnedTransferQueue::Operation pending(d.transfer);
    if(restart){
        bindSweepLineSearch<<<1,1,0,d.stream>>>(d.problem.pointer,d.lineOrigin.pointer,correction,d.lineMetric.pointer,maximumTrials,d.searchInput.pointer);
        checked(cudaGetLastError());d.search->begin(d.searchInput.pointer,d.stream);
    }else d.search->resume(d.stream);
    d.currentEndpoint=d.search->view().points;
    SweepLineSearchStatus result;d.transfer.download(&result,d.search->view().status,sizeof(result));d.transfer.finish();return result;
}
std::vector<Point> SweepResidualEvaluation::publishEndpoint(){
    auto& d=*data_;PinnedTransferQueue::Operation pending(d.transfer);std::vector<Point> result(d.points);
    d.transfer.download(result.data(),d.currentEndpoint,std::size_t(d.points)*sizeof(Point));d.transfer.finish();return result;
}
}}}
