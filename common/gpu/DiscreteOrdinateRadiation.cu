#include "DiscreteOrdinateRadiation.H"
#include "MiePhaseBalance.cuh"
#include "OrderedUpwindGraph.cuh"
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <climits>
#include <sstream>
namespace ugkp { namespace gpu { namespace radiationDetail {
void checked(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA discrete ordinates: ")+cudaGetErrorString(status));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t size=0;
    Array()=default;explicit Array(std::size_t n){resize(n);}
    ~Array(){if(ptr)cudaFree(ptr);}
    Array(const Array&)=delete;Array& operator=(const Array&)=delete;
    void resize(std::size_t n){if(n==size)return;if(ptr)checked(cudaFree(ptr));ptr=nullptr;size=n;if(n)checked(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)));}
    void upload(const std::vector<T>& x,PinnedTransferQueue& queue){resize(x.size());queue.upload(ptr,x.data(),size*sizeof(T));}
    void download(std::vector<T>& x,PinnedTransferQueue& queue){x.resize(size);queue.download(x.data(),ptr,size*sizeof(T));}
};
constexpr double fourPi=12.56637061435917295385057353311801153679;
__device__ double blackIntensity(double t,double band){const double square=t*t;return band*5.670374419e-8*square*square/(fourPi/4);}
struct Failure {double value;int code,iteration,cell,patch,face;};
enum Location {Incident=1,Specular,Diffuse,Scattering,Linear,Moment,Outgoing};
struct Errors {unsigned long long* first;Failure* records;};
__device__ double consume(double value,double reference,int code,int iteration,int cell,int patch,int face,
    Errors errors,int slot,unsigned int priority){
    if(std::isfinite(value)){
        if(value>=0)return value;
        const double magnitude=fabs(reference),scale=magnitude>1?magnitude:1;
        if(value>=-upwindIntensityRoundoff*scale)return 0;
    }
    errors.records[slot]={value,code,iteration,cell,patch,face};
    atomicMin(errors.first,(static_cast<unsigned long long>(priority)<<32)|unsigned(slot));
    return value;
}
struct Geometry {
    int cells,directions,faces,boundary;
    const int *owner,*neighbour,*boundaryOwner,*boundaryPatch,*boundaryFace,*reflected,*boundaryOffsets,*boundaryCells;
    const int *reflectionOffsets,*reflectionDonors;
    const double* reflectionWeights;
    const RadiationBoundary* boundaryType;
    const double *volume,*weights,*internalFlux,*transportDiagonal,*boundaryFlux,*cosine,*area,*denominator,*fixed;
};
struct Fields {
    double *absorption,*scattering,*extinction,*asymmetry,*emission,*wallBlack,*scaling,*reference;
    MieOpticalQuery* query;
    MieOpticalTableView table;const MieBracket* phaseBrackets;
    const double* emissivity;
};
__global__ void resetErrors(Errors errors){if(!threadIdx.x)*errors.first=ULLONG_MAX;}
__global__ void optics(Geometry g,Fields f,const double* epsilon,const double* temperature,const double* diameter,
    double cutoff,double* initial,unsigned char* valid){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    f.absorption[cell]=f.scattering[cell]=f.extinction[cell]=f.asymmetry[cell]=f.emission[cell]=0;
    f.query[cell]={diameter[cell],temperature[cell],0};initial[cell]=0;valid[cell]=0;
    if(epsilon[cell]<=cutoff)return;
    const auto optical=mieOpticalProperties(f.table,f.query[cell]);
    f.absorption[cell]=3*epsilon[cell]*optical.absorption/(2*diameter[cell]);
    f.scattering[cell]=3*epsilon[cell]*optical.scattering/(2*diameter[cell]);
    f.extinction[cell]=f.absorption[cell]+f.scattering[cell];f.asymmetry[cell]=optical.asymmetry;
    initial[cell]=f.emission[cell]=blackIntensity(temperature[cell],optical.band);valid[cell]=1;
}
__global__ void wallOptics(Geometry g,Fields f,const double* temperature,double* initial,unsigned char* valid){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const auto type=g.boundaryType[face];valid[g.cells+face]=0;initial[g.cells+face]=0;f.wallBlack[face]=0;
    if(type==RadiationBoundary::Diffuse||type==RadiationBoundary::Black){
        f.wallBlack[face]=blackIntensity(temperature[face],mieBand(f.table,temperature[face]));
        initial[g.cells+face]=f.wallBlack[face];valid[g.cells+face]=1;
    }else if(type==RadiationBoundary::Fixed){initial[g.cells+face]=g.fixed[face];valid[g.cells+face]=1;}
}
__global__ void fillInitial(std::size_t count,double* values,const double* reference){const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)values[i]=*reference;}
__device__ double incoming(Geometry g,Fields f,const double* intensities,const double* incident,int direction,int face,
    Errors errors,int slot,unsigned priority){
    const int cell=g.boundaryOwner[face],patch=g.boundaryPatch[face],local=g.boundaryFace[face];
    switch(g.boundaryType[face]){
    case RadiationBoundary::Specular:{
        double value=0;const int row=direction*g.boundary+face;
        for(int e=g.reflectionOffsets[row];e<g.reflectionOffsets[row+1];++e)
            value+=g.reflectionWeights[e]*consume(intensities[std::size_t(g.reflectionDonors[e])*g.cells+cell],
                intensities[std::size_t(direction)*g.cells+cell],Specular,0,cell,patch,local,errors,slot,priority);
        return value;
    }
    case RadiationBoundary::Zero:return 0;
    case RadiationBoundary::Fixed:return g.fixed[face];
    case RadiationBoundary::Black:return f.wallBlack[face];
    default:return f.emissivity[face]*f.wallBlack[face]+(1-f.emissivity[face])*
        consume(incident[face],incident[face],Diffuse,0,cell,patch,local,errors,slot,priority)/g.denominator[face];
    }
}
__global__ void incidentFlux(Geometry g,const double* intensities,double* incident,Errors errors){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const int cell=g.boundaryOwner[face];double result=0;
    for(int direction=0;direction<g.directions;++direction){const double cosine=g.cosine[direction*g.boundary+face];if(cosine>0){
        const double value=intensities[std::size_t(direction)*g.cells+cell];
        result+=g.weights[direction]*cosine*consume(value,value,Incident,0,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        if(!std::isfinite(value)||value < -upwindIntensityRoundoff*(fabs(value)>1?fabs(value):1))return;
    }}incident[face]=result;
}
__global__ void prepareDirection(Geometry g,Fields f,int direction,int iteration,const double* intensities,
    const double* incident,double* diagonal,double* source,double* solved,Errors errors){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    diagonal[cell]=g.transportDiagonal[std::size_t(direction)*g.cells+cell]+(f.absorption[cell]+f.scattering[cell])*g.volume[cell];
    double rhs=0;
    for(int i=g.boundaryOffsets[cell];i<g.boundaryOffsets[cell+1];++i){const int face=g.boundaryCells[i];const double flux=g.boundaryFlux[direction*g.boundary+face];
        if(flux<0){const double value=incoming(g,f,intensities,incident,direction,face,errors,cell,face);
            // A failed consumer is the first error for this cell; later work must
            // not overwrite its unique failure record before the stream completes.
            if(errors.records[cell].code)return;
            rhs-=flux*value;}}
    double mean=0;
    if(f.scattering[cell]!=0){
        const auto d=mieLogBracket(f.table.diameter,f.table.diameters,f.query[cell].diameter);
        const auto t=mieLinearBracket(f.table.temperature,f.table.temperatures,f.query[cell].temperature);
        for(int other=0;other<g.directions;++other){const double value=consume(intensities[std::size_t(other)*g.cells+cell],*f.reference,
            Scattering,iteration,cell,-1,-1,errors,cell,g.boundary+cell);
            if(errors.records[cell].code)return;
            mean+=mieDetail::phaseProbability(f.table,d,t,g.weights,f.phaseBrackets,f.scaling+std::size_t(cell)*g.directions,g.directions,direction,other)*value;}
    }
    source[cell]=rhs+g.volume[cell]*(f.absorption[cell]*f.emission[cell]+f.scattering[cell]*mean);
    solved[cell]=intensities[std::size_t(direction)*g.cells+cell];
}
__global__ void relaxedDirection(int cells,int direction,double relaxation,const double* old,double* solved,double* intensities,double* change,double* magnitude){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells)return;
    const std::size_t offset=std::size_t(direction)*cells+cell;const double before=old[offset];
    const double after=(1-relaxation)*before+relaxation*solved[cell];
    change[cell]=fabs(after-before);magnitude[cell]=fabs(after);solved[cell]=after;intensities[offset]=after;
}
__global__ void collectReports(int count,const UpwindLinearReport* const* input,UpwindLinearReport* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=*input[i];
}
__global__ void ratio(const double* a,const double* b,double* output){if(!threadIdx.x)*output=*a/ *b;}
__global__ void moments(Geometry g,Fields f,const double* intensities,int iteration,double* incident,double* power,Errors errors){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    double value=0;
    for(int direction=0;direction<g.directions;++direction){value+=g.weights[direction]*consume(intensities[std::size_t(direction)*g.cells+cell],
        *f.reference,Moment,iteration,cell,-1,-1,errors,cell,direction*g.cells+cell);if(errors.records[cell].code)return;}
    incident[cell]=value;power[cell]=f.absorption[cell]==0?0:f.absorption[cell]*(value-fourPi*f.emission[cell]);
}
__global__ void boundaryResult(Geometry g,Fields f,const double* intensities,const double* incident,int iteration,double* output,Errors errors){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const int cell=g.boundaryOwner[face];double result=0;
    for(int direction=0;direction<g.directions;++direction){const double cosine=g.cosine[direction*g.boundary+face];double value=0;
        if(cosine>0)value=consume(intensities[std::size_t(direction)*g.cells+cell],*f.reference,Outgoing,iteration,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        else if(cosine<0)value=incoming(g,f,intensities,incident,direction,face,errors,face,face);
        if(errors.records[face].code)return;result+=g.weights[direction]*cosine*value;
    }output[face]=result;
}
const char* location(int code){switch(code){case Incident:return "incident boundary-flux reconstruction";case Specular:return "specular boundary reflection";
case Diffuse:return "diffuse reflected incident flux";case Scattering:return "table-Mie scattering source";case Linear:return "FV upwind linear solve";
case Moment:return "incident-radiation moment";default:return "outgoing boundary intensity";}}
[[noreturn]] void fail(const Failure& error){throw RadiationError(std::string(std::isfinite(error.value)?"negative radiation intensity beyond roundoff at ":"nonfinite radiation intensity at ")+
    location(error.code),error.value,error.iteration,error.cell,error.patch,error.face);}
}}}
namespace ugkp { namespace gpu {
using namespace radiationDetail;
struct DiscreteOrdinateRadiation::Implementation {
    RadiationGeometry geometry;RadiationControls controls;MieOpticalProperties optical;
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    Array<int> owner,neighbour,boundaryOwner,boundaryPatch,boundaryFace,reflected,boundaryOffsets,boundaryCells;
    Array<RadiationBoundary> boundaryType;
    Array<double> volume,weights,internalFlux,transportDiagonal,boundaryFlux,cosines,area,denominator,fixed,directionCosines;
    Array<double> epsilon,temperature,diameter,wallTemperature,emissivity;
    Array<double> absorption,scattering,extinction,asymmetry,emission,wallBlack,scaling,nextScaling,rowError,columnError,reference;
    Array<MieOpticalQuery> queries;Array<MieBracket> phaseBrackets;Array<MiePhaseMetrics> phaseMetrics;
    Array<double> intensities,frozen,incident,initial,incidentRadiation,power,wallFlux,directionChange;
    // finish() is ordered on the owner stream; these reductions are shared scratch.
    Array<double> change,magnitude;
    Array<unsigned char> initialValid;
    Array<int> reflectionOffsets,reflectionDonors;Array<double> reflectionWeights;
    Array<OrderedMaximum> partialA,partialB;Array<double> reductionA,reductionB;
    Array<unsigned long long> firstError;Array<Failure> failures;
    Array<const UpwindLinearReport*> reportPointers;Array<UpwindLinearReport> reports;
    Geometry g{};Fields f{};
    struct DirectionWorkspace {
        Array<double> diagonal,source,solved,previous;
        Array<unsigned long long> firstError;Array<Failure> failures;
        cudaStream_t stream=nullptr;
        DirectionWorkspace(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
        ~DirectionWorkspace(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    };
    struct Direction {
        OrderedUpwindPlan plan;Array<int> offsets,neighbours,subtract,residualOrder,levels;Array<double> flux;
        std::shared_ptr<DirectionWorkspace> workspace;
        Array<double> &diagonal,&source,&solved,&previous;
        Array<unsigned long long>& firstError;Array<Failure>& failures;
        std::unique_ptr<OrderedUpwindGraph> graph;
        cudaStream_t stream=nullptr;cudaEvent_t complete=nullptr;
        explicit Direction(std::shared_ptr<DirectionWorkspace> w):workspace(std::move(w)),
            diagonal(workspace->diagonal),source(workspace->source),solved(workspace->solved),previous(workspace->previous),
            firstError(workspace->firstError),failures(workspace->failures),stream(workspace->stream){
            checked(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming));}
        ~Direction(){if(stream)cudaStreamSynchronize(stream);graph.reset();if(complete)cudaEventDestroy(complete);}
        Errors errors(){return {firstError.ptr,failures.ptr};}
    };
    int concurrentAngles=1;
    std::vector<std::unique_ptr<Direction>> directions;
    Implementation(const RadiationGeometry& input,const MieOpticalTable& table,const RadiationControls& options):geometry(input),controls(options),optical(table){
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);
    }
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);directions.clear();cudaStreamDestroy(stream);}}
    Errors errors(){return {firstError.ptr,failures.ptr};}
    void reset(){resetErrors<<<1,1,0,stream>>>(errors());checked(cudaMemsetAsync(failures.ptr,0,failures.size*sizeof(Failure),stream));}
    void checkErrors(){unsigned long long key=ULLONG_MAX;transfers.download(&key,firstError.ptr,sizeof(key));transfers.finish();
        if(key!=ULLONG_MAX)readFailure(key,failures.ptr);}
    void readFailure(unsigned long long key,const Failure* records){Failure error;transfers.download(&error,records+unsigned(key),sizeof(error));transfers.finish();fail(error);}
    void reset(Direction& d,cudaStream_t target){resetErrors<<<1,1,0,target>>>(d.errors());checked(cudaMemsetAsync(d.failures.ptr,0,d.failures.size*sizeof(Failure),target));}
    void maximum(int count,const double* input,const unsigned char* valid,double initialValue,double* output){
        int blocks=std::max(1,(count+255)/256);orderedMaximumPartials<<<blocks,256,0,stream>>>(count,input,valid,partialA.ptr);
        auto* current=partialA.ptr;auto* next=partialB.ptr;
        while(blocks>1){const int n=(blocks+255)/256;orderedMaximumMerge<<<n,256,0,stream>>>(blocks,current,next);std::swap(current,next);blocks=n;}
        upwindGraphDetail::maximumResult<<<1,1,0,stream>>>(current,initialValue,output);
    }
    void initialize(){
        PinnedTransferQueue::Operation operation(transfers);
        const int cells=geometry.cells,angles=geometry.directions,boundary=geometry.boundaryOwner.size();const std::size_t angular=std::size_t(cells)*angles;
        owner.upload(geometry.owner,transfers);neighbour.upload(geometry.neighbour,transfers);
        boundaryOwner.upload(geometry.boundaryOwner,transfers);boundaryPatch.upload(geometry.boundaryPatch,transfers);boundaryFace.upload(geometry.boundaryFace,transfers);
        reflected.upload(geometry.reflectedDirection,transfers);boundaryType.upload(geometry.boundaryType,transfers);
        volume.upload(geometry.volume,transfers);weights.upload(geometry.weights,transfers);internalFlux.upload(geometry.internalFlux,transfers);
        transportDiagonal.upload(geometry.transportDiagonal,transfers);boundaryFlux.upload(geometry.boundaryFlux,transfers);cosines.upload(geometry.boundaryCosine,transfers);
        area.upload(geometry.boundaryArea,transfers);denominator.upload(geometry.diffuseDenominator,transfers);fixed.upload(geometry.fixedIntensity,transfers);
        directionCosines.upload(geometry.directionCosines,transfers);
        std::vector<std::vector<int>> perCell(cells);for(int face=0;face<boundary;++face)perCell[geometry.boundaryOwner[face]].push_back(face);
        std::vector<int> offsets(1,0),entries;for(const auto& row:perCell){entries.insert(entries.end(),row.begin(),row.end());offsets.push_back(entries.size());}
        boundaryOffsets.upload(offsets,transfers);boundaryCells.upload(entries,transfers);
        for(auto* field:{&absorption,&scattering,&extinction,&asymmetry,&emission,&incidentRadiation,&power})field->resize(cells);
        wallBlack.resize(boundary);wallFlux.resize(boundary);reference.resize(1);queries.resize(cells);phaseMetrics.resize(cells);phaseBrackets.resize(std::size_t(angles)*angles);
        for(auto* field:{&scaling,&nextScaling,&rowError,&columnError,&intensities,&frozen})field->resize(angular);
        change.resize(cells);magnitude.resize(cells);
        incident.resize(boundary);initial.resize(cells+boundary);initialValid.resize(cells+boundary);directionChange.resize(angles);
        reductionA.resize(1);reductionB.resize(1);firstError.resize(1);failures.resize(std::max({1,cells,boundary}));
        partialA.resize(std::max(1,(std::max(cells+boundary,angles)+255)/256));partialB.resize(partialA.size);
        epsilon.resize(cells);temperature.resize(cells);diameter.resize(cells);wallTemperature.resize(boundary);emissivity.resize(boundary);
        if(std::find(geometry.boundaryType.begin(),geometry.boundaryType.end(),RadiationBoundary::Specular)!=geometry.boundaryType.end()
            &&geometry.reflectionOffsets.size()!=std::size_t(angles)*boundary+1)
            throw std::invalid_argument("specular boundary requires conservative reflection map");
        if(!geometry.reflectionOffsets.empty()){
            const auto& offsets=geometry.reflectionOffsets;
            if(offsets.size()!=std::size_t(angles)*boundary+1||offsets.front()!=0
                ||offsets.back()!=int(geometry.reflectionDonors.size())
                ||geometry.reflectionWeights.size()!=geometry.reflectionDonors.size())
                throw std::invalid_argument("invalid specular reflection map shape");
            for(std::size_t row=0;row+1<offsets.size();++row){
                if(offsets[row]<0||offsets[row]>offsets[row+1]||offsets[row+1]>offsets.back())
                    throw std::invalid_argument("invalid specular reflection offsets");
                for(int j=offsets[row];j<offsets[row+1];++j)
                    if(geometry.reflectionDonors[j]<0||geometry.reflectionDonors[j]>=angles
                        ||!std::isfinite(geometry.reflectionWeights[j])||geometry.reflectionWeights[j]<0)
                        throw std::invalid_argument("invalid specular reflection entry");
            }
        }
        reflectionOffsets.upload(geometry.reflectionOffsets,transfers);
        reflectionDonors.upload(geometry.reflectionDonors,transfers);
        reflectionWeights.upload(geometry.reflectionWeights,transfers);
        g={cells,angles,int(geometry.owner.size()),boundary,owner.ptr,neighbour.ptr,boundaryOwner.ptr,boundaryPatch.ptr,boundaryFace.ptr,
            reflected.ptr,boundaryOffsets.ptr,boundaryCells.ptr,reflectionOffsets.ptr,reflectionDonors.ptr,reflectionWeights.ptr,boundaryType.ptr,volume.ptr,weights.ptr,internalFlux.ptr,transportDiagonal.ptr,
            boundaryFlux.ptr,cosines.ptr,area.ptr,denominator.ptr,fixed.ptr};
        f={absorption.ptr,scattering.ptr,extinction.ptr,asymmetry.ptr,emission.ptr,wallBlack.ptr,scaling.ptr,reference.ptr,queries.ptr,optical.deviceTable(),phaseBrackets.ptr,emissivity.ptr};
        mieDetail::phaseBrackets<<<std::max(1,(angles*angles+127)/128),128,0,stream>>>(angles*angles,f.table,directionCosines.ptr,phaseBrackets.ptr);
        transfers.finish();
        concurrentAngles=controls.parallelAngles?std::max(1,std::min(angles,controls.maximumConcurrentAngles>0?controls.maximumConcurrentAngles:angles)):1;
        std::vector<std::shared_ptr<DirectionWorkspace>> workspaces;
        for(int i=0;i<concurrentAngles;++i)workspaces.push_back(std::make_shared<DirectionWorkspace>());
        std::vector<const UpwindLinearReport*> deviceReports;
        for(int angle=0;angle<angles;++angle){
            std::unique_ptr<Direction> next(new Direction(workspaces[angle%concurrentAngles]));auto& d=*next;
            std::vector<double> flux(geometry.internalFlux.begin()+std::size_t(angle)*g.faces,geometry.internalFlux.begin()+std::size_t(angle+1)*g.faces);
            d.plan.build(cells,geometry.owner,geometry.neighbour,geometry.cellOffsets,geometry.cellFaces,flux);
            d.offsets.upload(d.plan.offsets,transfers);d.neighbours.upload(d.plan.neighbour,transfers);d.subtract.upload(d.plan.subtract,transfers);
            d.residualOrder.upload(d.plan.residualOrder,transfers);d.levels.upload(d.plan.levelCells,transfers);d.flux.upload(d.plan.flux,transfers);
            for(auto* field:{&d.diagonal,&d.source,&d.solved,&d.previous})field->resize(cells);
            transfers.finish();
            d.firstError.resize(1);d.failures.resize(std::max(1,cells));
            OrderedUpwindDevice matrix{cells,d.offsets.ptr,d.neighbours.ptr,d.subtract.ptr,d.residualOrder.ptr,d.levels.ptr,d.flux.ptr};
            d.graph.reset(new OrderedUpwindGraph(d.plan,matrix,g.faces,owner.ptr,neighbour.ptr,internalFlux.ptr+std::size_t(angle)*g.faces,
                d.diagonal.ptr,d.source.ptr,d.solved.ptr,d.previous.ptr,reference.ptr,
                {controls.linearTolerance,controls.linearRelativeTolerance,controls.small,controls.linearIterations},d.firstError.ptr,controls.parallelAngles));
            deviceReports.push_back(d.graph->report());
            directions.push_back(std::move(next));
        }
        reportPointers.upload(deviceReports,transfers);reports.resize(angles);transfers.finish();
    }
};
DiscreteOrdinateRadiation::DiscreteOrdinateRadiation(const RadiationGeometry& geometry,const MieOpticalTable& table,const RadiationControls& controls):
    data_(new Implementation(geometry,table,controls)){data_->initialize();}
DiscreteOrdinateRadiation::~DiscreteOrdinateRadiation()=default;
RadiationResult DiscreteOrdinateRadiation::solve(const RadiationSnapshot& snapshot,bool warmStart){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    d.epsilon.upload(snapshot.epsilon,d.transfers);d.temperature.upload(snapshot.temperature,d.transfers);d.diameter.upload(snapshot.diameter,d.transfers);
    d.wallTemperature.upload(snapshot.wallTemperature,d.transfers);d.emissivity.upload(snapshot.emissivity,d.transfers);
    const auto g=d.g;const auto f=d.f;const int blocks=std::max(1,(g.cells+127)/128),boundaryBlocks=std::max(1,(g.boundary+127)/128);
    optics<<<blocks,128,0,d.stream>>>(g,f,d.epsilon.ptr,d.temperature.ptr,d.diameter.ptr,d.controls.activeEpsilonMinimum,d.initial.ptr,d.initialValid.ptr);
    wallOptics<<<boundaryBlocks,128,0,d.stream>>>(g,f,d.wallTemperature.ptr,d.initial.ptr,d.initialValid.ptr);
    d.maximum(g.cells+g.boundary,d.initial.ptr,d.initialValid.ptr,0,d.reference.ptr);
    if(g.cells)mieDetail::balancePhase<<<g.cells,128,0,d.stream>>>(g.cells,g.directions,f.table,d.queries.ptr,d.weights.ptr,d.phaseBrackets.ptr,
        d.controls.phaseTolerance,d.controls.phaseIterations,d.scaling.ptr,d.nextScaling.ptr,d.rowError.ptr,d.columnError.ptr,d.phaseMetrics.ptr,d.scattering.ptr);
    if(!warmStart)fillInitial<<<std::max(std::size_t(1),(d.intensities.size+127)/128),128,0,d.stream>>>(d.intensities.size,d.intensities.ptr,d.reference.ptr);
    RadiationResult result;result.directionWarnings.resize(g.directions);
    for(int iteration=1;iteration<=std::max(1,d.controls.sourceIterations);++iteration){
        d.reset();
        if(d.controls.parallelAngles)checked(cudaMemcpyAsync(d.frozen.ptr,d.intensities.ptr,d.intensities.size*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
        const double* angular=d.controls.parallelAngles?d.frozen.ptr:d.intensities.ptr;
        incidentFlux<<<boundaryBlocks,128,0,d.stream>>>(g,angular,d.incident.ptr,d.errors());d.checkErrors();
        const auto prepare=[&](int angle,cudaStream_t target){auto& q=*d.directions[angle];d.reset(q,target);
            prepareDirection<<<blocks,128,0,target>>>(g,f,angle,iteration,angular,d.incident.ptr,q.diagonal.ptr,q.source.ptr,q.solved.ptr,q.errors());};
        const auto finish=[&](int angle,const UpwindLinearReport& report){auto& q=*d.directions[angle];
            if(report.externalFailure!=ULLONG_MAX)d.readFailure(report.externalFailure,q.failures.ptr);
            if(report.errorCell>=0)fail({report.errorValue,Linear,iteration,report.errorCell,-1,-1});
            if(report.finalResidual>report.target)result.directionWarnings[angle]={report.finalResidual,report.target,report.iterations,iteration};
            relaxedDirection<<<blocks,128,0,d.stream>>>(g.cells,angle,d.controls.sourceRelaxation,angular,q.solved.ptr,d.intensities.ptr,d.change.ptr,d.magnitude.ptr);
            d.maximum(g.cells,d.change.ptr,nullptr,0,d.reductionA.ptr);d.maximum(g.cells,d.magnitude.ptr,nullptr,d.controls.small,d.reductionB.ptr);
            ratio<<<1,1,0,d.stream>>>(d.reductionA.ptr,d.reductionB.ptr,d.directionChange.ptr+angle);
        };
        if(d.controls.parallelAngles){
            // Frozen source semantics are unchanged across batches. Scratch is
            // reused only after each batch has finished publication on owner stream.
            for(int begin=0;begin<g.directions;begin+=d.concurrentAngles){
                const int end=std::min(g.directions,begin+d.concurrentAngles),count=end-begin;
                for(int angle=begin;angle<end;++angle){auto& q=*d.directions[angle];prepare(angle,q.stream);q.graph->begin(q.stream);q.graph->advance(q.stream,1);
                    checked(cudaEventRecord(q.complete,q.stream));checked(cudaStreamWaitEvent(d.stream,q.complete,0));}
                std::vector<UpwindLinearReport> reports(count);
                for(;;){
                    collectReports<<<std::max(1,(count+127)/128),128,0,d.stream>>>(count,d.reportPointers.ptr+begin,d.reports.ptr);
                    d.transfers.download(reports.data(),d.reports.ptr,count*sizeof(UpwindLinearReport));d.transfers.finish();bool active=false;
                    for(int angle=begin;angle<end;++angle)if(reports[angle-begin].active){auto& q=*d.directions[angle];q.graph->advance(q.stream);checked(cudaEventRecord(q.complete,q.stream));checked(cudaStreamWaitEvent(d.stream,q.complete,0));active=true;}
                    if(!active)break;
                }
                for(int angle=begin;angle<end;++angle)finish(angle,reports[angle-begin]);
                // Next batch reuses matrix/solution buffers on independent streams.
                checked(cudaStreamSynchronize(d.stream));
            }
        }else{
            for(int angle=0;angle<g.directions;++angle){prepare(angle,d.stream);finish(angle,d.directions[angle]->graph->launch(d.stream));}
        }
        d.maximum(g.directions,d.directionChange.ptr,nullptr,0,d.reductionA.ptr);
        d.transfers.download(&result.maximumRelativeChange,d.reductionA.ptr,sizeof(double));d.transfers.finish();result.sourceIterations=iteration;
        if(result.maximumRelativeChange<=d.controls.sourceTolerance)break;
    }
    d.reset();incidentFlux<<<boundaryBlocks,128,0,d.stream>>>(g,d.intensities.ptr,d.incident.ptr,d.errors());d.checkErrors();
    d.reset();moments<<<blocks,128,0,d.stream>>>(g,f,d.intensities.ptr,result.sourceIterations,d.incidentRadiation.ptr,d.power.ptr,d.errors());d.checkErrors();
    d.reset();boundaryResult<<<boundaryBlocks,128,0,d.stream>>>(g,f,d.intensities.ptr,d.incident.ptr,result.sourceIterations,d.wallFlux.ptr,d.errors());d.checkErrors();
    d.absorption.download(result.absorption,d.transfers);d.scattering.download(result.scattering,d.transfers);d.extinction.download(result.extinction,d.transfers);
    d.asymmetry.download(result.asymmetry,d.transfers);d.incidentRadiation.download(result.incident,d.transfers);d.power.download(result.power,d.transfers);
    d.wallFlux.download(result.boundaryFlux,d.transfers);d.phaseMetrics.download(result.phaseMetrics,d.transfers);d.transfers.finish();return result;
}
void DiscreteOrdinateRadiation::downloadIntensities(std::vector<double>& output)const{
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.intensities.download(output,d.transfers);d.transfers.finish();
}
void DiscreteOrdinateRadiation::uploadIntensities(const std::vector<double>& input){
    auto& d=*data_;if(input.size()!=d.intensities.size)throw std::invalid_argument("radiation warm-start shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfers);d.transfers.upload(d.intensities.ptr,input.data(),input.size()*sizeof(double));d.transfers.finish();
}
}}
