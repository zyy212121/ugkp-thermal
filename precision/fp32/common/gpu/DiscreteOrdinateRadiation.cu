#include "DiscreteOrdinateRadiation.H"
#include "RadiationPrecision.H"
#include "../Diagnostics.H"
#include "MiePhaseBalance.cuh"
#include "OrderedUpwindGraph.cuh"
#include "PinnedTransferQueue.cuh"
#include <algorithm>
#include <climits>
#include <cstring>
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
struct SourceReport {double maximumRelativeChange=0;int iterations=0;Failure failure{};};
// Keep convergence reports and sticky diagnostics beside the fields they
// describe. Publishing these diagnostics does not require a host decision.
__global__ void publishSourceDiagnostics(const SourceReport* report,
    const MiePhaseMetrics* phase,const RadiationDirectionWarning* direction,
    int cells,int angles,RadiationControls controls,diagnostics::DeviceContext* context){
    using namespace diagnostics;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0){
        const auto failure=report->failure;
        if(failure.code){
            auto record=makeRecord(isfinite(failure.value)?Code::InvalidPhysicalState:Code::NonFinite,
                34000,0,failure.iteration,failure.cell>=0?EntityKind::Cell:EntityKind::Face,
                failure.cell>=0?failure.cell:failure.face,failure.patch,failure.value);
            record.metadata=failure.code;recordFatal(context,record);
        }else if(report->maximumRelativeChange>controls.sourceTolerance){
            recordWarning(context,makeRecord(Code::AccuracyResidual,34001,0,report->iterations,
                EntityKind::None,-1,-1,report->maximumRelativeChange,controls.sourceTolerance,0,Quantity::Accuracy));
        }
    }
    if(i<cells&&(phase[i].rowError>controls.phaseTolerance||phase[i].columnError>controls.phaseTolerance)){
        recordWarning(context,makeRecord(Code::AccuracyResidual,34002,0,phase[i].iterations,
            EntityKind::Cell,i,-1,fmax(phase[i].rowError,phase[i].columnError),controls.phaseTolerance,0,Quantity::Accuracy));
    }
    if(i<angles&&direction[i].linearIteration>0){
        const auto warning=direction[i];
        recordWarning(context,makeRecord(Code::AccuracyResidual,34003,0,warning.sourceIteration,
            EntityKind::None,i,warning.linearIteration,warning.residual,warning.limit,0,Quantity::LinearSystem));
    }
}
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
__device__ double consumeOffset(double delta,double origin,double reference,int code,int iteration,int cell,int patch,int face,
    Errors errors,int slot,unsigned int priority){
    const double physical=origin+delta;
    const double accepted=consume(physical,reference,code,iteration,cell,patch,face,errors,slot,priority);
    return accepted==physical?delta:accepted-origin;
}
struct Geometry {
    int cells,directions,faces,boundary;
    const int *owner,*neighbour,*boundaryOwner,*boundaryPatch,*boundaryFace,*reflected,*boundaryOffsets,*boundaryCells;
    const int *reflectionOffsets,*reflectionDonors;
    const double* reflectionWeights;
    const RadiationBoundary* boundaryType;
    const double *volume,*weights,*internalFlux,*transportDiagonal,*boundaryFlux,*cosine,*area,*denominator,*fixed,*outgoingMeasure;
};
struct Fields {
    double *absorption,*scattering,*extinction,*asymmetry,*emission,*wallBlack,*scaling,*reference,*origin;
    MieOpticalQuery* query;
    MieOpticalTableView table;const MieBracket* phaseBrackets;
    const double* emissivity;
};
__global__ void resetErrors(Errors errors){if(!threadIdx.x)*errors.first=ULLONG_MAX;}
template<class Real> __global__ void importParticleState(int cells,RadiationParticleDeviceView input,
    const double* volume,double cutoff,MieOpticalTableView table,double* epsilon,double* mass,double* temperature,double* diameter){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells)return;
    const double m=static_cast<const Real*>(input.mass)[cell];mass[cell]=m;
    epsilon[cell]=m/(input.materialDensity*volume[cell]);temperature[cell]=input.minimumTemperature;diameter[cell]=0;
    if(m>0){temperature[cell]=double(static_cast<const Real*>(input.temperatureMass)[cell])/m;
        diameter[cell]=double(static_cast<const Real*>(input.diameterMass)[cell])/m;}
    if(epsilon[cell]>cutoff&&diameter[cell]>0&&std::isfinite(diameter[cell])){
        if(diameter[cell]<table.diameter[0])diameter[cell]=table.diameter[0];
        if(diameter[cell]>table.diameter[table.diameters-1])diameter[cell]=table.diameter[table.diameters-1];
    }
}
template<class Real> __global__ void importBoundaryState(int faces,RadiationParticleDeviceView input,
    const unsigned char* useTemperature,const unsigned char* contact,const double* area,double* temperature,double* emissivity){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=faces)return;const int full=input.boundaryOffset+face;
    if(useTemperature[face]){
        if(input.boundaryTemperatureFp32)temperature[face]=input.boundaryTemperatureFp32[full];
        else if(input.boundaryTemperature)temperature[face]=input.boundaryTemperature[full];
    }
    if(contact[face]&&input.boundaryEmissivity)emissivity[face]=input.boundaryEmissivity[full];
    if(contact[face]&&input.contactArea&&input.contactScale){
        // Match the prior host mirror's represented-area product in its native
        // storage precision, then the FP64 exposed-area correction.
        const double occupied=static_cast<const Real*>(input.contactArea)[full]*static_cast<const Real*>(input.contactScale)[full];
        const double covered=occupied/area[face];
        const double bounded=covered<1?covered:1;const double remaining=1-bounded;
        const double exposed=remaining>0?remaining:0;
        emissivity[face]*=exposed;
    }
}
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
// A binary32 anchor makes zero intensity exactly representable as -origin.
// Emission and all per-ordinate net differences retain binary64 arithmetic.
__global__ void chooseIntensityOrigin(const double* reference,double* origin){if(!threadIdx.x)*origin=double(RadiationReal(*reference));}
__global__ void fillInitial(std::size_t count,RadiationReal* values,const double* reference,const double* origin){const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)values[i]=RadiationReal(*reference-*origin);}
__global__ void rebaseIntensities(std::size_t count,RadiationReal* values,const double* oldOrigin,const double* origin){
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)values[i]=RadiationReal((*oldOrigin-*origin)+double(values[i]));
}
__device__ double incoming(Geometry g,Fields f,const RadiationReal* intensities,const double* incident,int direction,int face,
    Errors errors,int slot,unsigned priority){
    const int cell=g.boundaryOwner[face],patch=g.boundaryPatch[face],local=g.boundaryFace[face];
    switch(g.boundaryType[face]){
    case RadiationBoundary::Specular:{
        double value=0,weight=0;const int row=direction*g.boundary+face;
        for(int e=g.reflectionOffsets[row];e<g.reflectionOffsets[row+1];++e){
            value+=g.reflectionWeights[e]*consumeOffset(intensities[std::size_t(g.reflectionDonors[e])*g.cells+cell],*f.origin,
                *f.origin+double(intensities[std::size_t(direction)*g.cells+cell]),Specular,0,cell,patch,local,errors,slot,priority);
            weight+=g.reflectionWeights[e];
        }
        return value+*f.origin*(weight-1);
    }
    case RadiationBoundary::Zero:return -*f.origin;
    case RadiationBoundary::Fixed:return g.fixed[face]-*f.origin;
    case RadiationBoundary::Black:return f.wallBlack[face]-*f.origin;
    default:return f.emissivity[face]*(f.wallBlack[face]-*f.origin)+(1-f.emissivity[face])*
        (consumeOffset(incident[face],*f.origin*g.outgoingMeasure[face],incident[face]+*f.origin*g.outgoingMeasure[face],
            Diffuse,0,cell,patch,local,errors,slot,priority)+*f.origin*(g.outgoingMeasure[face]-g.denominator[face]))/g.denominator[face];
    }
}
__global__ void incidentFlux(Geometry g,Fields f,const RadiationReal* intensities,double* incident,Errors errors){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const int cell=g.boundaryOwner[face];double result=0;
    for(int direction=0;direction<g.directions;++direction){const double cosine=g.cosine[direction*g.boundary+face];if(cosine>0){
        const double value=*f.origin+double(intensities[std::size_t(direction)*g.cells+cell]);
        result+=g.weights[direction]*cosine*consumeOffset(intensities[std::size_t(direction)*g.cells+cell],*f.origin,value,Incident,0,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        if(!std::isfinite(value)||value < -upwindIntensityRoundoff*(fabs(value)>1?fabs(value):1))return;
    }}incident[face]=result;
}
__global__ void prepareDirection(Geometry g,Fields f,int direction,int iteration,const RadiationReal* intensities,
    const double* incident,RadiationReal* diagonal,RadiationReal* source,RadiationReal* solved,RadiationReal* solvedLow,
    double* trueDiagonal,double* trueSource,Errors errors,const SourceReport* progress=nullptr){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    trueDiagonal[cell]=g.transportDiagonal[std::size_t(direction)*g.cells+cell]+(f.absorption[cell]+f.scattering[cell])*g.volume[cell];
    diagonal[cell]=RadiationReal(trueDiagonal[cell]);
    double rhs=0;
    for(int i=g.boundaryOffsets[cell];i<g.boundaryOffsets[cell+1];++i){const int face=g.boundaryCells[i];const double flux=g.boundaryFlux[direction*g.boundary+face];
        if(flux<0){const double value=incoming(g,f,intensities,incident,direction,face,errors,cell,face);
            // A failed consumer is the first error for this cell; later work must
            // not overwrite its unique failure record before the stream completes.
            if(errors.records[cell].code)return;
            rhs-=flux*(*f.origin+value);}}
    double mean=0;
    if(f.scattering[cell]!=0){
        const auto d=mieLogBracket(f.table.diameter,f.table.diameters,f.query[cell].diameter);
        const auto t=mieLinearBracket(f.table.temperature,f.table.temperatures,f.query[cell].temperature);
        double deviation=0,probabilityWeight=0;
        for(int other=0;other<g.directions;++other){const double value=consumeOffset(intensities[std::size_t(other)*g.cells+cell],*f.origin,*f.reference,
            Scattering,iteration,cell,-1,-1,errors,cell,g.boundary+cell);
            if(errors.records[cell].code)return;
            const double probability=mieDetail::phaseProbability(f.table,d,t,g.weights,f.phaseBrackets,f.scaling+std::size_t(cell)*g.directions,g.directions,direction,other);
            deviation+=probability*value;probabilityWeight+=probability;}
        // Keep the phase row's actual closure defect, rather than assuming it
        // sums to one when reconstructing the physical FP64 scattering source.
        mean=*f.origin+deviation+*f.origin*(probabilityWeight-1);
    }
    // Preserve the original physical FP64 source for a true original-equation residual.
    trueSource[cell]=rhs+g.volume[cell]*(f.absorption[cell]*f.emission[cell]+f.scattering[cell]*mean);
    source[cell]=RadiationReal(trueSource[cell]);
    solved[cell]=intensities[std::size_t(direction)*g.cells+cell];solvedLow[cell]=0;
}
__global__ void relaxedDirection(int cells,int direction,double relaxation,const RadiationReal* old,RadiationReal* solved,const RadiationReal* solvedLow,RadiationReal* intensities,const double* origin,double* change,double* magnitude,const SourceReport* progress=nullptr){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells||(progress&&progress->failure.code))return;
    const std::size_t offset=std::size_t(direction)*cells+cell;const double before=old[offset];
    const RadiationReal after=RadiationReal((1-relaxation)*before+relaxation*(double(solved[cell])+double(solvedLow[cell])));
    change[cell]=fabs(double(after)-before);magnitude[cell]=fabs(*origin+double(after));solved[cell]=after;intensities[offset]=after;
}
__global__ void initializeSource(SourceReport* report){if(!threadIdx.x)*report={};}
__global__ void beginSourceIteration(SourceReport* report){if(!threadIdx.x)++report->iterations;}
__global__ void rememberFailure(SourceReport* report,Errors errors){
    if(!threadIdx.x&&!report->failure.code&&*errors.first!=ULLONG_MAX)
        report->failure=errors.records[unsigned(*errors.first)];
}
__global__ void stopFailedDirection(Errors errors,const SourceReport* report){
    if(!threadIdx.x&&report->failure.code){*errors.first=0;errors.records[0]=report->failure;}
}
__global__ void finishDirectionReport(int angle,const UpwindLinearReport* linear,Errors errors,
    SourceReport* source,RadiationDirectionWarning* warnings){
    if(threadIdx.x||source->failure.code)return;
    if(linear->externalFailure!=ULLONG_MAX){source->failure=errors.records[unsigned(linear->externalFailure)];return;}
    if(linear->errorCell>=0){source->failure={linear->errorValue,Linear,source->iterations,linear->errorCell,-1,-1};return;}
    if(linear->finalResidual>linear->target)
        warnings[angle]={linear->finalResidual,linear->target,linear->iterations,source->iterations};
    warnings[angle].publishedResidual=linear->finalResidual;warnings[angle].publishedTarget=linear->target;
}
__global__ void finishSourceIteration(SourceReport* report,const double* change,RadiationControls controls,
    cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;report->maximumRelativeChange=*change;
    cudaGraphSetConditional(handle,!report->failure.code&&report->iterations<(controls.sourceIterations>1?controls.sourceIterations:1)
        &&!(report->maximumRelativeChange<=controls.sourceTolerance));
}
__global__ void ratio(const double* a,const double* b,double* output){if(!threadIdx.x)*output=*a/ *b;}
__global__ void moments(Geometry g,Fields f,const RadiationReal* intensities,int iteration,double* incident,double* power,Errors errors,const SourceReport* progress=nullptr){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    double value=0,net=0,weight=0;
    for(int direction=0;direction<g.directions;++direction){const double delta=consumeOffset(intensities[std::size_t(direction)*g.cells+cell],*f.origin,
        *f.reference,Moment,iteration,cell,-1,-1,errors,cell,direction*g.cells+cell);if(errors.records[cell].code)return;
        value+=g.weights[direction]*delta;net+=g.weights[direction]*((*f.origin-f.emission[cell])+delta);weight+=g.weights[direction];}
    incident[cell]=*f.origin*weight+value;
    power[cell]=f.absorption[cell]==0?0:f.absorption[cell]*(net+(weight-fourPi)*f.emission[cell]);
}
__global__ void boundaryResult(Geometry g,Fields f,const RadiationReal* intensities,const double* incident,int iteration,double* output,Errors errors,const SourceReport* progress=nullptr){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    const int cell=g.boundaryOwner[face];double result=0,balance=0;
    for(int direction=0;direction<g.directions;++direction){const double cosine=g.cosine[direction*g.boundary+face];double value=0;
        if(cosine>0)value=consumeOffset(intensities[std::size_t(direction)*g.cells+cell],*f.origin,*f.reference,Outgoing,iteration,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        else if(cosine<0)value=incoming(g,f,intensities,incident,direction,face,errors,face,face);
        if(errors.records[face].code)return;result+=g.weights[direction]*cosine*value;balance+=g.weights[direction]*cosine;
    }output[face]=result+*f.origin*balance;
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
    Array<double> volume,weights,internalFlux,transportDiagonal,boundaryFlux,cosines,area,denominator,fixed,directionCosines,outgoingMeasure;
    Array<double> epsilon,mass,temperature,diameter,wallTemperature,emissivity;
    Array<unsigned char> residentWallTemperature,contactEmissivity;
    cudaEvent_t inputReady=nullptr;bool residentMass=false,resultReady=false;
    diagnostics::DeviceContext* diagnostics=nullptr;
    Array<double> absorption,scattering,extinction,asymmetry,emission,wallBlack,scaling,nextScaling,rowError,columnError,reference,origin,oldOrigin;
    Array<MieOpticalQuery> queries;Array<MieBracket> phaseBrackets;Array<MiePhaseMetrics> phaseMetrics;
    // Long-lived angular fields use one binary32 deviation each.
    Array<RadiationReal> intensities,frozen;
    Array<double> incident,initial,incidentRadiation,power,wallFlux,directionChange;
    // finish() is ordered on the owner stream; these reductions are shared scratch.
    Array<double> change,magnitude;
    Array<unsigned char> initialValid;
    Array<int> reflectionOffsets,reflectionDonors;Array<double> reflectionWeights;
    Array<OrderedMaximum> partialA,partialB;Array<double> reductionA,reductionB;
    Array<unsigned long long> firstError;Array<Failure> failures;
    Array<SourceReport> sourceReport;Array<RadiationDirectionWarning> warnings;
    cudaGraph_t sourceGraph=nullptr;cudaGraphExec_t sourceExecutable=nullptr;
    Geometry g{};Fields f{};
    struct DirectionWorkspace {
        // Refinement remainder is temporary per concurrent direction. source and
        // previous are reused for signed correction RHS and sweep increments.
        Array<RadiationReal> diagonal,source,solved,solvedLow,previous;
        Array<double> trueDiagonal,trueSource;
        Array<unsigned long long> firstError;Array<Failure> failures;
        cudaStream_t stream=nullptr;
        DirectionWorkspace(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
        ~DirectionWorkspace(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    };
    struct Direction {
        OrderedUpwindPlan plan;Array<int> offsets,neighbours,subtract,residualOrder,levels,fluxAddress;Array<RadiationReal> flux;
        OrderedUpwindDeviceT<RadiationReal> matrix{};
        std::shared_ptr<DirectionWorkspace> workspace;
        Array<RadiationReal> &diagonal,&source,&solved,&solvedLow,&previous;
        Array<double> &trueDiagonal,&trueSource;
        Array<unsigned long long>& firstError;Array<Failure>& failures;
        std::unique_ptr<OrderedUpwindGraphT<RadiationReal>> graph;
        cudaStream_t stream=nullptr;cudaEvent_t complete=nullptr;
        explicit Direction(std::shared_ptr<DirectionWorkspace> w):workspace(std::move(w)),
            diagonal(workspace->diagonal),source(workspace->source),solved(workspace->solved),solvedLow(workspace->solvedLow),previous(workspace->previous),
            trueDiagonal(workspace->trueDiagonal),trueSource(workspace->trueSource),
            firstError(workspace->firstError),failures(workspace->failures),stream(workspace->stream){
            checked(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming));}
        ~Direction(){if(stream)cudaStreamSynchronize(stream);graph.reset();if(complete)cudaEventDestroy(complete);}
        Errors errors(){return {firstError.ptr,failures.ptr};}
    };
    // Cached arrays are configuration inputs; the working wall arrays are
    // refreshed from these device copies before contact coverage is applied.
    Array<double> configuredWallTemperature,configuredEmissivity;
    std::vector<double> hostWallTemperature,hostEmissivity;
    std::vector<unsigned char> hostResidentWallTemperature,hostContactEmissivity;
    template<class T> void configure(Array<T>& target,std::vector<T>& previous,const std::vector<T>& next){
        if(target.size!=next.size()||previous.size()!=next.size()||
            (!next.empty()&&std::memcmp(previous.data(),next.data(),next.size()*sizeof(T)))){
            target.upload(next,transfers);previous=next;
        }
    }
    int concurrentAngles=1;
    std::vector<std::unique_ptr<Direction>> directions;
    Implementation(const RadiationGeometry& input,const MieOpticalTable& table,const RadiationControls& options):geometry(input),controls(options),optical(table){
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);
        checked(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));
    }
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);clearSourceGraph();directions.clear();cudaStreamDestroy(stream);}if(inputReady)cudaEventDestroy(inputReady);}
    void clearSourceGraph(){if(sourceExecutable)cudaGraphExecDestroy(sourceExecutable);if(sourceGraph)cudaGraphDestroy(sourceGraph);sourceExecutable=nullptr;sourceGraph=nullptr;}
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
        std::vector<double> outgoing(boundary,0);
        for(int face=0;face<boundary;++face)for(int angle=0;angle<angles;++angle)
            outgoing[face]+=geometry.weights[angle]*std::max(0.,geometry.boundaryCosine[angle*boundary+face]);
        outgoingMeasure.upload(outgoing,transfers);
        std::vector<std::vector<int>> perCell(cells);for(int face=0;face<boundary;++face)perCell[geometry.boundaryOwner[face]].push_back(face);
        std::vector<int> offsets(1,0),entries;for(const auto& row:perCell){entries.insert(entries.end(),row.begin(),row.end());offsets.push_back(entries.size());}
        boundaryOffsets.upload(offsets,transfers);boundaryCells.upload(entries,transfers);
        for(auto* field:{&absorption,&scattering,&extinction,&asymmetry,&emission,&incidentRadiation,&power})field->resize(cells);
        wallBlack.resize(boundary);wallFlux.resize(boundary);reference.resize(1);
        if(!origin.size){origin.resize(1);checked(cudaMemsetAsync(origin.ptr,0,sizeof(double),stream));}
        oldOrigin.resize(1);queries.resize(cells);phaseMetrics.resize(cells);phaseBrackets.resize(std::size_t(angles)*angles);
        for(auto* field:{&scaling,&nextScaling,&rowError,&columnError})field->resize(angular);
        intensities.resize(angular);frozen.resize(angular);
        change.resize(cells);magnitude.resize(cells);
        incident.resize(boundary);initial.resize(cells+boundary);initialValid.resize(cells+boundary);directionChange.resize(angles);
        reductionA.resize(1);reductionB.resize(1);firstError.resize(1);failures.resize(std::max({1,cells,boundary}));
        partialA.resize(std::max(1,(std::max(cells+boundary,angles)+255)/256));partialB.resize(partialA.size);
        epsilon.resize(cells);mass.resize(cells);temperature.resize(cells);diameter.resize(cells);wallTemperature.resize(boundary);emissivity.resize(boundary);
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
            boundaryFlux.ptr,cosines.ptr,area.ptr,denominator.ptr,fixed.ptr,outgoingMeasure.ptr};
        f={absorption.ptr,scattering.ptr,extinction.ptr,asymmetry.ptr,emission.ptr,wallBlack.ptr,scaling.ptr,reference.ptr,origin.ptr,queries.ptr,optical.deviceTable(),phaseBrackets.ptr,emissivity.ptr};
        mieDetail::phaseBrackets<<<std::max(1,(angles*angles+127)/128),128,0,stream>>>(angles*angles,f.table,directionCosines.ptr,phaseBrackets.ptr);
        transfers.finish();
        concurrentAngles=controls.parallelAngles?std::max(1,std::min(angles,controls.maximumConcurrentAngles>0?controls.maximumConcurrentAngles:angles)):1;
        std::vector<std::shared_ptr<DirectionWorkspace>> workspaces;
        for(int i=0;i<concurrentAngles;++i)workspaces.push_back(std::make_shared<DirectionWorkspace>());
        sourceReport.resize(1);warnings.resize(angles);
        for(int angle=0;angle<angles;++angle){
            std::unique_ptr<Direction> next(new Direction(workspaces[angle%concurrentAngles]));auto& d=*next;
            std::vector<double> flux(geometry.internalFlux.begin()+std::size_t(angle)*g.faces,geometry.internalFlux.begin()+std::size_t(angle+1)*g.faces);
            d.plan.build(cells,geometry.owner,geometry.neighbour,geometry.cellOffsets,geometry.cellFaces,flux);
            d.offsets.upload(d.plan.offsets,transfers);d.neighbours.upload(d.plan.neighbour,transfers);d.subtract.upload(d.plan.subtract,transfers);
            d.residualOrder.upload(d.plan.residualOrder,transfers);d.levels.upload(d.plan.levelCells,transfers);d.flux.upload(std::vector<RadiationReal>(d.plan.flux.begin(),d.plan.flux.end()),transfers);
            d.fluxAddress.upload(d.plan.face,transfers);
            for(auto* field:{&d.diagonal,&d.source,&d.solved,&d.solvedLow,&d.previous})field->resize(cells);
            d.trueDiagonal.resize(cells);d.trueSource.resize(cells);
            transfers.finish();
            d.firstError.resize(1);d.failures.resize(std::max(1,cells));
            d.matrix={cells,d.offsets.ptr,d.neighbours.ptr,d.subtract.ptr,d.residualOrder.ptr,d.levels.ptr,d.flux.ptr,
                internalFlux.ptr+std::size_t(angle)*g.faces,d.fluxAddress.ptr};
            d.graph.reset(new OrderedUpwindGraphT<RadiationReal>(d.plan,d.matrix,g.faces,owner.ptr,neighbour.ptr,internalFlux.ptr+std::size_t(angle)*g.faces,
                d.diagonal.ptr,d.source.ptr,d.solved.ptr,d.previous.ptr,reference.ptr,
                {controls.linearTolerance,controls.linearRelativeTolerance,controls.small,controls.linearIterations},d.firstError.ptr,controls.parallelAngles,false,d.solvedLow.ptr,d.trueDiagonal.ptr,d.trueSource.ptr,origin.ptr,d.source.ptr));
            directions.push_back(std::move(next));
        }
        transfers.finish();buildSourceGraph();
    }
    void buildSourceGraph(){
        clearSourceGraph();checked(cudaGraphCreate(&sourceGraph,0));
        using upwindGraphDetail::appendCaptured;
        const int blocks=std::max(1,(g.cells+127)/128),boundaryBlocks=std::max(1,(g.boundary+127)/128);
        const auto initialized=appendCaptured(sourceGraph,{},stream,[&]{
            initializeSource<<<1,1,0,stream>>>(sourceReport.ptr);
            if(warnings.size)checked(cudaMemsetAsync(warnings.ptr,0,warnings.size*sizeof(RadiationDirectionWarning),stream));
        });
        cudaGraphConditionalHandle handle;
        checked(cudaGraphConditionalHandleCreate(&handle,sourceGraph,1,cudaGraphCondAssignDefault));
        cudaGraph_t body=nullptr;
        const auto loop=upwindGraphDetail::addWhile(sourceGraph,handle,{initialized},body);
        const RadiationReal* angular=controls.parallelAngles?frozen.ptr:intensities.ptr;
        auto tail=appendCaptured(body,{},stream,[&]{
            beginSourceIteration<<<1,1,0,stream>>>(sourceReport.ptr);reset();
            if(controls.parallelAngles&&intensities.size)checked(cudaMemcpyAsync(frozen.ptr,intensities.ptr,intensities.size*sizeof(RadiationReal),cudaMemcpyDeviceToDevice,stream));
            incidentFlux<<<boundaryBlocks,128,0,stream>>>(g,f,angular,incident.ptr,errors());
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
        });
        for(int begin=0;begin<g.directions;begin+=concurrentAngles){
            const int end=std::min(g.directions,begin+concurrentAngles);
            std::vector<cudaGraphNode_t> complete;
            for(int angle=begin;angle<end;++angle){auto& q=*directions[angle];
                const auto prepared=appendCaptured(body,{tail},stream,[&]{reset(q,stream);
                    stopFailedDirection<<<1,1,0,stream>>>(q.errors(),sourceReport.ptr);
                    prepareDirection<<<blocks,128,0,stream>>>(g,f,angle,0,angular,incident.ptr,q.diagonal.ptr,q.source.ptr,q.solved.ptr,q.solvedLow.ptr,q.trueDiagonal.ptr,q.trueSource.ptr,q.errors(),sourceReport.ptr);
                });
                complete.push_back(q.graph->appendSolve(body,{prepared},stream));
            }
            // Reports and publications remain in angular order. The next batch
            // waits for publication before reusing its direction workspaces.
            tail=appendCaptured(body,complete,stream,[&]{
                for(int angle=begin;angle<end;++angle){auto& q=*directions[angle];
                    finishDirectionReport<<<1,1,0,stream>>>(angle,q.graph->report(),q.errors(),sourceReport.ptr,warnings.ptr);
                    relaxedDirection<<<blocks,128,0,stream>>>(g.cells,angle,controls.sourceRelaxation,angular,q.solved.ptr,q.solvedLow.ptr,intensities.ptr,origin.ptr,change.ptr,magnitude.ptr,sourceReport.ptr);
                    q.graph->measurePublished(q.matrix,owner.ptr,neighbour.ptr,internalFlux.ptr+std::size_t(angle)*g.faces,
                        q.diagonal.ptr,q.source.ptr,q.solved.ptr,{controls.linearTolerance,controls.linearRelativeTolerance,controls.small,controls.linearIterations},
                        stream,q.trueDiagonal.ptr,q.trueSource.ptr,origin.ptr);
                    finishDirectionReport<<<1,1,0,stream>>>(angle,q.graph->report(),q.errors(),sourceReport.ptr,warnings.ptr);
                    maximum(g.cells,change.ptr,nullptr,0,reductionA.ptr);maximum(g.cells,magnitude.ptr,nullptr,controls.small,reductionB.ptr);
                    ratio<<<1,1,0,stream>>>(reductionA.ptr,reductionB.ptr,directionChange.ptr+angle);
                }
            });
        }
        appendCaptured(body,{tail},stream,[&]{
            maximum(g.directions,directionChange.ptr,nullptr,0,reductionA.ptr);
            finishSourceIteration<<<1,1,0,stream>>>(sourceReport.ptr,reductionA.ptr,controls,handle);
        });
        appendCaptured(sourceGraph,{loop},stream,[&]{
            reset();incidentFlux<<<boundaryBlocks,128,0,stream>>>(g,f,intensities.ptr,incident.ptr,errors());
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            reset();moments<<<blocks,128,0,stream>>>(g,f,intensities.ptr,0,incidentRadiation.ptr,power.ptr,errors(),sourceReport.ptr);
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            reset();boundaryResult<<<boundaryBlocks,128,0,stream>>>(g,f,intensities.ptr,incident.ptr,0,wallFlux.ptr,errors(),sourceReport.ptr);
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
        });
        checked(cudaGraphInstantiateWithFlags(&sourceExecutable,sourceGraph,0));
    }
};
DiscreteOrdinateRadiation::DiscreteOrdinateRadiation(const RadiationGeometry& geometry,const MieOpticalTable& table,const RadiationControls& controls):
    data_(new Implementation(geometry,table,controls)){data_->initialize();}
DiscreteOrdinateRadiation::~DiscreteOrdinateRadiation()=default;
void DiscreteOrdinateRadiation::updateGeometry(const RadiationGeometry& geometry){
    auto& d=*data_;const auto& old=d.geometry;
    if(geometry.cells!=old.cells||geometry.directions!=old.directions
        ||geometry.owner!=old.owner||geometry.neighbour!=old.neighbour
        ||geometry.cellOffsets!=old.cellOffsets||geometry.cellFaces!=old.cellFaces
        ||geometry.boundaryOwner!=old.boundaryOwner||geometry.boundaryPatch!=old.boundaryPatch
        ||geometry.boundaryFace!=old.boundaryFace||geometry.weights!=old.weights
        ||geometry.directionCosines!=old.directionCosines)
        throw std::invalid_argument("radiation metric update requires unchanged topology and angular quadrature");
    d.transfers.finish();checked(cudaStreamSynchronize(d.stream));
    // Direction graphs capture metric addresses and upwind order. Rebuild them
    // even when a face changes its sign relative to an ordinate.
    d.resultReady=false;d.clearSourceGraph();d.directions.clear();d.geometry=geometry;d.initialize();
}

RadiationResult DiscreteOrdinateRadiation::solve(const RadiationSnapshot& snapshot,bool warmStart,bool exportFields){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.resultReady=false;
    const auto g=d.g;const auto f=d.f;const int blocks=std::max(1,(g.cells+127)/128),boundaryBlocks=std::max(1,(g.boundary+127)/128);
    if(snapshot.wallTemperature.size()!=std::size_t(g.boundary)||snapshot.emissivity.size()!=std::size_t(g.boundary))
        throw std::invalid_argument("radiation wall input shape mismatch");
    auto wallTemperature=snapshot.wallTemperature;auto emissivity=snapshot.emissivity;
    if(snapshot.resident.enabled&&snapshot.residentWallTemperature.size()==wallTemperature.size()
        &&snapshot.contactEmissivity.size()==emissivity.size()){
        for(int f=0;f<g.boundary;++f){
            if((snapshot.resident.boundaryTemperature||snapshot.resident.boundaryTemperatureFp32)&&snapshot.residentWallTemperature[f])wallTemperature[f]=0;
            if(snapshot.resident.boundaryEmissivity&&snapshot.contactEmissivity[f])emissivity[f]=0;
        }
    }
    d.configure(d.configuredWallTemperature,d.hostWallTemperature,wallTemperature);
    d.configure(d.configuredEmissivity,d.hostEmissivity,emissivity);
    if(g.boundary){
        checked(cudaMemcpyAsync(d.wallTemperature.ptr,d.configuredWallTemperature.ptr,g.boundary*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
        checked(cudaMemcpyAsync(d.emissivity.ptr,d.configuredEmissivity.ptr,g.boundary*sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
    }
    d.residentMass=snapshot.resident.enabled;
    d.diagnostics=snapshot.resident.enabled?snapshot.resident.diagnostics:nullptr;
    if(snapshot.resident.enabled){
        const auto& input=snapshot.resident;
        if(input.cells!=g.cells||input.boundaryFaces!=g.boundary||(g.cells&&(!input.mass||!input.temperatureMass||!input.diameterMass))
            ||snapshot.residentWallTemperature.size()!=std::size_t(g.boundary)||snapshot.contactEmissivity.size()!=std::size_t(g.boundary))
            throw std::invalid_argument("resident radiation snapshot shape mismatch");
        d.configure(d.residentWallTemperature,d.hostResidentWallTemperature,snapshot.residentWallTemperature);
        d.configure(d.contactEmissivity,d.hostContactEmissivity,snapshot.contactEmissivity);
        checked(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(input.producerStream)));checked(cudaStreamWaitEvent(d.stream,d.inputReady,0));
        importParticleState<double><<<blocks,128,0,d.stream>>>(g.cells,input,g.volume,d.controls.activeEpsilonMinimum,f.table,d.epsilon.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr);
        importBoundaryState<double><<<boundaryBlocks,128,0,d.stream>>>(g.boundary,input,d.residentWallTemperature.ptr,d.contactEmissivity.ptr,g.area,d.wallTemperature.ptr,d.emissivity.ptr);
        checked(cudaEventRecord(d.inputReady,d.stream));
        checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(input.producerStream),d.inputReady,0));
    }else{
        d.epsilon.upload(snapshot.epsilon,d.transfers);d.temperature.upload(snapshot.temperature,d.transfers);d.diameter.upload(snapshot.diameter,d.transfers);
    }
    optics<<<blocks,128,0,d.stream>>>(g,f,d.epsilon.ptr,d.temperature.ptr,d.diameter.ptr,d.controls.activeEpsilonMinimum,d.initial.ptr,d.initialValid.ptr);
    wallOptics<<<boundaryBlocks,128,0,d.stream>>>(g,f,d.wallTemperature.ptr,d.initial.ptr,d.initialValid.ptr);
    checked(cudaMemcpyAsync(d.oldOrigin.ptr,d.origin.ptr,sizeof(double),cudaMemcpyDeviceToDevice,d.stream));
    d.maximum(g.cells+g.boundary,d.initial.ptr,d.initialValid.ptr,0,d.reference.ptr);
    chooseIntensityOrigin<<<1,1,0,d.stream>>>(d.reference.ptr,d.origin.ptr);
    if(g.cells)mieDetail::balancePhase<<<g.cells,128,0,d.stream>>>(g.cells,g.directions,f.table,d.queries.ptr,d.weights.ptr,d.phaseBrackets.ptr,
        d.controls.phaseTolerance,d.controls.phaseIterations,d.scaling.ptr,d.nextScaling.ptr,d.rowError.ptr,d.columnError.ptr,d.phaseMetrics.ptr,d.scattering.ptr);
    if(!warmStart)fillInitial<<<std::max(std::size_t(1),(d.intensities.size+127)/128),128,0,d.stream>>>(d.intensities.size,d.intensities.ptr,d.reference.ptr,d.origin.ptr);
    if(warmStart)rebaseIntensities<<<std::max(std::size_t(1),(d.intensities.size+127)/128),128,0,d.stream>>>(d.intensities.size,d.intensities.ptr,d.oldOrigin.ptr,d.origin.ptr);
    checked(cudaGraphLaunch(d.sourceExecutable,d.stream));
    d.resultReady=true;
    if(!exportFields&&d.diagnostics){
        publishSourceDiagnostics<<<std::max(1,(std::max(g.cells,g.directions)+127)/128),128,0,d.stream>>>(
            d.sourceReport.ptr,d.phaseMetrics.ptr,d.warnings.ptr,g.cells,g.directions,d.controls,d.diagnostics);
        checked(cudaGetLastError());
        return {};
    }
    return collect(exportFields);
}
RadiationResult DiscreteOrdinateRadiation::collect(bool exportFields){
    auto& d=*data_;if(!d.resultReady)throw std::logic_error("radiation output requested before solve");
    PinnedTransferQueue::Operation operation(d.transfers);
    RadiationResult result;SourceReport report;
    d.transfers.download(&report,d.sourceReport.ptr,sizeof(report));
    if(exportFields){
        d.warnings.download(result.directionWarnings,d.transfers);
        d.absorption.download(result.absorption,d.transfers);d.scattering.download(result.scattering,d.transfers);d.extinction.download(result.extinction,d.transfers);
        d.asymmetry.download(result.asymmetry,d.transfers);d.incidentRadiation.download(result.incident,d.transfers);d.power.download(result.power,d.transfers);
        d.wallFlux.download(result.boundaryFlux,d.transfers);d.phaseMetrics.download(result.phaseMetrics,d.transfers);
    }
    d.transfers.finish();
    if(report.failure.code){d.resultReady=false;fail(report.failure);}
    result.sourceIterations=report.iterations;result.maximumRelativeChange=report.maximumRelativeChange;d.resultReady=true;return result;
}
RadiationDeviceView DiscreteOrdinateRadiation::deviceView()const noexcept{
    const auto& d=*data_;return {d.resultReady,d.g.cells,d.g.boundary,d.epsilon.ptr,d.residentMass?d.mass.ptr:nullptr,d.temperature.ptr,d.diameter.ptr,
        d.incidentRadiation.ptr,d.power.ptr,d.volume.ptr,d.area.ptr,d.wallFlux.ptr,d.stream,d.diagnostics};
}
void DiscreteOrdinateRadiation::downloadIntensities(std::vector<double>& output)const{
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    std::vector<RadiationReal> values;double origin=0;d.intensities.download(values,d.transfers);
    d.transfers.download(&origin,d.origin.ptr,sizeof(origin));d.transfers.finish();
    output.resize(values.size());for(std::size_t i=0;i<values.size();++i)output[i]=origin+double(values[i]);
}
void DiscreteOrdinateRadiation::uploadIntensities(const std::vector<double>& input){
    auto& d=*data_;if(input.size()!=d.intensities.size)throw std::invalid_argument("radiation warm-start shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfers);double origin=0;
    d.transfers.download(&origin,d.origin.ptr,sizeof(origin));d.transfers.finish();
    std::vector<RadiationReal> values(input.size());for(std::size_t i=0;i<input.size();++i)values[i]=RadiationReal(input[i]-origin);
    d.intensities.upload(values,d.transfers);d.transfers.finish();
}
}}
