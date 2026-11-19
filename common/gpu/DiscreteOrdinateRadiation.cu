#include <cstdio>
#include "WorkPlan.H"
#include "DiscreteOrdinateRadiation.H"
#include "RadiationWarmStart.cuh"
#include "RadiationGeometryUpdate.H"
#include "../Diagnostics.H"
#include "../DiagnosticsHost.H"
#include "MiePhaseBalance.cuh"
#include "OrderedUpwindGraph.cuh"
#include "SparseConjugateGradient.H"
#include "RadiationFgmres.cuh"
#include "PinnedTransferQueue.cuh"
#include "GraphContinuationTask.cuh"
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
    template<class U> void upload(const std::vector<U>& x,PinnedTransferQueue& queue){
        resize(x.size());const std::vector<T> converted(x.begin(),x.end());
        queue.upload(ptr,converted.data(),size*sizeof(T));
    }
    void download(std::vector<T>& x,PinnedTransferQueue& queue){x.resize(size);queue.download(x.data(),ptr,size*sizeof(T));}
};
constexpr RadiationReal fourPi=RadiationReal(12.56637061435917295385057353311801153679);
__device__ RadiationReal blackIntensity(RadiationReal t,RadiationReal band){const RadiationReal square=t*t;return band*RadiationReal(5.670374419e-8)*square*square/(fourPi/4);}
enum Location {Incident=1,Specular,Diffuse,Scattering,Linear,Moment,Outgoing,Dependency};
struct Failure {RadiationReal value;int code,iteration,cell,component,face;};
struct SourceReport {RadiationReal maximumRelativeChange=0,materialRelativeResidual=0,transportRelativeResidual=0,materialResidualTarget=0,transportResidualTarget=0;int iterations=0,preconditionerFallbackBlocks=0;unsigned long long linearSolves=0,totalLinearIterations=0;int lastLinearIterations=0,lastLinearStopReason=0;RadiationReal lastLinearResidual=0;bool lastLinearConverged=false,intensityUnchanged=false,correctPublishedResidual=false;
    unsigned long long matvecs=0,workLimit=0,remainingWork=0;
    int sourceStop=0, plateauSamples=0;
    RadiationReal plateauReference=0;
    double wideResidual=0, arithmeticDelta=0, minimumUnconstrainedIntensity=0;
    Failure failure{};};
__global__ void publishSourcePreconditionerWarning(const SourceReport* report,diagnostics::DeviceContext* context){
    if(threadIdx.x||blockIdx.x||!report->preconditionerFallbackBlocks)return;
    using namespace diagnostics;
    recordWarning(context,makeRecord(Code::MethodFallback,36010,0,report->iterations,
        EntityKind::None,-1,-1,report->preconditionerFallbackBlocks,0,0,Quantity::LinearSystem));
}


__global__ void publishSourceDiagnostics(const SourceReport* report,
    const MiePhaseMetrics* phase,const RadiationDirectionWarning* direction,
    int cells,int angles,RadiationControls controls,const RadiationReal* intensityReference,diagnostics::DeviceContext* context){
    using namespace diagnostics;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i==0){
        const auto failure=report->failure;
        if(failure.code){
            auto record=makeRecord(isfinite(failure.value)?Code::InvalidPhysicalState:Code::NonFinite,
                34000,0,failure.iteration,failure.cell>=0?EntityKind::Cell:EntityKind::Face,
                failure.cell>=0?failure.cell:failure.face,failure.component,failure.value);
            if(failure.code==Moment){
                record.reference=minimumRoundoffIntensity(*intensityReference);
                record.quantityId=Quantity::RadiationIntensity;
            }
            record.metadata=failure.code;recordFatal(context,record);
        }else if(!(report->maximumRelativeChange<=controls.sourceTolerance)){
            recordWarning(context,makeRecord(Code::AccuracyResidual,34001,0,report->iterations,
                EntityKind::None,-1,-1,report->maximumRelativeChange,controls.sourceTolerance,0,Quantity::Accuracy));
        }
    }
    if(i==0&&!report->failure.code){
        const auto roundoffFloor=minimumRoundoffIntensity(*intensityReference);
        if(report->minimumUnconstrainedIntensity<roundoffFloor)
            recordWarning(context,makeRecord(Code::AccuracyResidual,34004,0,report->iterations,
                EntityKind::None,-1,-1,report->minimumUnconstrainedIntensity,roundoffFloor,0,Quantity::RadiationIntensity));
        if(!(report->materialRelativeResidual<=report->materialResidualTarget))
            recordWarning(context,makeRecord(Code::AccuracyResidual,34001,0,report->iterations,
                EntityKind::None,-1,1,report->materialRelativeResidual,report->materialResidualTarget,0,Quantity::Accuracy));
        if(!(report->transportRelativeResidual<=report->transportResidualTarget))
            recordWarning(context,makeRecord(Code::AccuracyResidual,34001,0,report->iterations,
                EntityKind::None,-1,2,report->transportRelativeResidual,report->transportResidualTarget,0,Quantity::Accuracy));
    }
    if(i<cells&&(phase[i].rowError>controls.phaseTolerance||phase[i].columnError>controls.phaseTolerance)){
        recordWarning(context,makeRecord(Code::AccuracyResidual,34002,0,phase[i].iterations,
            EntityKind::Cell,i,-1,fmax(phase[i].rowError,phase[i].columnError),controls.phaseTolerance,0,Quantity::Accuracy));
    }
    if(i<angles&&(direction[i].linearIteration>0||direction[i].stopReason)){
        const auto warning=direction[i];
        auto record=makeRecord(Code::AccuracyResidual,34003,0,warning.sourceIteration,
            EntityKind::None,i,warning.linearIteration,warning.residual,warning.limit,0,Quantity::LinearSystem);
        record.metadata=warning.stopReason;recordWarning(context,record);
    }
}
__device__ inline void retainDependencyFailure(SourceReport* report,const diagnostics::DeviceContext* context){
    if(report->failure.code||!diagnostics::hasFatal(context))return;
    const auto failure=context->firstFatal;
    const int entity=failure.entityIndex>=0&&failure.entityIndex<=INT_MAX?int(failure.entityIndex):-1;
    report->failure={RadiationReal(failure.observed),Dependency,report->iterations,
        failure.entityKind==diagnostics::EntityKind::Cell?entity:-1,-1,
        failure.entityKind==diagnostics::EntityKind::Face?entity:-1};
}
struct Errors {unsigned long long* first;Failure* records;};
__device__ RadiationReal consume(RadiationReal value,RadiationReal reference,int code,int iteration,int cell,int patch,int face,
    Errors errors,int slot,unsigned int priority){
    if(std::isfinite(value)){
        if(value>=0)return value;
        if(value>=minimumRoundoffIntensity(reference))return 0;
    }
    errors.records[slot]={value,code,iteration,cell,patch,face};
    atomicMin(errors.first,(static_cast<unsigned long long>(priority)<<32)|unsigned(slot));
    return value;
}
struct Geometry {
    int cells,directions,faces,boundary;
    const int *owner,*neighbour,*boundaryOwner,*boundaryPatch,*boundaryFace,*reflected,*boundaryOffsets,*boundaryCells;
    const int *reflectionOffsets,*reflectionDonors;
    const RadiationReal* reflectionWeights;
    const RadiationBoundary* boundaryType;
    const RadiationReal *volume,*weights,*internalFlux,*transportDiagonal,*boundaryFlux,*cosine,*area,*denominator,*fixed;
};
struct Fields {
    RadiationReal *absorption,*scattering,*extinction,*asymmetry,*emission,*wallBlack,*scaling,*reference;
    MieOpticalQuery* query;
    MieOpticalTableView table;const MieBracket* phaseBrackets;
    const RadiationReal* emissivity;
    const RadiationThermalCouplingDeviceView* thermal=nullptr;
};
#include "RadiationTransportKernels.cuh"
__global__ void domLinearWarning(int angles,const RadiationFgmres::Report* report,SourceReport* source,
    RadiationDirectionWarning* warnings,RadiationControls controls){
    const int angle=blockIdx.x*blockDim.x+threadIdx.x;if(angle>=angles||source->failure.code)return;


    if(angle==0){++source->linearSolves;source->totalLinearIterations+=report->iterations;
        source->minimumUnconstrainedIntensity=fmin(source->minimumUnconstrainedIntensity,
            -__longlong_as_double(static_cast<long long>(report->negativeIntensityBits)));
        source->matvecs+=2ULL*report->iterations+report->residualChecks;
        source->lastLinearIterations=report->iterations;source->lastLinearStopReason=int(report->stopReason);
        source->lastLinearResidual=report->finalResidual;source->lastLinearConverged=report->converged;}
    const RadiationReal target=controls.linearTolerance;
    if(!report->converged){
        warnings[angle]={report->finalResidual,target,report->iterations,source->iterations};
        warnings[angle].stopReason=int(report->stopReason);
        warnings[angle].linearRelativeResidual=report->relativeResidual;warnings[angle].linearRelativeLimit=controls.linearRelativeTolerance;
    }
}
__global__ void resetErrors(Errors errors){if(!threadIdx.x)*errors.first=ULLONG_MAX;}
template<class Real> __global__ void importParticleState(int cells,RadiationParticleDeviceView input,
    const RadiationReal* volume,RadiationReal cutoff,MieOpticalTableView table,RadiationReal* epsilon,RadiationReal* mass,RadiationReal* temperature,RadiationReal* diameter){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=cells)return;
    const RadiationReal m=static_cast<const Real*>(input.mass)[cell];mass[cell]=m;
    epsilon[cell]=m/(input.materialDensity*volume[cell]);temperature[cell]=input.minimumTemperature;diameter[cell]=0;
    if(m>0){temperature[cell]=RadiationReal(static_cast<const Real*>(input.temperatureMass)[cell])/m;
        diameter[cell]=RadiationReal(static_cast<const Real*>(input.diameterMass)[cell])/m;}
    if(epsilon[cell]>cutoff&&diameter[cell]>0&&std::isfinite(diameter[cell])){
        if(diameter[cell]<table.diameter[0])diameter[cell]=table.diameter[0];
        if(diameter[cell]>table.diameter[table.diameters-1])diameter[cell]=table.diameter[table.diameters-1];
    }
}
template<class Real> __global__ void importBoundaryState(int faces,RadiationParticleDeviceView input,
    const unsigned char* useTemperature,const unsigned char* contact,const RadiationReal* area,RadiationReal* temperature,RadiationReal* emissivity){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=faces)return;const int full=input.boundaryOffset+face;
    if(useTemperature[face]&&input.boundaryTemperature)temperature[face]=static_cast<const Real*>(input.boundaryTemperature)[full];
    if(contact[face]&&input.boundaryEmissivity)emissivity[face]=input.boundaryEmissivity[full];
    if(contact[face]&&input.contactArea&&input.contactScale){


        const RadiationReal occupied=static_cast<const Real*>(input.contactArea)[full]*static_cast<const Real*>(input.contactScale)[full];
        const RadiationReal covered=occupied/area[face];
        const RadiationReal bounded=covered<1?covered:1;const RadiationReal remaining=1-bounded;
        const RadiationReal exposed=remaining>0?remaining:0;
        emissivity[face]*=exposed;
    }
}
__global__ void optics(Geometry g,Fields f,const RadiationReal* epsilon,const RadiationReal* temperature,const RadiationReal* diameter,
    RadiationReal cutoff,RadiationReal* initial,unsigned char* valid){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    if(f.thermal&&f.thermal->trialMeanTemperature)temperature=f.thermal->trialMeanTemperature;
    f.absorption[cell]=f.scattering[cell]=f.extinction[cell]=f.asymmetry[cell]=f.emission[cell]=0;
    f.query[cell]={diameter[cell],temperature[cell],0};initial[cell]=0;valid[cell]=0;
    if(epsilon[cell]<=cutoff)return;
    const auto optical=mieOpticalProperties(f.table,f.query[cell]);
    f.absorption[cell]=3*epsilon[cell]*optical.absorption/(2*diameter[cell]);
    f.scattering[cell]=3*epsilon[cell]*optical.scattering/(2*diameter[cell]);
    f.extinction[cell]=f.absorption[cell]+f.scattering[cell];f.asymmetry[cell]=optical.asymmetry;
    initial[cell]=f.emission[cell]=blackIntensity(temperature[cell],optical.band);valid[cell]=1;
}
__global__ void wallOptics(Geometry g,Fields f,const RadiationReal* temperature,RadiationReal* initial,unsigned char* valid){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const auto type=g.boundaryType[face];valid[g.cells+face]=0;initial[g.cells+face]=0;f.wallBlack[face]=0;
    if(type==RadiationBoundary::Diffuse||type==RadiationBoundary::Black){
        f.wallBlack[face]=blackIntensity(temperature[face],mieBand(f.table,temperature[face]));
        initial[g.cells+face]=f.wallBlack[face];valid[g.cells+face]=1;
    }else if(type==RadiationBoundary::Fixed){initial[g.cells+face]=g.fixed[face];valid[g.cells+face]=1;}
}
__global__ void fillInitial(std::size_t count,RadiationReal* values,const RadiationReal* reference){const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<count)values[i]=*reference;}
__device__ RadiationReal incoming(Geometry g,Fields f,const RadiationReal* intensities,const RadiationReal* incident,int direction,int face,
    Errors errors,int slot,unsigned priority){
    const int cell=g.boundaryOwner[face],patch=g.boundaryPatch[face],local=g.boundaryFace[face];
    switch(g.boundaryType[face]){
    case RadiationBoundary::Specular:{
        RadiationReal value=0;const int row=direction*g.boundary+face;
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
__global__ void incidentFlux(Geometry g,const RadiationReal* intensities,RadiationReal* incident,Errors errors){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    const int cell=g.boundaryOwner[face];RadiationReal result=0;
    for(int direction=0;direction<g.directions;++direction){const RadiationReal cosine=g.cosine[direction*g.boundary+face];if(cosine>0){
        const RadiationReal value=intensities[std::size_t(direction)*g.cells+cell];
        result+=g.weights[direction]*cosine*consume(value,value,Incident,0,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        if(!std::isfinite(value)||value < minimumRoundoffIntensity(value))return;
    }}incident[face]=result;
}
__global__ void prepareDirection(Geometry g,Fields f,int direction,int iteration,const RadiationReal* intensities,
    const RadiationReal* incident,RadiationReal* diagonal,RadiationReal* source,RadiationReal* solved,Errors errors,const SourceReport* progress=nullptr,bool physicalSource=false){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    diagonal[cell]=g.transportDiagonal[std::size_t(direction)*g.cells+cell]+(f.absorption[cell]+f.scattering[cell])*g.volume[cell];
    RadiationReal rhs=0;
    for(int i=g.boundaryOffsets[cell];i<g.boundaryOffsets[cell+1];++i){const int face=g.boundaryCells[i];const RadiationReal flux=g.boundaryFlux[direction*g.boundary+face];
        if(flux<0){const RadiationReal value=incoming(g,f,intensities,incident,direction,face,errors,cell,face);


            if(errors.records[cell].code)return;
            rhs-=flux*value;}}
    RadiationReal mean=0;
    if(f.scattering[cell]!=0){
        const auto d=mieLogBracket(f.table.diameter,f.table.diameters,f.query[cell].diameter);
        const auto t=mieLinearBracket(f.table.temperature,f.table.temperatures,f.query[cell].temperature);
        for(int other=0;other<g.directions;++other){const RadiationReal value=consume(intensities[std::size_t(other)*g.cells+cell],*f.reference,
            Scattering,iteration,cell,-1,-1,errors,cell,g.boundary+cell);
            if(errors.records[cell].code)return;
            mean+=mieDetail::phaseProbability(f.table,d,t,g.weights,f.phaseBrackets,f.scaling+std::size_t(cell)*g.directions,g.directions,direction,other)*value;}
    }
    RadiationReal emission=f.emission[cell];
    if(!physicalSource&&f.thermal&&f.thermal->linearizedEmission&&f.thermal->thermalFeedback){
        RadiationReal G=0;
        for(int other=0;other<g.directions;++other)
            G+=g.weights[other]*consume(intensities[std::size_t(other)*g.cells+cell],*f.reference,
                Scattering,iteration,cell,-1,-1,errors,cell,g.boundary+cell);
        if(errors.records[cell].code)return;

        emission=f.thermal->linearizedEmission[cell]+f.thermal->thermalFeedback[cell]*G/fourPi;
    }
    source[cell]=rhs+g.volume[cell]*(f.absorption[cell]*emission+f.scattering[cell]*mean);
    solved[cell]=intensities[std::size_t(direction)*g.cells+cell];
}
struct DirectionChange {
    RadiationMaximum difference{},magnitude{};
};
__device__ inline DirectionChange joinMaximum(DirectionChange left,DirectionChange right){
    return {ugkp::gpu::joinMaximum(left.difference,right.difference),
        ugkp::gpu::joinMaximum(left.magnitude,right.magnitude)};
}
__device__ inline void storeDirectionChange(DirectionChange local,DirectionChange* shared,DirectionChange* partial,
    RadiationReal* result,unsigned char* changed,RadiationReal small){
    shared[threadIdx.x]=local;
    reduceOrderedMaximumBlock(shared);
    if(!threadIdx.x){
        if(gridDim.x==1){
            const auto difference=applyMaximum(0,shared[0].difference);
            result[blockIdx.y]=difference/applyMaximum(small,shared[0].magnitude);
            changed[blockIdx.y]=difference!=0;
        }else partial[std::size_t(blockIdx.y)*gridDim.x+blockIdx.x]=shared[0];
    }
}
__global__ void relaxDirections(int cells,RadiationReal relaxation,const RadiationReal* old,RadiationReal* intensities,
    RadiationReal small,const SourceReport* progress,DirectionChange* partial,RadiationReal* result,unsigned char* changed){
    extern __shared__ __align__(16) unsigned char storage[];
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    DirectionChange value;
    if(cell<cells&&!progress->failure.code){
        const auto index=std::size_t(blockIdx.y)*cells+cell;
        const RadiationReal before=old[index],solved=intensities[index];
        const RadiationReal after=solved==before?before:(1-relaxation)*before+relaxation*solved;
        intensities[index]=after;
        value={maximumSegment(RadiationReal(fabs(after-before))),maximumSegment(RadiationReal(fabs(after)))};
    }
    storeDirectionChange(value,reinterpret_cast<DirectionChange*>(storage),partial,result,changed,small);
}
__global__ void reduceDirectionChanges(int count,const DirectionChange* input,DirectionChange* partial,
    RadiationReal* result,unsigned char* changed,RadiationReal small){
    extern __shared__ __align__(16) unsigned char storage[];
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    const auto value=i<count?input[std::size_t(blockIdx.y)*count+i]:DirectionChange{};
    storeDirectionChange(value,reinterpret_cast<DirectionChange*>(storage),partial,result,changed,small);
}
__global__ void fillParentInitial(std::size_t count,RadiationReal* values,const RadiationReal* reference,const int* warm){
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count&&!*warm)values[i]=*reference;
}
__global__ void acceptParentWarmStart(const SourceReport* report,int* warm){
    if(!threadIdx.x&&!report->failure.code)*warm=1;
}
__global__ void initializeSource(SourceReport* report,RadiationControls controls){
    if(threadIdx.x)return;
    *report={};
    const auto maximumSolve=1+4ULL*max(1,controls.linearIterations);
    report->workLimit=maximumSolve*static_cast<unsigned long long>(ceil(sqrt(double(max(1,controls.sourceIterations)))));
    report->remainingWork=report->workLimit;
}
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
}
__global__ void recordPublishedDirection(int angle,const UpwindLinearReport* linear,RadiationReal target,Errors errors,
    RadiationDirectionWarning* warnings,Failure* failures){
    if(threadIdx.x)return;
    const auto first=*errors.first;
    failures[angle]=first==ULLONG_MAX?Failure{}:errors.records[unsigned(first)];
    warnings[angle].publishedResidual=linear->maximumResidual/linear->maximumMagnitude;
    warnings[angle].publishedTarget=target;
}
__global__ void collectPublishedDirections(int angles,const RadiationDirectionWarning* warnings,
    const Failure* failures,const unsigned char* changed,SourceReport* report){
    if(threadIdx.x)return;
    RadiationReal maximum=0;bool fixed=true;
    for(int angle=0;angle<angles;++angle){
        fixed=fixed&&!changed[angle];
        if(!report->failure.code&&failures[angle].code)report->failure=failures[angle];
        const RadiationReal r=warnings[angle].publishedResidual;
        if(std::isfinite(maximum))maximum=std::isfinite(r)?fmax(maximum,r):r;
    }
    report->transportRelativeResidual=maximum;
    report->intensityUnchanged=fixed;
}
__global__ void auditPublishedRows(OrderedUpwindDevice matrix,const RadiationReal* diagonal,
    const RadiationReal* source,const RadiationReal* x,const UpwindLinearReport* measured,SourceReport* report){
    __shared__ double wide[128],delta[128];
    const int lane=threadIdx.x;
    double maxWide=0,maxDelta=0;
    for(int cell=lane;cell<matrix.cells;cell+=blockDim.x){
        RadiationReal native=diagonal[cell]*x[cell]-source[cell];
        double precise=double(diagonal[cell])*x[cell]-double(source[cell]);
        for(int j=matrix.offsets[cell];j<matrix.offsets[cell+1];++j){
            const int e=matrix.residualOrder[j];
            const auto flux=matrix.subtract[e]?matrix.flux[e]:-matrix.flux[e];
            native+=flux*x[matrix.neighbour[e]];
            precise+=double(flux)*x[matrix.neighbour[e]];
        }
        maxWide=fmax(maxWide,fabs(precise));maxDelta=fmax(maxDelta,fabs(precise-native));
    }
    wide[lane]=maxWide;delta[lane]=maxDelta;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){
        if(lane<stride){wide[lane]=fmax(wide[lane],wide[lane+stride]);delta[lane]=fmax(delta[lane],delta[lane+stride]);}
        __syncthreads();
    }
    if(!lane){
        report->wideResidual=fmax(report->wideResidual,wide[0]/measured->maximumMagnitude);
        report->arithmeticDelta=fmax(report->arithmeticDelta,delta[0]/measured->maximumMagnitude);
    }
}
__global__ void finishSourceIteration(SourceReport* report,const RadiationReal* change,RadiationControls controls,
    cudaGraphConditionalHandle handle,const RadiationThermalCouplingDeviceView* thermal,int* stage,const diagnostics::DeviceContext* context,bool materialUpdates){
    if(threadIdx.x)return;retainDependencyFailure(report,context);report->maximumRelativeChange=*change;
    report->materialRelativeResidual=thermal->materialRelativeResidual?*thermal->materialRelativeResidual:0;
    report->materialResidualTarget=thermal->materialRelativeTolerance;
    report->transportResidualTarget=controls.linearTolerance;
    const bool materialReady=report->materialRelativeResidual<=report->materialResidualTarget;
    const bool intensityReady=*change<=controls.sourceTolerance;
    const bool transportReady=report->transportRelativeResidual<=controls.linearTolerance;
    const bool correctionWasRequested=report->correctPublishedResidual;
    const bool transportUpdateApplied=controls.sourceRelaxation==1||intensityReady;
    report->correctPublishedResidual=!transportReady&&materialReady&&transportUpdateApplied&&report->lastLinearConverged;
    const bool fixedTransport=!materialUpdates&&report->intensityUnchanged
        &&(correctionWasRequested||(!report->correctPublishedResidual&&report->lastLinearIterations==0));
    const bool finite=isfinite(*change)&&isfinite(report->materialRelativeResidual)&&isfinite(report->transportRelativeResidual)
        &&isfinite(report->lastLinearResidual)&&report->lastLinearStopReason!=int(radiationFgmresDetail::StopReason::NonFinite);
    if(!finite&&!report->failure.code)report->failure={RadiationReal(NAN),Linear,report->iterations,-1,-1,-1};





    const auto defect=report->transportRelativeResidual;
    const bool watch=intensityReady&&materialReady&&!transportReady&&!report->lastLinearConverged
        &&*change<=16*radiationEpsilon&&defect<=16*radiationEpsilon&&report->lastLinearResidual<=32*radiationEpsilon;
    if(!watch){report->plateauSamples=0;report->plateauReference=0;}
    else if(!report->plateauSamples||defect<report->plateauReference*RadiationReal(0.875)){
        report->plateauReference=defect;report->plateauSamples=1;
    }else ++report->plateauSamples;


    const auto spent=report->matvecs+report->totalLinearIterations;
    report->remainingWork=spent<report->workLimit?report->workLimit-spent:0;
    report->sourceStop=report->failure.code?5:intensityReady&&materialReady&&transportReady?1:
        report->iterations>=controls.sourceIterations?2:report->plateauSamples>=16?3:
        report->remainingWork<5?4:
        fixedTransport?6:0;
    const bool again=report->sourceStop==0;
    cudaGraphSetConditional(handle,again);*stage=again?1:4;

}

__global__ void selectSourceStage(const int* stage,int expected,cudaGraphConditionalHandle handle){
    if(!threadIdx.x)cudaGraphSetConditional(handle,*stage==expected);
}
__global__ void setSourceStage(int* stage,int value){if(!threadIdx.x)*stage=value;}
__global__ void resumeSourceLoop(int* stage,SourceReport* report,const diagnostics::DeviceContext* context,cudaGraphConditionalHandle handle){
    if(threadIdx.x)return;retainDependencyFailure(report,context);
    if(report->failure.code)*stage=4;
    cudaGraphSetConditional(handle,*stage==1||*stage==2);
}
__global__ void sourcePreparationReady(int* stage,const int* pending,cudaGraphConditionalHandle loop,
    SourceReport* source,const ConjugateGradientReport* prepared){
    if(threadIdx.x)return;
    if(pending&&*pending)cudaGraphSetConditional(loop,0);
    else{*stage=3;source->preconditionerFallbackBlocks=max(source->preconditionerFallbackBlocks,prepared->preconditionerFallbackBlocks);}
}
__global__ void moments(Geometry g,Fields f,const RadiationReal* intensities,int iteration,RadiationReal* incident,RadiationReal* power,Errors errors,const SourceReport* progress=nullptr){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=g.cells)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    RadiationReal value=0;
    for(int direction=0;direction<g.directions;++direction){value+=g.weights[direction]*consume(intensities[std::size_t(direction)*g.cells+cell],
        *f.reference,Moment,iteration,cell,direction,-1,errors,cell,direction*g.cells+cell);if(errors.records[cell].code)return;}
    incident[cell]=value;power[cell]=f.absorption[cell]==0?0:f.absorption[cell]*(value-fourPi*f.emission[cell]);
}
__global__ void boundaryResult(Geometry g,Fields f,const RadiationReal* intensities,const RadiationReal* incident,int iteration,RadiationReal* output,Errors errors,const SourceReport* progress=nullptr){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face>=g.boundary)return;
    if(progress){if(progress->failure.code)return;iteration=progress->iterations;}
    const int cell=g.boundaryOwner[face];RadiationReal result=0;
    for(int direction=0;direction<g.directions;++direction){const RadiationReal cosine=g.cosine[direction*g.boundary+face];RadiationReal value=0;
        if(cosine>0)value=consume(intensities[std::size_t(direction)*g.cells+cell],*f.reference,Outgoing,iteration,cell,g.boundaryPatch[face],g.boundaryFace[face],errors,face,face);
        else if(cosine<0)value=incoming(g,f,intensities,incident,direction,face,errors,face,face);
        if(errors.records[face].code)return;result+=g.weights[direction]*cosine*value;
    }output[face]=result;
}
const char* location(int code){switch(code){case Incident:return "incident boundary-flux reconstruction";case Specular:return "specular boundary reflection";
case Diffuse:return "diffuse reflected incident flux";case Scattering:return "table-Mie scattering source";case Linear:return "FV upwind linear solve";
case Moment:return "incident-radiation moment";default:return "outgoing boundary intensity";}}
[[noreturn]] void fail(const Failure& error){
    std::string reason;
    if(error.code==Dependency)reason="radiation stopped after a fatal producer or material error";
    else{
        reason=std::isfinite(error.value)?"negative radiation intensity beyond roundoff at ":"nonfinite radiation intensity at ";
        reason+=location(error.code);
        if(error.code==Moment)reason+=" in direction "+std::to_string(error.component);
    }
    throw RadiationError(reason,error.value,error.iteration,error.cell,error.code==Moment?-1:error.component,error.face);
}
}}}
namespace ugkp { namespace gpu {
using namespace radiationDetail;
struct DiscreteOrdinateRadiation::Implementation {
    RadiationGeometry geometry;RadiationControls controls;MieOpticalProperties optical;
    RadiationGeometryUpdate geometryUpdate{};
    unsigned long long graphRevision=0;
    bool forceColdStart=false;
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    Array<int> owner,neighbour,boundaryOwner,boundaryPatch,boundaryFace,reflected,boundaryOffsets,boundaryCells;
    Array<RadiationBoundary> boundaryType;
    Array<RadiationReal> volume,weights,internalFlux,transportDiagonal,boundaryFlux,cosines,area,denominator,fixed,directionCosines;
    Array<RadiationReal> epsilon,mass,temperature,diameter,wallTemperature,emissivity;
    Array<unsigned char> residentWallTemperature,contactEmissivity;
    cudaEvent_t inputReady=nullptr;bool residentMass=false,resultReady=false,sourceInFlight=false,preconditionerWarningPublished=false;
    diagnostics::DeviceContext* diagnostics=nullptr;
    Array<RadiationReal> absorption,scattering,extinction,asymmetry,emission,wallBlack,scaling,nextScaling,rowError,columnError,reference;
    Array<MieOpticalQuery> queries;Array<MieBracket> phaseBrackets;Array<MiePhaseMetrics> phaseMetrics;
    Array<RadiationReal> intensities,frozen,incident,initial,incidentRadiation,power,wallFlux,directionChange;

    Array<DirectionChange> changePartials,changeScratch;
    Array<unsigned char> intensityChanged;
    int changeBatch=1;
    Array<unsigned char> initialValid;
    Array<int> reflectionOffsets,reflectionDonors;Array<RadiationReal> reflectionWeights;
    Array<RadiationMaximum> partialA,partialB;Array<RadiationReal> reductionA;
    Array<unsigned long long> firstError;Array<Failure> failures,publishedDirectionFailures;
    GraphContinuationTask sourceTask;std::shared_future<void> externalCompletion;Array<int> sourceStage;Array<SourceReport> sourceReport;Array<RadiationDirectionWarning> warnings;
    Array<RadiationThermalCouplingDeviceView> thermalCoupling;
    RadiationThermalCouplingDeviceView hostThermalCoupling;
    RadiationParticleDeviceView preparedResident;
    Array<int> parentWarm;bool parentInputsPrepared=false;
    RadiationThermalIterationHook thermalIterationHook;
    unsigned long long thermalGraphOwner=0,thermalGraphRevision=0;
    int sourceGraphWidth=0,sourceSparseWidth=0;
    cudaGraph_t sourceGraph=nullptr;cudaGraphExec_t sourceExecutable=nullptr;
    Geometry g{};Fields f{};
    RadiationFgmres krylov;
    SparseConjugateGradient lowOrder;
    Array<OrderedUpwindDevice> transportMatrices;
    Array<int> cellOffsets,cellFaces;
    Array<RadiationReal> transportRhs,sweepRhs,sweepCorrection;
    Array<RadiationReal> lowDiagonal,lowUpper,lowLower,lowSource,lowBoundaryDiagonal,lowBoundarySource,lowRhs,lowCorrection;
    std::vector<int> lowBoundaryNeighbour;
    struct DirectionWorkspace {
        Array<RadiationReal> diagonal,source,solved,previous;
        Array<unsigned long long> firstError;Array<Failure> failures;
        cudaStream_t stream=nullptr;
        DirectionWorkspace(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
        ~DirectionWorkspace(){if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
    };
    struct Direction {
        OrderedUpwindPlan plan;Array<int> offsets,neighbours,subtract,residualOrder,levels,levelOffsets;Array<RadiationReal> flux;
        std::shared_ptr<DirectionWorkspace> workspace;
        Array<RadiationReal> &diagonal,&source,&solved,&previous;
        Array<unsigned long long>& firstError;Array<Failure>& failures;
        OrderedUpwindDevice matrix{};
        std::unique_ptr<OrderedUpwindGraph> graph;
        cudaStream_t stream=nullptr;cudaEvent_t complete=nullptr;
        explicit Direction(std::shared_ptr<DirectionWorkspace> w):workspace(std::move(w)),
            diagonal(workspace->diagonal),source(workspace->source),solved(workspace->solved),previous(workspace->previous),
            firstError(workspace->firstError),failures(workspace->failures),stream(workspace->stream){
            checked(cudaEventCreateWithFlags(&complete,cudaEventDisableTiming));}
        ~Direction(){if(stream)cudaStreamSynchronize(stream);graph.reset();if(complete)cudaEventDestroy(complete);}
        Errors errors(){return {firstError.ptr,failures.ptr};}
    };


    Array<RadiationReal> configuredWallTemperature,configuredEmissivity;
    std::vector<double> hostWallTemperature,hostEmissivity;
    std::vector<unsigned char> hostResidentWallTemperature,hostContactEmissivity;
    template<class T,class U> void configure(Array<T>& target,std::vector<U>& previous,const std::vector<U>& next){
        if(target.size!=next.size()||previous.size()!=next.size()||
            (!next.empty()&&std::memcmp(previous.data(),next.data(),next.size()*sizeof(U)))){
            target.upload(next,transfers);previous=next;
        }
    }
    int concurrentAngles=1;
    std::vector<std::unique_ptr<Direction>> directions;
    Implementation(const RadiationGeometry& input,const MieOpticalTable& table,const RadiationControls& options):geometry(input),controls(options),optical(table){
        if(!controls.controlsAlreadyResolved){
            controls.phaseTolerance=radiationPhaseTolerance(controls.phaseTolerance,input.directions);
            controls.sourceTolerance=radiationConvergenceTolerance(controls.sourceTolerance);
            controls.linearTolerance=radiationConvergenceTolerance(controls.linearTolerance);
        }
        controls.controlsAlreadyResolved=true;
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);
        checked(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));
    }
    ~Implementation(){if(externalCompletion.valid())externalCompletion.wait();sourceTask.reset();transfers.discard();if(stream){cudaStreamSynchronize(stream);clearSourceGraph();directions.clear();cudaStreamDestroy(stream);}if(inputReady)cudaEventDestroy(inputReady);}
    void clearSourceGraph(){sourceTask.wait();if(sourceExecutable)cudaGraphExecDestroy(sourceExecutable);if(sourceGraph)cudaGraphDestroy(sourceGraph);sourceExecutable=nullptr;sourceGraph=nullptr;}
    Errors errors(){return {firstError.ptr,failures.ptr};}
    void reset(){resetErrors<<<1,1,0,stream>>>(errors());checked(cudaMemsetAsync(failures.ptr,0,failures.size*sizeof(Failure),stream));}
    void checkErrors(){unsigned long long key=ULLONG_MAX;transfers.download(&key,firstError.ptr,sizeof(key));transfers.finish();
        if(key!=ULLONG_MAX)readFailure(key,failures.ptr);}
    void readFailure(unsigned long long key,const Failure* records){Failure error;transfers.download(&error,records+unsigned(key),sizeof(error));transfers.finish();fail(error);}
    void reset(Direction& d,cudaStream_t target){resetErrors<<<1,1,0,target>>>(d.errors());checked(cudaMemsetAsync(d.failures.ptr,0,d.failures.size*sizeof(Failure),target));}
    void maximum(int count,const RadiationReal* input,const unsigned char* valid,RadiationReal initialValue,RadiationReal* output){
        enqueueOrderedMaximum(count,input,valid,initialValue,output,partialA.ptr,partialB.ptr,workPlan().radiationBlockThreads,stream);
    }
    void initialize(){
        radiationGeometryDetail::validate(geometry);
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
        incident.resize(boundary);initial.resize(cells+boundary);initialValid.resize(cells+boundary);directionChange.resize(angles);intensityChanged.resize(angles);
        reductionA.resize(1);firstError.resize(1);failures.resize(std::max({1,cells,boundary}));
        partialA.resize(std::max(1,(std::max(cells+boundary,angles)+ugkp::gpu::workPlan().radiationBlockThreads-1)/ugkp::gpu::workPlan().radiationBlockThreads));partialB.resize(partialA.size);
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
            boundaryFlux.ptr,cosines.ptr,area.ptr,denominator.ptr,fixed.ptr};
        f={absorption.ptr,scattering.ptr,extinction.ptr,asymmetry.ptr,emission.ptr,wallBlack.ptr,scaling.ptr,reference.ptr,queries.ptr,optical.deviceTable(),phaseBrackets.ptr,emissivity.ptr};
        mieDetail::phaseBrackets<<<std::max(1,(angles*angles+ugkp::gpu::workPlan().radiationBlockThreads-1)/ugkp::gpu::workPlan().radiationBlockThreads),ugkp::gpu::workPlan().radiationBlockThreads,0,stream>>>(angles*angles,f.table,directionCosines.ptr,phaseBrackets.ptr);
        transfers.finish();
        concurrentAngles=controls.parallelAngles?std::max(1,std::min(angles,controls.maximumConcurrentAngles>0?controls.maximumConcurrentAngles:angles)):1;
        const int width=workPlan().radiationBlockThreads;
        const int parts=cells?(cells-1)/width+1:1;
        const int lanes=std::max(1,int(width*sizeof(RadiationReal)/sizeof(DirectionChange)));
        changeBatch=std::min(concurrentAngles,lanes);
        if(parts>1){
            changePartials.resize(std::size_t(parts)*changeBatch);
            if(parts>width)changeScratch.resize(std::size_t((parts-1)/width+1)*changeBatch);
        }
        std::vector<std::shared_ptr<DirectionWorkspace>> workspaces;
        for(int i=0;i<concurrentAngles;++i)workspaces.push_back(std::make_shared<DirectionWorkspace>());
        sourceReport.resize(1);sourceStage.resize(1);warnings.resize(angles);publishedDirectionFailures.resize(angles);
        thermalCoupling.resize(1);checked(cudaMemsetAsync(thermalCoupling.ptr,0,sizeof(RadiationThermalCouplingDeviceView),stream));f.thermal=thermalCoupling.ptr;
        for(int angle=0;angle<angles;++angle){
            std::unique_ptr<Direction> next(new Direction(workspaces[angle%concurrentAngles]));auto& d=*next;
            std::vector<RadiationReal> flux(geometry.internalFlux.begin()+std::size_t(angle)*g.faces,geometry.internalFlux.begin()+std::size_t(angle+1)*g.faces);
            d.plan.build(cells,geometry.owner,geometry.neighbour,geometry.cellOffsets,geometry.cellFaces,flux,true);
            d.offsets.upload(d.plan.offsets,transfers);d.neighbours.upload(d.plan.neighbour,transfers);d.subtract.upload(d.plan.subtract,transfers);
            d.residualOrder.upload(d.plan.residualOrder,transfers);d.levels.upload(d.plan.levelCells,transfers);d.levelOffsets.upload(d.plan.levelOffsets,transfers);d.flux.upload(d.plan.flux,transfers);
            for(auto* field:{&d.diagonal,&d.source,&d.solved,&d.previous})field->resize(cells);
            transfers.finish();
            d.firstError.resize(1);d.failures.resize(std::max(1,cells));
            d.matrix={cells,d.offsets.ptr,d.neighbours.ptr,d.subtract.ptr,d.residualOrder.ptr,d.levels.ptr,d.flux.ptr,d.plan.allUpstreamReady};
            d.graph.reset(new OrderedUpwindGraph(d.plan,d.matrix,g.faces,owner.ptr,neighbour.ptr,internalFlux.ptr+std::size_t(angle)*g.faces,
                d.diagonal.ptr,d.source.ptr,d.solved.ptr,d.previous.ptr,reference.ptr,
                {controls.linearTolerance,controls.linearRelativeTolerance,controls.small,controls.linearIterations,controls.controlsAlreadyResolved},d.firstError.ptr,controls.parallelAngles,false));
            directions.push_back(std::move(next));
        }
        std::vector<OrderedUpwindDevice> matrices;for(const auto& direction:directions)matrices.push_back(direction->matrix);
        transportMatrices.upload(matrices,transfers);
        cellOffsets.upload(geometry.cellOffsets,transfers);cellFaces.upload(geometry.cellFaces,transfers);
        for(auto* field:{&transportRhs,&sweepRhs,&sweepCorrection})field->resize(angular);
        for(auto* field:{&lowDiagonal,&lowSource,&lowRhs,&lowCorrection})field->resize(cells);
        lowUpper.resize(g.faces);lowLower.resize(g.faces);lowBoundaryDiagonal.resize(boundary);lowBoundarySource.resize(boundary);
        if(cells)checked(cudaMemsetAsync(lowSource.ptr,0,cells*sizeof(RadiationReal),stream));
        if(boundary)checked(cudaMemsetAsync(lowBoundarySource.ptr,0,boundary*sizeof(RadiationReal),stream));
        lowBoundaryNeighbour.assign(boundary,-1);
        SparseLduView lowMatrix;lowMatrix.rows=cells;lowMatrix.faces=g.faces;lowMatrix.boundaryFaces=boundary;
        lowMatrix.owner=geometry.owner.data();lowMatrix.neighbour=geometry.neighbour.data();
        lowMatrix.boundaryOwner=geometry.boundaryOwner.data();lowMatrix.boundaryNeighbour=lowBoundaryNeighbour.data();
        SparseDeviceLduValues lowValues{lowDiagonal.ptr,lowUpper.ptr,lowLower.ptr,lowSource.ptr,lowBoundaryDiagonal.ptr,lowBoundarySource.ptr,stream};
        ConjugateGradientControls lowControls;lowControls.maximumIterations=1;lowControls.minimumIterations=1;lowControls.tolerance=0;
        transfers.finish();lowOrder.configureResidentLocalLdu(lowMatrix,lowValues,lowControls);
        krylov.configure(int(angular));
        invalidateSourceGraph();
    }
    ExecutionGraphNode appendTransportApply(ExecutionGraph graph,ExecutionGraphNode dependency,const RadiationReal* input,RadiationReal* output){
        return executionGraphDetail::appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
            const int width=workPlan().radiationBlockThreads;
            domApply<false><<<std::max(std::size_t(1),(intensities.size+width-1)/width),width,0,stream>>>(g,f,transportMatrices.ptr,input,output);
        });
    }
    ExecutionGraphNode appendTransportPreconditioner(ExecutionGraph graph,ExecutionGraphNode dependency,const RadiationReal* input,RadiationReal* output){
        using executionGraphDetail::appendRecorded;const int width=workPlan().radiationBlockThreads;
        const int blocks=std::max(1,(g.cells+width-1)/width);
        const std::size_t angularBlocks=std::max(std::size_t(1),(intensities.size+width-1)/width);
        auto node=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
            domRestrict<<<blocks,width,0,stream>>>(g,input,lowRhs.ptr);
        });
        node=lowOrder.appendResidentLocalPreconditioner(graph,node,lowRhs.ptr,lowCorrection.ptr);
        node=appendRecorded(graph,{node},stream,[&]{
            domApply<true><<<angularBlocks,width,0,stream>>>(g,f,transportMatrices.ptr,lowCorrection.ptr,sweepRhs.ptr,input);
        });
        std::vector<cudaGraphNode_t> finished;
        for(int angle=0;angle<g.directions;++angle){auto& q=*directions[angle];
            auto swept=appendRecorded(graph,{node},stream,[&]{
                for(std::size_t level=0;level+1<q.plan.levelOffsets.size();){
                    const int begin=q.plan.levelOffsets[level],end=q.plan.levelOffsets[level+1];std::size_t next=level+1;
                    if(end-begin<=width){
                        while(next+1<q.plan.levelOffsets.size()&&next-level<std::size_t(32)
                            &&q.plan.levelOffsets[next+1]-q.plan.levelOffsets[next]<=width)++next;
                    }
                    if(next>level+1)domSweepNarrowLevels<<<1,width,0,stream>>>(int(level),int(next),q.levelOffsets.ptr,q.matrix,g,f,angle,
                        sweepRhs.ptr+std::size_t(angle)*g.cells,sweepCorrection.ptr+std::size_t(angle)*g.cells);
                    else if(end>begin)domSweepLevel<<<(end-begin+width-1)/width,width,0,stream>>>(begin,end,q.matrix,g,f,angle,
                        sweepRhs.ptr+std::size_t(angle)*g.cells,sweepCorrection.ptr+std::size_t(angle)*g.cells);
                    level=next;
                }
            });
            finished.push_back(swept);
        }
        if(finished.empty())finished.push_back(node);
        return appendRecorded(graph,finished,stream,[&]{
            domAddCorrection<<<angularBlocks,width,0,stream>>>(g,lowCorrection.ptr,sweepCorrection.ptr,output);
        });
    }
    void prepareInputs(const RadiationSnapshot& snapshot){
        auto& d=*this;const auto g=d.g;
    if(snapshot.wallTemperature.size()!=std::size_t(g.boundary)||snapshot.emissivity.size()!=std::size_t(g.boundary))
        throw std::invalid_argument("radiation wall input shape mismatch");
    auto wallTemperature=snapshot.wallTemperature;auto emissivity=snapshot.emissivity;
    if(snapshot.resident.enabled&&snapshot.residentWallTemperature.size()==wallTemperature.size()
        &&snapshot.contactEmissivity.size()==emissivity.size()){
        for(int f=0;f<g.boundary;++f){
            if(snapshot.resident.boundaryTemperature&&snapshot.residentWallTemperature[f])wallTemperature[f]=0;
            if(snapshot.resident.boundaryEmissivity&&snapshot.contactEmissivity[f])emissivity[f]=0;
        }
    }
    d.configure(d.configuredWallTemperature,d.hostWallTemperature,wallTemperature);
    d.configure(d.configuredEmissivity,d.hostEmissivity,emissivity);
    if(snapshot.thermalCoupling.producerStream){
        checked(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(snapshot.thermalCoupling.producerStream)));
        checked(cudaStreamWaitEvent(d.stream,d.inputReady,0));
    }
    d.hostThermalCoupling=snapshot.thermalCoupling;
    d.transfers.upload(d.thermalCoupling.ptr,&d.hostThermalCoupling,sizeof(d.hostThermalCoupling));
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
    }else{
        d.epsilon.upload(snapshot.epsilon,d.transfers);d.temperature.upload(snapshot.temperature,d.transfers);d.diameter.upload(snapshot.diameter,d.transfers);
    }
        d.preparedResident=snapshot.resident;
    }
    void enqueueInputs(bool warmStart,bool parent){
        auto& d=*this;const auto g=d.g;const auto f=d.f;
        const int width=workPlan().radiationBlockThreads;
        const int blocks=std::max(1,(g.cells+width-1)/width),boundaryBlocks=std::max(1,(g.boundary+width-1)/width);
    if(g.boundary){
        checked(cudaMemcpyAsync(d.wallTemperature.ptr,d.configuredWallTemperature.ptr,g.boundary*sizeof(RadiationReal),cudaMemcpyDeviceToDevice,d.stream));
        checked(cudaMemcpyAsync(d.emissivity.ptr,d.configuredEmissivity.ptr,g.boundary*sizeof(RadiationReal),cudaMemcpyDeviceToDevice,d.stream));
    }
        if(d.preparedResident.enabled){const auto input=d.preparedResident;
        importParticleState<RadiationReal><<<blocks,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(g.cells,input,g.volume,d.controls.activeEpsilonMinimum,f.table,d.epsilon.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr);
        importBoundaryState<RadiationReal><<<boundaryBlocks,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(g.boundary,input,d.residentWallTemperature.ptr,d.contactEmissivity.ptr,g.area,d.wallTemperature.ptr,d.emissivity.ptr);
            if(!parent){
        checked(cudaEventRecord(d.inputReady,d.stream));
        checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(input.producerStream),d.inputReady,0));
            }
        }
    optics<<<blocks,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(g,f,d.epsilon.ptr,d.temperature.ptr,d.diameter.ptr,d.controls.activeEpsilonMinimum,d.initial.ptr,d.initialValid.ptr);
    wallOptics<<<boundaryBlocks,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(g,f,d.wallTemperature.ptr,d.initial.ptr,d.initialValid.ptr);
    d.maximum(g.cells+g.boundary,d.initial.ptr,d.initialValid.ptr,0,d.reference.ptr);
    if(g.cells)mieDetail::balancePhase<<<g.cells,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(g.cells,g.directions,f.table,d.queries.ptr,d.weights.ptr,d.phaseBrackets.ptr,
        d.controls.phaseTolerance,d.controls.phaseIterations,d.scaling.ptr,d.nextScaling.ptr,d.rowError.ptr,d.columnError.ptr,d.phaseMetrics.ptr,d.scattering.ptr);
    const auto angularBlocks=std::max(std::size_t(1),(d.intensities.size+width-1)/width);
    if(parent)fillParentInitial<<<angularBlocks,width,0,d.stream>>>(d.intensities.size,d.intensities.ptr,d.reference.ptr,d.parentWarm.ptr);
    else if(!warmStart)fillInitial<<<angularBlocks,width,0,d.stream>>>(d.intensities.size,d.intensities.ptr,d.reference.ptr);
    }
    void invalidateSourceGraph(){
        clearSourceGraph();
        sourceGraphWidth=workPlan().radiationBlockThreads;
        sourceSparseWidth=workPlan().sparseBlockThreads;
        ++graphRevision;
    }
    void ensureStandaloneSourceGraph(){
        if(sourceExecutable)return;
        checked(cudaGraphCreate(&sourceGraph,0));
        try{
            appendSourceTo(sourceGraph,nullptr);
            checked(cudaGraphInstantiateWithFlags(&sourceExecutable,sourceGraph,0));
        }catch(...){
            clearSourceGraph();
            throw;
        }
    }
    ExecutionGraphNode appendSourceTo(ExecutionGraph graph,ExecutionGraphNode dependency){
        using executionGraphDetail::appendRecorded;
        const int width=workPlan().radiationBlockThreads;
        const int blocks=std::max(1,(g.cells+width-1)/width),boundaryBlocks=std::max(1,(g.boundary+width-1)/width);
        const std::size_t angularBlocks=std::max(std::size_t(1),(intensities.size+width-1)/width);
        auto branch=[&](cudaGraph_t graph,cudaGraphNode_t dependency,int expected,cudaGraph_t& child){
            cudaGraphConditionalHandle select;checked(cudaGraphConditionalHandleCreate(&select,graph,0,cudaGraphCondAssignDefault));
            const auto node=appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&]{
                selectSourceStage<<<1,1,0,stream>>>(sourceStage.ptr,expected,select);
            });
            return executionGraphDetail::addIf(graph,select,{node},child);
        };
        cudaGraph_t initialBody=nullptr;
        const auto initialized=branch(graph,dependency,0,initialBody);
        appendRecorded(initialBody,{},stream,[&]{
            initializeSource<<<1,1,0,stream>>>(sourceReport.ptr,controls);
            if(warnings.size)checked(cudaMemsetAsync(warnings.ptr,0,warnings.size*sizeof(RadiationDirectionWarning),stream));
            setSourceStage<<<1,1,0,stream>>>(sourceStage.ptr,1);
        });
        cudaGraphConditionalHandle handle;
        checked(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        const auto resumed=appendRecorded(graph,{initialized},stream,[&]{resumeSourceLoop<<<1,1,0,stream>>>(sourceStage.ptr,sourceReport.ptr,diagnostics,handle);});
        cudaGraph_t body=nullptr;
        const auto loop=upwindGraphDetail::addWhile(graph,handle,{resumed},body);
        cudaGraph_t beginBody=nullptr;
        const auto begun=branch(body,nullptr,1,beginBody);
        appendRecorded(beginBody,{},stream,[&]{
            beginSourceIteration<<<1,1,0,stream>>>(sourceReport.ptr);reset();
            if(intensities.size)checked(cudaMemcpyAsync(frozen.ptr,intensities.ptr,intensities.size*sizeof(RadiationReal),cudaMemcpyDeviceToDevice,stream));
            incidentFlux<<<boundaryBlocks,width,0,stream>>>(g,intensities.ptr,incident.ptr,errors());
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            domRhs<<<angularBlocks,width,0,stream>>>(g,f,transportRhs.ptr);
            domLowFace<<<std::max(1,(g.faces+width-1)/width),width,0,stream>>>(g,lowUpper.ptr,lowLower.ptr);
            domLowRows<<<blocks,width,0,stream>>>(g,f,cellOffsets.ptr,cellFaces.ptr,lowUpper.ptr,lowDiagonal.ptr);
            domLowBoundary<<<boundaryBlocks,width,0,stream>>>(g,f,lowBoundaryDiagonal.ptr);
            setSourceStage<<<1,1,0,stream>>>(sourceStage.ptr,2);
        });
        auto prepared=lowOrder.appendResidentLocalPreparation(body,begun);
        prepared=appendRecorded(body,{prepared},stream,[&]{sourcePreparationReady<<<1,1,0,stream>>>(sourceStage.ptr,lowOrder.capacityPendingDevice(),handle,sourceReport.ptr,lowOrder.deviceReport());});
        cudaGraph_t solveBody=nullptr;
        branch(body,prepared,3,solveBody);
        cudaGraphNode_t tail=nullptr;
        RadiationFgmres::Controls linearControls;
        linearControls.tolerance=controls.linearTolerance;linearControls.relativeTolerance=controls.linearRelativeTolerance;
        linearControls.small=controls.small;linearControls.maximumIterations=std::max(1,controls.linearIterations);
        linearControls.availableWork=&sourceReport.ptr->remainingWork;
        linearControls.correctPublishedResidual=&sourceReport.ptr->correctPublishedResidual;
        tail=krylov.append(solveBody,tail,stream,intensities.ptr,transportRhs.ptr,linearControls,
            [&](ExecutionGraph graph,ExecutionGraphNode dep,const RadiationReal* input,RadiationReal* output){
                return appendTransportApply(graph,dep,input,output);
            },
            [&](ExecutionGraph graph,ExecutionGraphNode dep,const RadiationReal* input,RadiationReal* output){
                return appendTransportPreconditioner(graph,dep,input,output);
            });
        tail=appendRecorded(solveBody,{tail},stream,[&]{
            domLinearWarning<<<std::max(1,(g.directions+width-1)/width),width,0,stream>>>(g.directions,krylov.report(),sourceReport.ptr,warnings.ptr,controls);
            for(int first=0;first<g.directions;first+=changeBatch){
                const int angles=std::min(changeBatch,g.directions-first);
                const auto offset=std::size_t(first)*g.cells;
                auto* current=changePartials.ptr;auto* scratch=changeScratch.ptr;
                relaxDirections<<<dim3(blocks,angles),width,width*sizeof(DirectionChange),stream>>>(g.cells,
                    controls.sourceRelaxation,frozen.ptr+offset,intensities.ptr+offset,controls.small,sourceReport.ptr,
                    current,directionChange.ptr+first,intensityChanged.ptr+first);
                int count=blocks;
                while(count>1){
                    const int next=(count-1)/width+1;
                    reduceDirectionChanges<<<dim3(next,angles),width,width*sizeof(DirectionChange),stream>>>(count,current,scratch,
                        directionChange.ptr+first,intensityChanged.ptr+first,controls.small);
                    std::swap(current,scratch);count=next;
                }
            }
        });
        if(thermalIterationHook){
            tail=appendRecorded(solveBody,{tail},stream,[&]{
                reset();moments<<<blocks,width,0,stream>>>(g,f,intensities.ptr,0,incidentRadiation.ptr,power.ptr,errors(),sourceReport.ptr);
                rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            });
            const RadiationThermalIterationDeviceView thermalView{g.cells,incidentRadiation.ptr,power.ptr,volume.ptr,
                epsilon.ptr,mass.ptr,diameter.ptr,thermalCoupling.ptr};
            tail=thermalIterationHook(solveBody,tail,stream,thermalView);
            tail=appendRecorded(solveBody,{tail},stream,[&]{
                optics<<<blocks,width,0,stream>>>(g,f,epsilon.ptr,temperature.ptr,diameter.ptr,controls.activeEpsilonMinimum,initial.ptr,initialValid.ptr);
                if(g.cells)mieDetail::balancePhase<<<g.cells,width,0,stream>>>(g.cells,g.directions,f.table,queries.ptr,weights.ptr,phaseBrackets.ptr,
                    controls.phaseTolerance,controls.phaseIterations,scaling.ptr,nextScaling.ptr,rowError.ptr,columnError.ptr,phaseMetrics.ptr,scattering.ptr);
            });
        }
        tail=appendRecorded(solveBody,{tail},stream,[&]{
            reset();incidentFlux<<<boundaryBlocks,width,0,stream>>>(g,intensities.ptr,incident.ptr,errors());
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            moments<<<blocks,width,0,stream>>>(g,f,intensities.ptr,0,incidentRadiation.ptr,power.ptr,errors(),sourceReport.ptr);
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
        });
        std::vector<cudaGraphNode_t> measured;
        for(int lane=0;lane<concurrentAngles;++lane){
            const auto laneStream=directions[lane]->stream;
            measured.push_back(appendRecorded(solveBody,{tail},laneStream,[&]{
                for(int angle=lane;angle<g.directions;angle+=concurrentAngles){
                    auto& q=*directions[angle];
                    reset(q,laneStream);
                    prepareDirection<<<blocks,width,0,laneStream>>>(g,f,angle,0,intensities.ptr,incident.ptr,
                        q.diagonal.ptr,q.source.ptr,q.solved.ptr,q.errors(),sourceReport.ptr,true);
                    q.graph->measurePublished(q.matrix,owner.ptr,neighbour.ptr,internalFlux.ptr+std::size_t(angle)*g.faces,
                        q.diagonal.ptr,q.source.ptr,q.solved.ptr,{controls.linearTolerance,controls.linearRelativeTolerance,
                        controls.small,controls.linearIterations,controls.controlsAlreadyResolved},laneStream);
                    recordPublishedDirection<<<1,1,0,laneStream>>>(angle,q.graph->report(),controls.linearTolerance,q.errors(),warnings.ptr,publishedDirectionFailures.ptr);
                }
            }));
        }
        appendRecorded(solveBody,measured,stream,[&]{
            collectPublishedDirections<<<1,1,0,stream>>>(g.directions,warnings.ptr,publishedDirectionFailures.ptr,intensityChanged.ptr,sourceReport.ptr);
            maximum(g.directions,directionChange.ptr,nullptr,0,reductionA.ptr);
            finishSourceIteration<<<1,1,0,stream>>>(sourceReport.ptr,reductionA.ptr,controls,handle,thermalCoupling.ptr,sourceStage.ptr,diagnostics,bool(thermalIterationHook));
        });
        cudaGraph_t finalBody=nullptr;const auto completed=branch(graph,loop,4,finalBody);
        appendRecorded(finalBody,{},stream,[&]{


            for(int angle=0;angle<g.directions;++angle){
                auto& q=*directions[angle];reset(q,stream);
                prepareDirection<<<blocks,width,0,stream>>>(g,f,angle,0,intensities.ptr,incident.ptr,
                    q.diagonal.ptr,q.source.ptr,q.solved.ptr,q.errors(),sourceReport.ptr,true);
                auditPublishedRows<<<1,128,0,stream>>>(q.matrix,q.diagonal.ptr,q.source.ptr,q.solved.ptr,q.graph->report(),sourceReport.ptr);
            }
            reset();incidentFlux<<<boundaryBlocks,width,0,stream>>>(g,intensities.ptr,incident.ptr,errors());
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            reset();moments<<<blocks,width,0,stream>>>(g,f,intensities.ptr,0,incidentRadiation.ptr,power.ptr,errors(),sourceReport.ptr);
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
            reset();boundaryResult<<<boundaryBlocks,width,0,stream>>>(g,f,intensities.ptr,incident.ptr,0,wallFlux.ptr,errors(),sourceReport.ptr);
            rememberFailure<<<1,1,0,stream>>>(sourceReport.ptr,errors());
        });
        return completed;
    }
};
DiscreteOrdinateRadiation::DiscreteOrdinateRadiation(const RadiationGeometry& geometry,const MieOpticalTable& table,const RadiationControls& controls):
    data_(new Implementation(geometry,table,controls)){data_->initialize();}
DiscreteOrdinateRadiation::~DiscreteOrdinateRadiation()=default;
void DiscreteOrdinateRadiation::updateGeometry(const RadiationGeometry& geometry){
    radiationGeometryDetail::validate(geometry);
    awaitCompletion();auto& d=*data_;const auto& old=d.geometry;
    d.transfers.finish();checked(cudaStreamSynchronize(d.stream));


    RadiationGeometryUpdate report{};report.geometryRevision=d.geometryUpdate.geometryRevision+1;
    report.updatedReflection=geometry.reflectedDirection!=old.reflectedDirection
        ||geometry.reflectionOffsets!=old.reflectionOffsets||geometry.reflectionDonors!=old.reflectionDonors
        ||geometry.reflectionWeights!=old.reflectionWeights||geometry.boundaryType!=old.boundaryType;
    d.resultReady=false;
    const bool topologyChanged=!radiationGeometryDetail::sameTopology(old,geometry);
    const bool widthChanged=d.sourceGraphWidth!=workPlan().radiationBlockThreads||d.sourceSparseWidth!=workPlan().sparseBlockThreads;
    if(topologyChanged||widthChanged){
        d.clearSourceGraph();d.directions.clear();d.geometry=geometry;
        d.parentInputsPrepared=false;d.forceColdStart=d.forceColdStart||topologyChanged;d.initialize();
        report.rebuiltStructure=true;report.rebuiltDirections=geometry.directions;report.rebuiltSourceGraph=true;
        report.executionRevision=d.graphRevision;d.geometryUpdate=report;return;
    }
    PinnedTransferQueue::Operation operation(d.transfers);
    bool changedAddresses=false;
    const auto upload=[&](auto& target,const auto& before,const auto& after){
        if(before==after)return;
        changedAddresses=changedAddresses||target.size!=after.size();
        target.upload(after,d.transfers);
    };
    upload(d.boundaryPatch,old.boundaryPatch,geometry.boundaryPatch);
    upload(d.boundaryFace,old.boundaryFace,geometry.boundaryFace);
    upload(d.boundaryType,old.boundaryType,geometry.boundaryType);
    upload(d.reflected,old.reflectedDirection,geometry.reflectedDirection);
    upload(d.volume,old.volume,geometry.volume);upload(d.weights,old.weights,geometry.weights);
    upload(d.internalFlux,old.internalFlux,geometry.internalFlux);
    upload(d.transportDiagonal,old.transportDiagonal,geometry.transportDiagonal);
    upload(d.boundaryFlux,old.boundaryFlux,geometry.boundaryFlux);
    upload(d.cosines,old.boundaryCosine,geometry.boundaryCosine);
    upload(d.area,old.boundaryArea,geometry.boundaryArea);
    upload(d.denominator,old.diffuseDenominator,geometry.diffuseDenominator);
    upload(d.fixed,old.fixedIntensity,geometry.fixedIntensity);
    upload(d.reflectionOffsets,old.reflectionOffsets,geometry.reflectionOffsets);
    upload(d.reflectionDonors,old.reflectionDonors,geometry.reflectionDonors);
    upload(d.reflectionWeights,old.reflectionWeights,geometry.reflectionWeights);
    if(old.directionCosines!=geometry.directionCosines){
        d.directionCosines.upload(geometry.directionCosines,d.transfers);
        const int count=geometry.directions*geometry.directions,width=workPlan().radiationBlockThreads;
        mieDetail::phaseBrackets<<<std::max(1,(count+width-1)/width),width,0,d.stream>>>(count,d.f.table,d.directionCosines.ptr,d.phaseBrackets.ptr);
    }
    d.g.reflected=d.reflected.ptr;d.g.reflectionOffsets=d.reflectionOffsets.ptr;
    d.g.reflectionDonors=d.reflectionDonors.ptr;d.g.reflectionWeights=d.reflectionWeights.ptr;
    const std::size_t faces=geometry.owner.size();
    std::vector<int> rebuild;
    for(int angle=0;angle<geometry.directions;++angle){
        auto& direction=*d.directions[angle];const std::size_t begin=std::size_t(angle)*faces;
        bool planChanged=false,valuesChanged=false;
        for(std::size_t f=0;f<faces;++f){
            const double before=old.internalFlux[begin+f],after=geometry.internalFlux[begin+f];
            valuesChanged=valuesChanged||RadiationReal(before)!=RadiationReal(after);
            planChanged=planChanged||radiationGeometryDetail::sign(before)!=radiationGeometryDetail::sign(after);
        }
        if(planChanged){
            direction.graph.reset();
            std::vector<RadiationReal> flux(geometry.internalFlux.begin()+begin,geometry.internalFlux.begin()+begin+faces);
            direction.plan.build(geometry.cells,geometry.owner,geometry.neighbour,geometry.cellOffsets,geometry.cellFaces,flux,true);
            direction.offsets.upload(direction.plan.offsets,d.transfers);direction.neighbours.upload(direction.plan.neighbour,d.transfers);
            direction.subtract.upload(direction.plan.subtract,d.transfers);direction.residualOrder.upload(direction.plan.residualOrder,d.transfers);
            direction.levels.upload(direction.plan.levelCells,d.transfers);direction.levelOffsets.upload(direction.plan.levelOffsets,d.transfers);
            direction.flux.upload(direction.plan.flux,d.transfers);
            direction.matrix={geometry.cells,direction.offsets.ptr,direction.neighbours.ptr,direction.subtract.ptr,
                direction.residualOrder.ptr,direction.levels.ptr,direction.flux.ptr,direction.plan.allUpstreamReady};
            rebuild.push_back(angle);++report.rebuiltDirections;
        }else if(valuesChanged){
            for(std::size_t i=0;i<direction.plan.face.size();++i)
                direction.plan.flux[i]=RadiationReal(geometry.internalFlux[begin+direction.plan.face[i]]);
            direction.flux.upload(direction.plan.flux,d.transfers);++report.updatedDirectionValues;
        }
    }
    if(!rebuild.empty()){
        std::vector<OrderedUpwindDevice> matrices;for(const auto& direction:d.directions)matrices.push_back(direction->matrix);
        d.transportMatrices.upload(matrices,d.transfers);
    }
    d.transfers.finish();
    for(int angle:rebuild){auto& direction=*d.directions[angle];
        direction.graph.reset(new OrderedUpwindGraph(direction.plan,direction.matrix,d.g.faces,d.owner.ptr,d.neighbour.ptr,
            d.internalFlux.ptr+std::size_t(angle)*d.g.faces,direction.diagonal.ptr,direction.source.ptr,direction.solved.ptr,
            direction.previous.ptr,d.reference.ptr,{d.controls.linearTolerance,d.controls.linearRelativeTolerance,d.controls.small,
                d.controls.linearIterations,d.controls.controlsAlreadyResolved},direction.firstError.ptr,d.controls.parallelAngles,false));
    }
    d.geometry=geometry;
    if(changedAddresses||!rebuild.empty()){d.invalidateSourceGraph();report.rebuiltSourceGraph=true;}
    report.executionRevision=d.graphRevision;d.geometryUpdate=report;
}
RadiationGeometryUpdate DiscreteOrdinateRadiation::lastGeometryUpdate()const noexcept{return data_->geometryUpdate;}
unsigned long long DiscreteOrdinateRadiation::executionRevision()const noexcept{return data_->graphRevision;}

void DiscreteOrdinateRadiation::enqueue(const RadiationSnapshot& snapshot,bool warmStart,bool publishDiagnostics){
    auto& d=*data_;
    if(d.sourceInFlight)throw std::logic_error("radiation submission still owns its input/result buffers");
    d.externalCompletion={};
    PinnedTransferQueue::Operation operation(d.transfers);d.resultReady=false;d.preconditionerWarningPublished=false;
    d.parentInputsPrepared=false;
    const auto* previousDiagnostics=d.diagnostics;
    d.prepareInputs(snapshot);
    if(d.sourceExecutable&&previousDiagnostics!=d.diagnostics)d.invalidateSourceGraph();
    d.ensureStandaloneSourceGraph();
    d.enqueueInputs(warmStart&&!d.forceColdStart,false);d.forceColdStart=false;
    setSourceStage<<<1,1,0,d.stream>>>(d.sourceStage.ptr,0);
    d.sourceTask.start(d.stream,d.sourceExecutable,[&d]{return d.lowOrder.servicePendingCapacity();},[&d,publishDiagnostics]{


        if(d.diagnostics){
            publishSourcePreconditionerWarning<<<1,1,0,d.stream>>>(d.sourceReport.ptr,d.diagnostics);
            checked(cudaGetLastError());d.preconditionerWarningPublished=true;
        }
        if(publishDiagnostics&&d.diagnostics){
            const auto g=d.g;const int width=workPlan().radiationBlockThreads;
            publishSourceDiagnostics<<<std::max(1,(std::max(g.cells,g.directions)+width-1)/width),width,0,d.stream>>>(
                d.sourceReport.ptr,d.phaseMetrics.ptr,d.warnings.ptr,g.cells,g.directions,d.controls,d.reference.ptr,d.diagnostics);
            checked(cudaGetLastError());
        }
    });
    d.sourceInFlight=true;d.resultReady=true;
}
void DiscreteOrdinateRadiation::prepareParentInputs(const RadiationSnapshot& snapshot,bool warmStart){
    auto& d=*data_;
    if(d.sourceInFlight)throw std::logic_error("radiation parent inputs still belong to an active submission");
    PinnedTransferQueue::Operation operation(d.transfers);
    d.externalCompletion={};d.prepareInputs(snapshot);d.parentWarm.resize(1);
    const int warm=warmStart&&!d.forceColdStart?1:0;d.transfers.upload(d.parentWarm.ptr,&warm,sizeof(warm));d.forceColdStart=false;
    d.parentInputsPrepared=true;d.resultReady=true;d.preconditionerWarningPublished=true;
}
ExecutionGraphNode DiscreteOrdinateRadiation::appendInputsTo(ExecutionGraph graph,ExecutionGraphNode dependency){
    auto& d=*data_;
    if(!d.parentInputsPrepared)throw std::logic_error("bind parent radiation inputs before graph construction");
    return executionGraphDetail::appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        d.enqueueInputs(true,true);setSourceStage<<<1,1,0,d.stream>>>(d.sourceStage.ptr,0);
    });
}
ExecutionGraphNode DiscreteOrdinateRadiation::appendSourceTo(ExecutionGraph graph,ExecutionGraphNode dependency){
    return data_->appendSourceTo(graph,dependency);
}
void DiscreteOrdinateRadiation::resetContinuation(void* stream){
    setSourceStage<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(data_->sourceStage.ptr,0);
}
const int* DiscreteOrdinateRadiation::continuationStage()const noexcept{return data_->sourceStage.ptr;}
const int* DiscreteOrdinateRadiation::capacityPendingDevice()const noexcept{return data_->lowOrder.capacityPendingDevice();}
bool DiscreteOrdinateRadiation::servicePendingCapacity(){return data_->lowOrder.servicePendingCapacity();}
void DiscreteOrdinateRadiation::acceptExternalCompletion(const std::shared_future<void>& completion){
    auto& d=*data_;d.externalCompletion=completion;d.sourceInFlight=true;d.resultReady=true;
}
void DiscreteOrdinateRadiation::acceptExternalExecution(void* producerStream){
    auto& d=*data_;
    checked(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(producerStream)));
    checked(cudaStreamWaitEvent(d.stream,d.inputReady,0));
}
ExecutionGraphNode DiscreteOrdinateRadiation::appendDiagnosticsTo(ExecutionGraph graph,ExecutionGraphNode dependency){
    auto& d=*data_;
    return executionGraphDetail::appendRecorded(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
        if(d.parentWarm.ptr)acceptParentWarmStart<<<1,1,0,d.stream>>>(d.sourceReport.ptr,d.parentWarm.ptr);
        if(!d.diagnostics)return;
        publishSourcePreconditionerWarning<<<1,1,0,d.stream>>>(d.sourceReport.ptr,d.diagnostics);
        const int width=workPlan().radiationBlockThreads;
        publishSourceDiagnostics<<<std::max(1,(std::max(d.g.cells,d.g.directions)+width-1)/width),width,0,d.stream>>>(
            d.sourceReport.ptr,d.phaseMetrics.ptr,d.warnings.ptr,d.g.cells,d.g.directions,d.controls,d.reference.ptr,d.diagnostics);
        checked(cudaGetLastError());
    });
}
void* DiscreteOrdinateRadiation::producerStream()const noexcept{return data_->stream;}
void DiscreteOrdinateRadiation::awaitCompletion()const{
    auto& d=*data_;d.sourceTask.wait();if(d.externalCompletion.valid())d.externalCompletion.get();d.sourceInFlight=false;
}
RadiationResult DiscreteOrdinateRadiation::solve(const RadiationSnapshot& snapshot,bool warmStart,bool exportFields){


    awaitCompletion();enqueue(snapshot,warmStart,!exportFields);awaitCompletion();
    if(!exportFields&&data_->diagnostics)return {};
    return collect(exportFields);
}
RadiationResult DiscreteOrdinateRadiation::collect(bool exportFields){
    awaitCompletion();auto& d=*data_;if(!d.resultReady)throw std::logic_error("radiation output requested before solve");
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
    if(!d.preconditionerWarningPublished&&report.preconditionerFallbackBlocks){
        diagnostics::hostWarning(diagnostics::Code::MethodFallback,
            {36010,__FILE__,__func__,__LINE__,"DOM low-order AMG construction used diagonal Jacobi"},
            report.preconditionerFallbackBlocks);
        d.preconditionerWarningPublished=true;
    }
    if(report.failure.code){d.resultReady=false;fail(report.failure);}
    result.sourceStop=report.sourceStop;result.matvecs=report.matvecs;result.workLimit=report.workLimit;
    result.wideResidual=report.wideResidual;result.arithmeticDelta=report.arithmeticDelta;
    result.linearSolves=report.linearSolves;result.totalLinearIterations=report.totalLinearIterations;
    result.lastLinearIterations=report.lastLinearIterations;result.lastLinearStopReason=report.lastLinearStopReason;
    result.lastLinearResidual=report.lastLinearResidual;result.lastLinearConverged=report.lastLinearConverged;
    result.sourceResidualTarget=d.controls.sourceTolerance;
    result.sourceConverged=report.maximumRelativeChange<=d.controls.sourceTolerance
        &&report.materialRelativeResidual<=report.materialResidualTarget
        &&report.transportRelativeResidual<=report.transportResidualTarget;
    result.sourceIterations=report.iterations;result.maximumRelativeChange=report.maximumRelativeChange;result.materialRelativeResidual=report.materialRelativeResidual;result.transportRelativeResidual=report.transportRelativeResidual;result.materialResidualTarget=report.materialResidualTarget;result.transportResidualTarget=report.transportResidualTarget;d.resultReady=true;return result;
}
RadiationDeviceView DiscreteOrdinateRadiation::deviceView()const noexcept{
    const auto& d=*data_;return {d.resultReady,d.g.cells,d.g.boundary,d.epsilon.ptr,d.residentMass?d.mass.ptr:nullptr,d.hostThermalCoupling.trialMeanTemperature?d.hostThermalCoupling.trialMeanTemperature:d.temperature.ptr,d.diameter.ptr,
        d.incidentRadiation.ptr,d.power.ptr,d.volume.ptr,d.area.ptr,d.wallFlux.ptr,d.stream,d.diagnostics};
}
void DiscreteOrdinateRadiation::setThermalIterationHook(RadiationThermalIterationHook hook,unsigned long long owner,unsigned long long revision){
    awaitCompletion();auto& d=*data_;
    const bool sameStructure=bool(hook)==bool(d.thermalIterationHook);
    const bool reusable=sameStructure&&(!hook||(owner&&owner==d.thermalGraphOwner&&revision==d.thermalGraphRevision));
    const bool widthChanged=d.sourceGraphWidth!=workPlan().radiationBlockThreads||d.sourceSparseWidth!=workPlan().sparseBlockThreads;
    const bool rebuild=!reusable||widthChanged;
    if(rebuild)checked(cudaStreamSynchronize(d.stream));
    d.thermalIterationHook=std::move(hook);d.thermalGraphOwner=owner;d.thermalGraphRevision=revision;
    if(widthChanged){


        d.clearSourceGraph();d.directions.clear();d.initialize();
    }else if(rebuild)d.invalidateSourceGraph();
}
void DiscreteOrdinateRadiation::freezeIntensities(RadiationWarmStart& snapshot, void* nextWriter) const {
    awaitCompletion();
    auto& d=*data_;
    auto& slot=*snapshot.storage_;
    const auto cells=snapshot.header_[3], directions=snapshot.header_[4];
    if (directions && cells>std::numeric_limits<std::size_t>::max()/directions)
        throw std::length_error("DOM snapshot element count overflow");
    if (d.intensities.size!=cells*directions)
        throw std::runtime_error("DOM snapshot intensity field shape differs from accepted geometry");
    SnapshotLayout layout;
    slot.range=layout.allocate<RadiationReal>(d.intensities.size);
    slot.transfer.freeze(layout.bytes,d.stream,[&](unsigned char* bytes) {
        slot.range.copy(bytes,d.intensities.ptr,d.stream);
    });
    if (nextWriter) slot.transfer.orderNextWriter(static_cast<cudaStream_t>(nextWriter));
}
void DiscreteOrdinateRadiation::downloadIntensities(std::vector<double>& output)const{
    awaitCompletion();auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);std::vector<RadiationReal> values;d.intensities.download(values,d.transfers);d.transfers.finish();
    output.assign(values.begin(),values.end());
}
void DiscreteOrdinateRadiation::uploadIntensities(const std::vector<double>& input){
    awaitCompletion();auto& d=*data_;if(input.size()!=d.intensities.size)throw std::invalid_argument("radiation warm-start shape mismatch");
    PinnedTransferQueue::Operation operation(d.transfers);d.intensities.upload(input,d.transfers);d.transfers.finish();d.forceColdStart=false;
}
}}
