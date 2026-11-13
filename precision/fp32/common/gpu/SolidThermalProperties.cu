#include "SolidThermalProperties.H"
#include "../Diagnostics.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <stdexcept>
#include <algorithm>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA solid properties: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t count){if(count<=capacity)return;T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),sizeof(T)*count));if(ptr)cudaFree(ptr);ptr=replacement;capacity=count;}
    template<class Host> void upload(const Host* host,std::size_t count,PinnedTransferQueue& queue){reserve(count);queue.uploadConverted(ptr,host,count);}
};
__global__ void fixedPointResidual(int n,const GpuReal* current,const GpuReal* previous,GpuReal* residual){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)residual[i]=current[i]-previous[i];
}
__global__ void aitkenRelaxation(int n,const GpuReal* residual,const GpuReal* previous,bool havePrevious,
    GpuReal relaxation,GpuReal minimum,GpuReal maximum,GpuReal small,GpuReal* result,const SolidThermalIteration* iteration=nullptr){
    if(blockIdx.x)return;
    if(iteration){havePrevious=iteration->iteration>1;relaxation=iteration->relaxation;}
    if(!havePrevious){if(!threadIdx.x)*result=relaxation;return;}
    __shared__ GpuReal products[256],squares[256];
    GpuReal numerator=0,denominator=0;
    // Coalesced reads and products, followed by the unchanged ordered sum.
    for(int first=0;first<n;first+=256){
        const int i=first+threadIdx.x;
        if(i<n){const GpuReal difference=residual[i]-previous[i];
            products[threadIdx.x]=previous[i]*difference;squares[threadIdx.x]=difference*difference;}
        __syncthreads();
        if(!threadIdx.x)for(int j=0;j<256&&first+j<n;++j){numerator+=products[j];denominator+=squares[j];}
        __syncthreads();
    }
    if(threadIdx.x)return;
    GpuReal candidate=relaxation;
    if(std::isfinite(denominator)&&denominator>small){
        const GpuReal value=-relaxation*numerator/denominator;
        if(std::isfinite(value))candidate=value;
    }
    candidate=candidate<minimum?minimum:candidate;
    *result=candidate<maximum?candidate:maximum;
}
__global__ void applyRelaxation(int n,const GpuReal* previous,const GpuReal* residual,const GpuReal* relaxation,
    GpuReal* current,GpuReal* history){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){current[i]=previous[i]+*relaxation*residual[i];history[i]=residual[i];}
}
__global__ void evaluateProperties(int n,const ThermalPropertyDeviceTable* tables,const GpuReal* knots,int count,
    const ThermalPropertyQuery* queries,ThermalPropertyResult* results,const GpuReal* temperature,int offset,int cells){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){auto query=queries[i];
        if(temperature&&i>=offset&&i-offset<cells)query.trial=temperature[i-offset];
        results[i]=thermalProperties(tables,knots,count,query);}
}
__global__ void evaluateResidentProperties(int cells,int boundary,unsigned fields,bool cellsFirst,
    const ThermalPropertyDeviceTable* tables,const GpuReal* knots,int knotsCount,const GpuReal* anchors,
    const GpuReal* temperature,const GpuReal* previous,const ThermalPropertyQuery* boundaryQueries,ThermalPropertyResult* results,
    const GpuReal* boundaryTemperature,const int* boundarySources,
    const GpuReal* initialBoundary=nullptr,const SolidThermalIteration* iteration=nullptr){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=cells+boundary)return;
    const int cell=cellsFirst?i:i-boundary;ThermalPropertyQuery query;
    if(cell>=0&&cell<cells)query={temperature[cell],anchors[cell],fields,previous?previous[cell]:0};
    else{const int index=cellsFirst?i-cells:i;query=boundaryQueries[index];
        if(boundaryTemperature&&boundarySources){const int source=boundarySources[index];
            const GpuReal* boundary=iteration&&iteration->iteration==1?initialBoundary:boundaryTemperature;
            query.trial=source>=0?temperature[source]:boundary[-1-source];if(cellsFirst)query.anchor=query.trial;}}
    results[i]=thermalProperties(tables,knots,knotsCount,query);
}
__global__ void extractPropertyFields(int n,const ThermalPropertyResult* results,GpuReal* capacity,GpuReal* conductivity){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){capacity[i]=results[i].capacity;conductivity[i]=results[i].conductivity;}
}
__global__ void interfaceThermalState(int count,const int* faces,const int* contact,const GpuReal* boundaryTemperature,
    const ThermalPropertyDeviceTable* tables,const GpuReal* knots,int knotsCount,GpuReal* temperature,GpuReal* effusivity,ThermalPropertyResult* status,
    bool includeEmissivity,GpuReal* emissivity){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const GpuReal T=boundaryTemperature[faces[i]];temperature[i]=T;effusivity[i]=0;
    ThermalPropertyResult value;
    if(contact[i]){value=thermalProperties(tables,knots,knotsCount,{T,T,ThermalCapacity|ThermalConductivity});
        effusivity[i]=sqrtf(value.capacity*value.conductivity);}
    if(includeEmissivity)value.emissivity=thermalProperty(tables[3],T,3,value.status);
    emissivity[i]=value.emissivity;status[i]=value;
}

__global__ void initializePropertyDiagnostic(int count,int* indices){if(!threadIdx.x){indices[0]=count;indices[1]=count;}}
__global__ void publishPropertyDiagnostic(const ThermalPropertyDiagnostic* value,diagnostics::DeviceContext* context){
    using namespace diagnostics;if(threadIdx.x)return;
    if(value->firstError>=0){const auto status=value->error;
        auto record=makeRecord(Code::InvalidPhysicalState,34010,0,static_cast<unsigned>(status.error),EntityKind::Cell,
            value->firstError,status.property,status.value,status.temperature);recordFatal(context,record);}
    if(value->firstExtrapolation>=0){const auto status=value->extrapolation;
        auto record=makeRecord(Code::AccuracyResidual,34011,0,status.extrapolated,EntityKind::Cell,
            value->firstExtrapolation,status.property,status.temperature,status.value);recordWarning(context,record);}
}
__global__ void propertyDiagnosticIndices(int n,const ThermalPropertyResult* result,int* indices){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    if(result[i].status.error!=ThermalPropertyError::None)atomicMin(indices,i);
    if(result[i].status.extrapolated)atomicMin(indices+1,i);
}
__global__ void propertyDiagnosticValues(int n,const ThermalPropertyResult* result,const int* indices,ThermalPropertyDiagnostic* out){
    ThermalPropertyDiagnostic d;if(indices[0]<n){d.firstError=indices[0];d.error=result[d.firstError].status;}
    if(indices[1]<n){d.firstExtrapolation=indices[1];d.extrapolation=result[d.firstExtrapolation].status;}*out=d;
}
__device__ void mergeConvergence(SolidThermalConvergence& a,const SolidThermalConvergence& b){
    a.enthalpyChange.merge(b.enthalpyChange);a.externalEnergy.merge(b.externalEnergy);a.interfaceEnergy.merge(b.interfaceEnergy);
    a.temperatureChange=fmaxf(a.temperatureChange,b.temperatureChange);a.surfaceChange=fmaxf(a.surfaceChange,b.surfaceChange);
    a.surfaceFluxChange=fmaxf(a.surfaceFluxChange,b.surfaceFluxChange);
}
__global__ void thermalConvergenceParts(int cells,int surfaces,int boundary,const ThermalPropertyResult* properties,const GpuReal* volume,
    const int* surfaceIndex,const GpuReal* flux,const GpuReal* temperature,GpuReal* previous,const GpuReal* gradient,
    const GpuReal* energy,const GpuReal* area,GpuReal interval,SolidThermalConvergence* partial,const double* deviceInterval=nullptr){
    if(deviceInterval)interval=static_cast<GpuReal>(*deviceInterval);
    __shared__ __align__(16) unsigned char storage[128*sizeof(SolidThermalConvergence)];auto* values=reinterpret_cast<SolidThermalConvergence*>(storage);const int i=blockIdx.x*blockDim.x+threadIdx.x;SolidThermalConvergence v;
    if(i<cells){const auto value=properties[surfaces+i];v.enthalpyChange.product(volume[i],value.energy);v.temperatureChange=value.relativeChange;}
    if(i<boundary){const int source=surfaceIndex[i];
        if(source==-2)v.externalEnergy.product(-interval,flux[i]);
        else v.interfaceEnergy.add(energy[i]);
        if(source>=0){const GpuReal T=temperature[i],old=previous[i];
            v.surfaceChange=fabsf(T-old)/fmaxf(fmaxf(fabsf(T),fabsf(old)),1.0f);
            const GpuReal prescribed=energy[i]/(area[i]*interval),actual=properties[source].conductivity*gradient[i];
            v.surfaceFluxChange=fabsf(actual-prescribed)/fmaxf(fmaxf(fabsf(actual),fabsf(prescribed)),1.0f);}
        previous[i]=temperature[i];}
    values[threadIdx.x]=v;__syncthreads();
    for(int offset=64;offset;offset/=2){if(threadIdx.x<offset)mergeConvergence(values[threadIdx.x],values[threadIdx.x+offset]);__syncthreads();}
    if(!threadIdx.x)partial[blockIdx.x]=values[0];
}
__global__ void thermalConvergenceFinish(int count,const SolidThermalConvergence* partial,SolidThermalConvergence* result){
    __shared__ __align__(16) unsigned char storage[128*sizeof(SolidThermalConvergence)];auto* values=reinterpret_cast<SolidThermalConvergence*>(storage);SolidThermalConvergence v;
    for(int i=threadIdx.x;i<count;i+=blockDim.x)mergeConvergence(v,partial[i]);values[threadIdx.x]=v;__syncthreads();
    for(int offset=64;offset;offset/=2){if(threadIdx.x<offset)mergeConvergence(values[threadIdx.x],values[threadIdx.x+offset]);__syncthreads();}
    if(!threadIdx.x)*result=values[0];
}

__global__ void contactThermalState(int count,const int* cells,const GpuReal* input,const ThermalPropertyDeviceTable* tables,
    const GpuReal* knots,int knotsCount,GpuReal* temperature,GpuReal* conductivity,ThermalPropertyResult* status){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;const GpuReal T=input[cells[i]];
    const auto value=thermalProperties(tables,knots,knotsCount,{T,T,ThermalConductivity});
    temperature[i]=T;conductivity[i]=value.conductivity;status[i]=value;
}
__global__ void contactExchangeEnergy(int count,const int* primaryForAuxiliary,const GpuReal* primaryT,const GpuReal* primaryK,
    const GpuReal* auxiliaryT,const GpuReal* auxiliaryK,const GpuReal* primaryDelta,const GpuReal* auxiliaryDelta,
    const GpuReal* area,GpuReal interval,GpuReal* energy,const double* deviceInterval=nullptr){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=count)return;const int p=primaryForAuxiliary[f];
    const GpuReal auxiliaryDenominator=auxiliaryK[f]*auxiliaryDelta[f],primaryDenominator=primaryK[p]*primaryDelta[p];
    const GpuReal resistance=1/auxiliaryDenominator+1/primaryDenominator;
    energy[f]=area[f]*(deviceInterval?static_cast<GpuReal>(*deviceInterval):interval)*(primaryT[p]-auxiliaryT[f])/resistance;
}
__global__ void contactOppositeEnergy(int count,const int* offsets,const int* sources,const GpuReal* input,GpuReal* output){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=count)return;GpuReal value=0;
    for(int i=offsets[f];i<offsets[f+1];++i)value+=-input[sources[i]];output[f]=value;
}
__device__ void mergeContactConvergence(SolidContactConvergence& a,const SolidContactConvergence& b){
    a.temperatureChange=fmaxf(a.temperatureChange,b.temperatureChange);a.energyDifference+=b.energyDifference;a.energyScale+=b.energyScale;
}
__global__ void contactConvergenceParts(int primary,int auxiliary,const GpuReal* primaryT,const GpuReal* auxiliaryT,
    const GpuReal* previousPrimary,const GpuReal* previousAuxiliary,const GpuReal* energy,GpuReal* previousEnergy,SolidContactConvergence* parts){
    __shared__ GpuReal temperature[128],difference[128],scale[128];const int i=blockIdx.x*blockDim.x+threadIdx.x;GpuReal change=0,delta=0,total=0;
    if(i<primary)change=fabsf(primaryT[i]-previousPrimary[i])/fmaxf(1.0f,fmaxf(fabsf(primaryT[i]),fabsf(previousPrimary[i])));
    if(i<auxiliary){change=fmaxf(change,fabsf(auxiliaryT[i]-previousAuxiliary[i])/fmaxf(1.0f,fmaxf(fabsf(auxiliaryT[i]),fabsf(previousAuxiliary[i]))));
        delta=fabsf(energy[i]-previousEnergy[i]);total=fabsf(energy[i]);previousEnergy[i]=energy[i];}
    temperature[threadIdx.x]=change;difference[threadIdx.x]=delta;scale[threadIdx.x]=total;__syncthreads();
    for(int d=64;d;d/=2){if(threadIdx.x<d){temperature[threadIdx.x]=fmaxf(temperature[threadIdx.x],temperature[threadIdx.x+d]);difference[threadIdx.x]+=difference[threadIdx.x+d];scale[threadIdx.x]+=scale[threadIdx.x+d];}__syncthreads();}
    if(!threadIdx.x)parts[blockIdx.x]={temperature[0],difference[0],scale[0]};
}
__global__ void contactConvergenceFinish(int count,const SolidContactConvergence* parts,SolidContactConvergence* output){
    // Only compact block summaries reach this ordered final reduction.
    SolidContactConvergence result;
    for(int i=0;i<count;++i)mergeContactConvergence(result,parts[i]);*output=result;
}

}
struct SolidThermalProperties::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;int knotCount=0;
    Array<GpuReal> temperatures[4],values[4],knots;Array<ThermalPropertyDeviceTable> tables;
    Array<GpuReal> current,previous,residual,residualHistory,relaxation;
    Array<ThermalPropertyQuery> queries;Array<ThermalPropertyResult> results;
    struct EvaluationPlan {
        Array<ThermalPropertyQuery> queries;Array<int> sources;
        int cells=0,boundary=0;unsigned fields=0;bool cellsFirst=false;
    };
    std::vector<std::unique_ptr<EvaluationPlan>> evaluationPlans;
    Array<int> interfaceFaces,interfaceContact;Array<GpuReal> interfaceTemperature,interfaceEffusivity,interfaceEmissivity;Array<ThermalPropertyResult> interfaceStatus;
    std::vector<int> cachedInterfaceFaces,cachedInterfaceContact;
    Array<int> contactCells;Array<GpuReal> contactTemperature,contactConductivity;Array<ThermalPropertyResult> contactStatus;
    std::vector<int> cachedContactCells;int contactCount=0;
    Array<int> diagnosticIndices,convergenceSurfaceIndex;Array<ThermalPropertyDiagnostic> diagnosticResult;
    Array<GpuReal> convergencePreviousSurface;Array<SolidThermalConvergence> convergenceParts,convergenceResult;int convergenceBoundaryCount=0;
    Array<GpuReal> capacity,conductivity,cellAnchors;Array<int> boundarySources;int fieldCount=0;
    cudaEvent_t externalReady=nullptr,externalDone=nullptr;
    void waitFor(void* owner){
        if(!externalReady)checked(cudaEventCreateWithFlags(&externalReady,cudaEventDisableTiming));
        checked(cudaEventRecord(externalReady,static_cast<cudaStream_t>(owner)));
        checked(cudaStreamWaitEvent(stream,externalReady,0));
    }
    void publishTo(void* owner){
        if(!externalDone)checked(cudaEventCreateWithFlags(&externalDone,cudaEventDisableTiming));
        checked(cudaEventRecord(externalDone,stream));checked(cudaStreamWaitEvent(static_cast<cudaStream_t>(owner),externalDone,0));
    }
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}if(externalReady)cudaEventDestroy(externalReady);if(externalDone)cudaEventDestroy(externalDone);}
};
SolidThermalProperties::SolidThermalProperties(const ThermalPropertyTable* tables,const double* knots,int count):data_(new Implementation){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);ThermalPropertyDeviceTable device[4];
    for(int i=0;i<4;++i){
        device[i]={tables[i].constant,tables[i].clamp,tables[i].allowNonpositive,static_cast<GpuReal>(tables[i].constantValue),tables[i].count};d.temperatures[i].upload(tables[i].temperatures,tables[i].count,d.transfers);
        d.values[i].upload(tables[i].values,tables[i].count,d.transfers);
        device[i].temperatures=d.temperatures[i].ptr;device[i].values=d.values[i].ptr;
    }
    d.tables.upload(device,4,d.transfers);d.knots.upload(knots,count,d.transfers);d.knotCount=count;d.transfers.finish();
}
SolidThermalProperties::SolidThermalProperties(const ThermalPropertyDeviceTable* tables,const GpuReal* knots,int count):data_(new Implementation){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);ThermalPropertyDeviceTable device[4];
    for(int i=0;i<4;++i){
        device[i]={tables[i].constant,tables[i].clamp,tables[i].allowNonpositive,static_cast<GpuReal>(tables[i].constantValue),tables[i].count};d.temperatures[i].upload(tables[i].temperatures,tables[i].count,d.transfers);
        d.values[i].upload(tables[i].values,tables[i].count,d.transfers);
        device[i].temperatures=d.temperatures[i].ptr;device[i].values=d.values[i].ptr;
    }
    d.tables.upload(device,4,d.transfers);d.knots.upload(knots,count,d.transfers);d.knotCount=count;d.transfers.finish();
}
SolidThermalProperties::~SolidThermalProperties()=default;
double SolidThermalProperties::relax(double* current,const double* previous,int count,bool havePrevious,
    double relaxation,double minimum,double maximum,double small){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.current.upload(current,count,d.transfers);d.previous.upload(previous,count,d.transfers);
    d.residual.reserve(count);d.residualHistory.reserve(count);d.relaxation.reserve(1);
    if(count)fixedPointResidual<<<(count+127)/128,128,0,d.stream>>>(count,d.current.ptr,d.previous.ptr,d.residual.ptr);
    aitkenRelaxation<<<1,256,0,d.stream>>>(count,d.residual.ptr,d.residualHistory.ptr,havePrevious,
        relaxation,minimum,maximum,small,d.relaxation.ptr);
    if(count){applyRelaxation<<<(count+127)/128,128,0,d.stream>>>(count,d.previous.ptr,d.residual.ptr,
        d.relaxation.ptr,d.current.ptr,d.residualHistory.ptr);
        d.transfers.downloadConverted(current,d.current.ptr,count);}
    d.transfers.downloadConverted(&relaxation,d.relaxation.ptr,1);
    d.transfers.finish();return relaxation;
}
GpuReal SolidThermalProperties::relaxResident(GpuReal* current,const GpuReal* previous,int count,bool havePrevious,
    GpuReal relaxation,GpuReal minimum,GpuReal maximum,GpuReal small,void* ownerStream){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.waitFor(ownerStream);
    d.residual.reserve(count);d.residualHistory.reserve(count);d.relaxation.reserve(1);
    if(count)fixedPointResidual<<<(count+127)/128,128,0,d.stream>>>(count,current,previous,d.residual.ptr);
    aitkenRelaxation<<<1,256,0,d.stream>>>(count,d.residual.ptr,d.residualHistory.ptr,havePrevious,
        relaxation,minimum,maximum,small,d.relaxation.ptr);
    if(count)applyRelaxation<<<(count+127)/128,128,0,d.stream>>>(count,previous,d.residual.ptr,d.relaxation.ptr,current,d.residualHistory.ptr);
    d.transfers.downloadConverted(&relaxation,d.relaxation.ptr,1);d.publishTo(ownerStream);d.transfers.finish();return relaxation;
}
void SolidThermalProperties::evaluate(const std::vector<ThermalPropertyQuery>& queries,std::vector<ThermalPropertyResult>& results,
    const GpuReal* temperature,int cellOffset,int cellCount,void* ownerStream){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=queries.size();d.queries.upload(queries.data(),n,d.transfers);d.results.reserve(n);results.resize(n);
    if(temperature)d.waitFor(ownerStream);
    d.capacity.reserve(n);d.conductivity.reserve(n);d.fieldCount=n;
    if(n){evaluateProperties<<<(n+127)/128,128,0,d.stream>>>(n,d.tables.ptr,d.knots.ptr,d.knotCount,d.queries.ptr,d.results.ptr,temperature,cellOffset,cellCount);
        extractPropertyFields<<<(n+127)/128,128,0,d.stream>>>(n,d.results.ptr,d.capacity.ptr,d.conductivity.ptr);
        d.transfers.download(results.data(),d.results.ptr,n*sizeof(ThermalPropertyResult));}
    d.transfers.finish();
}
void SolidThermalProperties::setCellAnchors(const double* temperatures,int count){
    auto& d=*data_;d.cellAnchors.upload(temperatures,count,d.transfers);
}
void SolidThermalProperties::evaluateResidentCells(const std::vector<ThermalPropertyQuery>& boundary,
    const GpuReal* temperature,const GpuReal* previous,int cells,unsigned fields,bool cellsFirst,
    void* ownerStream,std::vector<ThermalPropertyResult>& results,const GpuReal* boundaryTemperature,const std::vector<int>* boundarySources,bool download){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=cells+boundary.size();
    d.waitFor(ownerStream);d.queries.upload(boundary.data(),boundary.size(),d.transfers);
    if(boundaryTemperature&&boundarySources)d.boundarySources.upload(boundarySources->data(),boundarySources->size(),d.transfers);
    d.results.reserve(n);d.capacity.reserve(n);d.conductivity.reserve(n);d.fieldCount=n;if(download)results.resize(n);else results.clear();
    if(n){evaluateResidentProperties<<<(n+127)/128,128,0,d.stream>>>(cells,boundary.size(),fields,cellsFirst,
        d.tables.ptr,d.knots.ptr,d.knotCount,d.cellAnchors.ptr,temperature,previous,d.queries.ptr,d.results.ptr,boundaryTemperature,boundarySources?d.boundarySources.ptr:nullptr);
        extractPropertyFields<<<(n+127)/128,128,0,d.stream>>>(n,d.results.ptr,d.capacity.ptr,d.conductivity.ptr);
        if(download)d.transfers.download(results.data(),d.results.ptr,n*sizeof(ThermalPropertyResult));}
    if(download)d.transfers.finish();
}
int SolidThermalProperties::prepareResidentEvaluation(const std::vector<ThermalPropertyQuery>& queries,
    const std::vector<int>& sources,int cells,unsigned fields,bool cellsFirst,int index){
    if(queries.size()!=sources.size()||cells<0)throw std::invalid_argument("resident thermal query layout mismatch");
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    if(index<0)index=d.evaluationPlans.size();
    if(index>static_cast<int>(d.evaluationPlans.size()))throw std::out_of_range("resident thermal evaluation plan");
    if(index==static_cast<int>(d.evaluationPlans.size()))d.evaluationPlans.emplace_back(new Implementation::EvaluationPlan);
    auto& plan=d.evaluationPlans[index];
    plan->cells=cells;plan->boundary=queries.size();plan->fields=fields;plan->cellsFirst=cellsFirst;
    plan->queries.upload(queries.data(),queries.size(),d.transfers);plan->sources.upload(sources.data(),sources.size(),d.transfers);
    const int n=cells+queries.size();d.results.reserve(n);d.capacity.reserve(n);d.conductivity.reserve(n);
    return index;
}
void SolidThermalProperties::enqueueResidentEvaluation(int index,const GpuReal* temperature,const GpuReal* previous,
    const GpuReal* boundaryTemperature,void* executionStream,const SolidThermalIteration* iteration){
    auto& d=*data_;if(index<0||index>=static_cast<int>(d.evaluationPlans.size()))throw std::out_of_range("resident thermal evaluation plan");
    const auto& plan=*d.evaluationPlans[index];const int n=plan.cells+plan.boundary;d.fieldCount=n;
    const auto stream=static_cast<cudaStream_t>(executionStream);
    if(n){evaluateResidentProperties<<<(n+127)/128,128,0,stream>>>(plan.cells,plan.boundary,plan.fields,plan.cellsFirst,
        d.tables.ptr,d.knots.ptr,d.knotCount,d.cellAnchors.ptr,temperature,previous,plan.queries.ptr,d.results.ptr,boundaryTemperature,plan.sources.ptr,d.convergencePreviousSurface.ptr,iteration);
        extractPropertyFields<<<(n+127)/128,128,0,stream>>>(n,d.results.ptr,d.capacity.ptr,d.conductivity.ptr);}
}
void SolidThermalProperties::prepareResidentIteration(int cells){
    auto& d=*data_;d.cellAnchors.reserve(cells);d.residual.reserve(cells);d.residualHistory.reserve(cells);d.relaxation.reserve(1);
    d.convergenceParts.reserve((std::max(cells,d.convergenceBoundaryCount)+127)/128);d.convergenceResult.reserve(1);
    d.diagnosticIndices.reserve(2);d.diagnosticResult.reserve(1);
}
void SolidThermalProperties::enqueueResidentRelaxation(GpuReal* current,const GpuReal* previous,int count,
    SolidThermalIteration* iteration,const SolidThermalIterationControls& controls,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    if(count)fixedPointResidual<<<(count+127)/128,128,0,stream>>>(count,current,previous,d.residual.ptr);
    aitkenRelaxation<<<1,256,0,stream>>>(count,d.residual.ptr,d.residualHistory.ptr,false,
        controls.initialRelaxation,controls.minimumRelaxation,controls.maximumRelaxation,controls.small,d.relaxation.ptr,iteration);
    if(count)applyRelaxation<<<(count+127)/128,128,0,stream>>>(count,previous,d.residual.ptr,d.relaxation.ptr,current,d.residualHistory.ptr);
    checked(cudaMemcpyAsync(&iteration->relaxation,d.relaxation.ptr,sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
}
void SolidThermalProperties::enqueueResidentDiagnostic(diagnostics::DeviceContext* context,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    initializePropertyDiagnostic<<<1,1,0,stream>>>(d.fieldCount,d.diagnosticIndices.ptr);
    if(d.fieldCount)propertyDiagnosticIndices<<<(d.fieldCount+127)/128,128,0,stream>>>(d.fieldCount,d.results.ptr,d.diagnosticIndices.ptr);
    propertyDiagnosticValues<<<1,1,0,stream>>>(d.fieldCount,d.results.ptr,d.diagnosticIndices.ptr,d.diagnosticResult.ptr);
    publishPropertyDiagnostic<<<1,1,0,stream>>>(d.diagnosticResult.ptr,context);
}
void SolidThermalProperties::enqueueResidentAnchors(const GpuReal* temperature,const GpuReal* boundaryTemperature,int cells,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    if(cells)checked(cudaMemcpyAsync(d.cellAnchors.ptr,temperature,cells*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
    if(boundaryTemperature&&d.convergenceBoundaryCount)checked(cudaMemcpyAsync(d.convergencePreviousSurface.ptr,boundaryTemperature,d.convergenceBoundaryCount*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
}
const SolidThermalConvergence* SolidThermalProperties::enqueueResidentConvergence(int cells,int surfaces,const GpuReal* volume,
    const GpuReal* flux,const GpuReal* temperature,const GpuReal* gradient,const GpuReal* energy,const GpuReal* area,double interval,void* executionStream,const double* deviceInterval){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    const int blocks=(std::max(cells,d.convergenceBoundaryCount)+127)/128;
    if(blocks)thermalConvergenceParts<<<blocks,128,0,stream>>>(cells,surfaces,d.convergenceBoundaryCount,d.results.ptr,volume,
        d.convergenceSurfaceIndex.ptr,flux,temperature,d.convergencePreviousSurface.ptr,gradient,energy,area,interval,d.convergenceParts.ptr,deviceInterval);
    thermalConvergenceFinish<<<1,128,0,stream>>>(blocks,d.convergenceParts.ptr,d.convergenceResult.ptr);return d.convergenceResult.ptr;
}
void SolidThermalProperties::setResidentCellAnchors(const GpuReal* temperature,int count,void* stream){
    auto& d=*data_;d.waitFor(stream);d.cellAnchors.reserve(count);
    if(count)checked(cudaMemcpyAsync(d.cellAnchors.ptr,temperature,count*sizeof(GpuReal),cudaMemcpyDeviceToDevice,d.stream));
}
const GpuReal* SolidThermalProperties::deviceCellAnchors()const{return data_->cellAnchors.ptr;}
void SolidThermalProperties::setConvergenceBoundary(const std::vector<int>& index,const std::vector<double>& temperature){
    auto& d=*data_;d.convergenceBoundaryCount=index.size();d.convergenceSurfaceIndex.upload(index.data(),index.size(),d.transfers);
    d.convergencePreviousSurface.upload(temperature.data(),temperature.size(),d.transfers);
}
ThermalPropertyDiagnostic SolidThermalProperties::diagnostic()const{
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.diagnosticIndices.reserve(2);d.diagnosticResult.reserve(1);
    initializePropertyDiagnostic<<<1,1,0,d.stream>>>(d.fieldCount,d.diagnosticIndices.ptr);
    if(d.fieldCount)propertyDiagnosticIndices<<<(d.fieldCount+127)/128,128,0,d.stream>>>(d.fieldCount,d.results.ptr,d.diagnosticIndices.ptr);
    propertyDiagnosticValues<<<1,1,0,d.stream>>>(d.fieldCount,d.results.ptr,d.diagnosticIndices.ptr,d.diagnosticResult.ptr);
    ThermalPropertyDiagnostic result;d.transfers.download(&result,d.diagnosticResult.ptr,sizeof(result));d.transfers.finish();return result;
}
SolidThermalConvergence SolidThermalProperties::convergence(int cells,int surfaces,const GpuReal* volume,
    const GpuReal* flux,const GpuReal* temperature,const GpuReal* gradient,const GpuReal* energy,const GpuReal* area,double interval,void* stream){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.waitFor(stream);
    prepareResidentIteration(cells);
    enqueueResidentConvergence(cells,surfaces,volume,flux,temperature,gradient,energy,area,interval,d.stream);
    SolidThermalConvergence result;d.transfers.download(&result,d.convergenceResult.ptr,sizeof(result));d.transfers.finish();return result;
}

ThermalSolidContactDeviceView SolidThermalProperties::prepareResidentContact(const std::vector<int>& cells){
    auto& d=*data_;d.contactCount=cells.size();
    if(d.cachedContactCells!=cells){d.contactCells.upload(cells.data(),cells.size(),d.transfers);d.cachedContactCells=cells;}
    d.contactTemperature.reserve(cells.size());d.contactConductivity.reserve(cells.size());d.contactStatus.reserve(cells.size());
    d.diagnosticIndices.reserve(2);d.diagnosticResult.reserve(1);return residentContact();
}
ThermalSolidContactDeviceView SolidThermalProperties::residentContact()const{
    const auto& d=*data_;return {d.contactCount,d.contactTemperature.ptr,d.contactConductivity.ptr,d.stream};
}
void SolidThermalProperties::enqueueResidentContact(const GpuReal* temperature,diagnostics::DeviceContext* context,void* executionStream){
    auto& d=*data_;const int n=d.contactCount;const auto stream=static_cast<cudaStream_t>(executionStream);
    if(n)contactThermalState<<<(n+127)/128,128,0,stream>>>(n,d.contactCells.ptr,temperature,d.tables.ptr,d.knots.ptr,d.knotCount,
        d.contactTemperature.ptr,d.contactConductivity.ptr,d.contactStatus.ptr);
    if(context){
        initializePropertyDiagnostic<<<1,1,0,stream>>>(n,d.diagnosticIndices.ptr);
        if(n)propertyDiagnosticIndices<<<(n+127)/128,128,0,stream>>>(n,d.contactStatus.ptr,d.diagnosticIndices.ptr);
        propertyDiagnosticValues<<<1,1,0,stream>>>(n,d.contactStatus.ptr,d.diagnosticIndices.ptr,d.diagnosticResult.ptr);
        publishPropertyDiagnostic<<<1,1,0,stream>>>(d.diagnosticResult.ptr,context);
    }
}
ThermalSolidContactDeviceView SolidThermalProperties::contactState(const std::vector<int>& cells,const GpuReal* temperature,
    void* stream,std::vector<ThermalPropertyResult>& status){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.waitFor(stream);
    const auto view=prepareResidentContact(cells);enqueueResidentContact(temperature,nullptr,d.stream);
    status.resize(cells.size());if(!status.empty())d.transfers.download(status.data(),d.contactStatus.ptr,status.size()*sizeof(ThermalPropertyResult));
    d.transfers.finish();return view;
}

ThermalInterfaceStateDeviceView SolidThermalProperties::interfaceState(const std::vector<int>& faces,
    const std::vector<int>& contact,const GpuReal* temperature,void* ownerStream,std::vector<ThermalPropertyResult>& status,bool includeEmissivity,diagnostics::DeviceContext* context,void* diagnosticStream){
    auto& d=*data_;if(!temperature)return {};
    if(faces.size()!=contact.size())throw std::runtime_error("solid interface contact map size mismatch");
    PinnedTransferQueue::Operation operation(d.transfers);d.waitFor(ownerStream);const int n=faces.size();
    if(d.cachedInterfaceFaces!=faces){d.interfaceFaces.upload(faces.data(),n,d.transfers);d.cachedInterfaceFaces=faces;}
    if(d.cachedInterfaceContact!=contact){d.interfaceContact.upload(contact.data(),n,d.transfers);d.cachedInterfaceContact=contact;}
    d.interfaceTemperature.reserve(n);d.interfaceEffusivity.reserve(n);d.interfaceStatus.reserve(n);if(context)status.clear();else status.resize(n);
    d.interfaceEmissivity.reserve(n);
    if(context)d.waitFor(diagnosticStream);
    if(n){interfaceThermalState<<<(n+127)/128,128,0,d.stream>>>(n,d.interfaceFaces.ptr,d.interfaceContact.ptr,temperature,
        d.tables.ptr,d.knots.ptr,d.knotCount,d.interfaceTemperature.ptr,d.interfaceEffusivity.ptr,d.interfaceStatus.ptr,
            includeEmissivity,d.interfaceEmissivity.ptr);
        if(!context)d.transfers.download(status.data(),d.interfaceStatus.ptr,n*sizeof(ThermalPropertyResult));}
    if(context){
        status.clear();d.diagnosticIndices.reserve(2);d.diagnosticResult.reserve(1);
        initializePropertyDiagnostic<<<1,1,0,d.stream>>>(n,d.diagnosticIndices.ptr);
        if(n)propertyDiagnosticIndices<<<(n+127)/128,128,0,d.stream>>>(n,d.interfaceStatus.ptr,d.diagnosticIndices.ptr);
        propertyDiagnosticValues<<<1,1,0,d.stream>>>(n,d.interfaceStatus.ptr,d.diagnosticIndices.ptr,d.diagnosticResult.ptr);
        publishPropertyDiagnostic<<<1,1,0,d.stream>>>(d.diagnosticResult.ptr,context);
    }else d.transfers.finish();
    return {true,n,d.interfaceTemperature.ptr,d.interfaceEffusivity.ptr,d.interfaceContact.ptr,d.stream,
        includeEmissivity?d.interfaceEmissivity.ptr:nullptr};
}

SolidThermalDeviceFields SolidThermalProperties::deviceFields()const{
    const auto& d=*data_;return {d.capacity.ptr,d.conductivity.ptr,d.fieldCount,d.stream};
}
struct SolidThermalContact::Implementation {
    cudaStream_t stream=nullptr;cudaEvent_t inputReady=nullptr;PinnedTransferQueue transfers;
    Array<int> mapping,offsets,sources;Array<GpuReal> primaryDelta,auxiliaryDelta,area,primaryEnergy,auxiliaryEnergy;
    Array<GpuReal> previousPrimary,previousAuxiliary,previousEnergy;
    Array<SolidContactConvergence> parts,result;int primary=0,auxiliary=0;
    std::vector<int> cachedMapping;std::vector<double> cachedPrimaryDelta,cachedAuxiliaryDelta,cachedArea;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);checked(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));}
    ~Implementation(){transfers.discard();cudaStreamSynchronize(stream);cudaStreamDestroy(stream);cudaEventDestroy(inputReady);}
    void waitFor(void* producer){checked(cudaEventRecord(inputReady,static_cast<cudaStream_t>(producer)));checked(cudaStreamWaitEvent(stream,inputReady,0));}
};
SolidThermalContact::SolidThermalContact():data_(new Implementation){}
SolidThermalContact::~SolidThermalContact()=default;
void SolidThermalContact::begin(const std::vector<int>& mapping,const std::vector<double>& primaryDelta,
    const std::vector<double>& auxiliaryDelta,const std::vector<double>& area){
    auto& d=*data_;d.primary=primaryDelta.size();d.auxiliary=auxiliaryDelta.size();
    if(mapping.size()!=auxiliaryDelta.size()||area.size()!=auxiliaryDelta.size())throw std::runtime_error("solid contact geometry extent mismatch");
    if(d.cachedMapping!=mapping||d.cachedPrimaryDelta.size()!=primaryDelta.size()){
        std::vector<int> offsets(d.primary+1,0),sources(mapping.size());
        for(int p:mapping){if(p<0||p>=d.primary)throw std::runtime_error("solid contact mapping outside primary interface");++offsets[p+1];}
        for(int p=0;p<d.primary;++p)offsets[p+1]+=offsets[p];auto cursor=offsets;
        for(int f=0;f<d.auxiliary;++f)sources[cursor[mapping[f]]++]=f;
        d.mapping.upload(mapping.data(),mapping.size(),d.transfers);d.offsets.upload(offsets.data(),offsets.size(),d.transfers);d.sources.upload(sources.data(),sources.size(),d.transfers);d.cachedMapping=mapping;
    }
    if(d.cachedPrimaryDelta!=primaryDelta){d.primaryDelta.upload(primaryDelta.data(),primaryDelta.size(),d.transfers);d.cachedPrimaryDelta=primaryDelta;}
    if(d.cachedAuxiliaryDelta!=auxiliaryDelta){d.auxiliaryDelta.upload(auxiliaryDelta.data(),auxiliaryDelta.size(),d.transfers);d.cachedAuxiliaryDelta=auxiliaryDelta;}
    if(d.cachedArea!=area){d.area.upload(area.data(),area.size(),d.transfers);d.cachedArea=area;}
    d.primaryEnergy.reserve(d.primary);d.auxiliaryEnergy.reserve(d.auxiliary);d.previousPrimary.reserve(d.primary);d.previousAuxiliary.reserve(d.auxiliary);d.previousEnergy.reserve(d.auxiliary);
    d.parts.reserve((std::max(d.primary,d.auxiliary)+127)/128);d.result.reserve(1);
    if(d.primary)checked(cudaMemsetAsync(d.primaryEnergy.ptr,0,d.primary*sizeof(GpuReal),d.stream));
    if(d.auxiliary)checked(cudaMemsetAsync(d.auxiliaryEnergy.ptr,0,d.auxiliary*sizeof(GpuReal),d.stream));
    enqueueWindowStart(d.stream);
}
void SolidThermalContact::evaluate(const ThermalSolidContactDeviceView& primary,const ThermalSolidContactDeviceView& auxiliary,
    double interval,ThermalInterfaceEnergyDeviceView& primaryEnergy,ThermalInterfaceEnergyDeviceView& auxiliaryEnergy,
    std::vector<double>& primaryAudit,std::vector<double>& auxiliaryAudit){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    if(primary.faces!=d.primary||auxiliary.faces!=d.auxiliary)throw std::runtime_error("solid contact state extent mismatch");
    d.waitFor(primary.producerStream);d.waitFor(auxiliary.producerStream);
    enqueueEvaluate(primary,auxiliary,interval,nullptr,d.stream);
    primaryAudit.resize(d.primary);auxiliaryAudit.resize(d.auxiliary);
    if(d.primary)d.transfers.downloadConverted(primaryAudit.data(),d.primaryEnergy.ptr,d.primary);
    if(d.auxiliary)d.transfers.downloadConverted(auxiliaryAudit.data(),d.auxiliaryEnergy.ptr,d.auxiliary);d.transfers.finish();
    primaryEnergy={true,d.primary,d.primaryEnergy.ptr,d.stream};auxiliaryEnergy={true,d.auxiliary,d.auxiliaryEnergy.ptr,d.stream};
}
SolidContactConvergence SolidThermalContact::convergence(const ThermalSolidContactDeviceView& primary,const ThermalSolidContactDeviceView& auxiliary){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.waitFor(primary.producerStream);d.waitFor(auxiliary.producerStream);
    enqueueConvergence(primary,auxiliary,d.stream);SolidContactConvergence result;
    d.transfers.download(&result,d.result.ptr,sizeof(result));d.transfers.finish();return result;
}

ThermalInterfaceEnergyDeviceView SolidThermalContact::primaryEnergy(const double* interval)const{
    const auto& d=*data_;return {true,d.primary,d.primaryEnergy.ptr,d.stream,interval};
}
ThermalInterfaceEnergyDeviceView SolidThermalContact::auxiliaryEnergy(const double* interval)const{
    const auto& d=*data_;return {true,d.auxiliary,d.auxiliaryEnergy.ptr,d.stream,interval};
}
void SolidThermalContact::enqueueWindowStart(void* executionStream){
    auto& d=*data_;if(d.auxiliary)checked(cudaMemsetAsync(d.previousEnergy.ptr,0,d.auxiliary*sizeof(GpuReal),static_cast<cudaStream_t>(executionStream)));
}
void SolidThermalContact::enqueueEvaluate(const ThermalSolidContactDeviceView& primary,const ThermalSolidContactDeviceView& auxiliary,
    double interval,const double* deviceInterval,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    if(d.primary)checked(cudaMemcpyAsync(d.previousPrimary.ptr,primary.temperature,d.primary*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
    if(d.auxiliary){checked(cudaMemcpyAsync(d.previousAuxiliary.ptr,auxiliary.temperature,d.auxiliary*sizeof(GpuReal),cudaMemcpyDeviceToDevice,stream));
        contactExchangeEnergy<<<(d.auxiliary+127)/128,128,0,stream>>>(d.auxiliary,d.mapping.ptr,primary.temperature,primary.conductivity,
            auxiliary.temperature,auxiliary.conductivity,d.primaryDelta.ptr,d.auxiliaryDelta.ptr,d.area.ptr,interval,d.auxiliaryEnergy.ptr,deviceInterval);}
    if(d.primary)contactOppositeEnergy<<<(d.primary+127)/128,128,0,stream>>>(d.primary,d.offsets.ptr,d.sources.ptr,d.auxiliaryEnergy.ptr,d.primaryEnergy.ptr);
}
const SolidContactConvergence* SolidThermalContact::enqueueConvergence(const ThermalSolidContactDeviceView& primary,
    const ThermalSolidContactDeviceView& auxiliary,void* executionStream){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(executionStream);
    const int blocks=(std::max(d.primary,d.auxiliary)+127)/128;
    if(blocks)contactConvergenceParts<<<blocks,128,0,stream>>>(d.primary,d.auxiliary,primary.temperature,auxiliary.temperature,
        d.previousPrimary.ptr,d.previousAuxiliary.ptr,d.auxiliaryEnergy.ptr,d.previousEnergy.ptr,d.parts.ptr);
    contactConvergenceFinish<<<1,1,0,stream>>>(blocks,d.parts.ptr,d.result.ptr);return d.result.ptr;
}
}}
