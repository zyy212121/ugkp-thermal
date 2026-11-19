#include "WorkPlan.H"
#include "ParticleRadiationMaterial.H"
#include "../Diagnostics.H"
#include "RadiationImplicitMath.H"
#include "PinnedTransferQueue.cuh"
#include "ExecutionGraph.cuh"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <stdexcept>
#include <string>
#include <atomic>
#include "AnchoredMaterialKernels.cuh"
namespace ugkp { namespace gpu { namespace {
unsigned long long nextMaterialGraphOwner(){static std::atomic<unsigned long long> next{1};return next.fetch_add(1,std::memory_order_relaxed);}
void checkedMaterial(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA particle radiation: ")+cudaGetErrorString(status));}
template<class T> struct MaterialArray {
    T* ptr=nullptr;std::size_t capacity=0;
    ~MaterialArray(){if(ptr)cudaFree(ptr);}
    void clear(){if(ptr)checkedMaterial(cudaFree(ptr));ptr=nullptr;capacity=0;}
    void reserve(std::size_t size){if(size<=capacity)return;T* next=nullptr;checkedMaterial(cudaMalloc(reinterpret_cast<void**>(&next),size*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=size;}
    template<class U> void upload(const std::vector<U>& values,PinnedTransferQueue& transfer){
        reserve(values.size());const std::vector<T> converted(values.begin(),values.end());
        transfer.upload(ptr,converted.data(),converted.size()*sizeof(T));
    }
};
constexpr RadiationReal fourPiMaterial=RadiationReal(12.56637061435917295385057353311801153679);
__host__ __device__ RadiationReal materialBlack(RadiationReal t,RadiationReal band){const RadiationReal square=t*t;return band*RadiationReal(5.670374419e-8)*square*square/(fourPiMaterial/4);}
struct MaterialBalance {
    MieOpticalTableView table;RadiationReal diameter,incident;
    __host__ __device__ RadiationReal operator()(RadiationReal temperature)const{return incident-fourPiMaterial*materialBlack(temperature,mieBand(table,temperature));}
};
struct MaterialPower {
    MaterialBalance balance;RadiationReal epsilon,volume;

    __host__ __device__ RadiationReal operator()(RadiationReal temperature)const{
        const auto d=mieLogBracket(balance.table.diameter,balance.table.diameters,balance.diameter);
        const auto t=mieLinearBracket(balance.table.temperature,balance.table.temperatures,temperature);
        const RadiationReal absorption=3*epsilon*mieBilinear(balance.table.absorption,balance.table.temperatures,d,t)/(2*balance.diameter);

        return volume*(absorption==0?0:absorption*balance(temperature));
    }
};
__global__ void countResponseParticles(ParticleRadiationResponseView view,int cells,int* counts,diagnostics::DeviceContext* context){
    if(diagnostics::hasFatal(context))return;
    view=resolveParticleRadiationResponse(view);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*view.count;i+=blockDim.x*gridDim.x){
        if(view.status[i]!=1||view.wallState[i]!=0)continue;
        const int c=view.cell[i];if(c<0||c>=cells){
            using namespace diagnostics;recordFatal(context,makeRecord(Code::InvalidIndex,37002,0,0,
                EntityKind::Particle,i,-1,c,cells,0,Quantity::Index));return;
        }
        atomicAdd(counts+c,1);
    }
}
__global__ void indexResponseParticles(ParticleRadiationResponseView view,int* cursor,int* indices,diagnostics::DeviceContext* context){
    if(diagnostics::hasFatal(context))return;
    view=resolveParticleRadiationResponse(view);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*view.count;i+=blockDim.x*gridDim.x){
        if(view.status[i]!=1||view.wallState[i]!=0)continue;
        indices[atomicAdd(cursor+view.cell[i],1)]=i;
    }
}
__global__ void selectResponseIndex(const ParticleRadiationResponseView* response,
    cudaGraphConditionalHandle condition,diagnostics::DeviceContext* context){
    if(!blockIdx.x&&!threadIdx.x)cudaGraphSetConditional(condition,
        !diagnostics::hasFatal(context)&&response->enabled&&!response->cellOffsets);
}
__global__ void materialRoots(int count,MieOpticalTableView table,const RadiationReal* epsilon,const RadiationReal* mass,const RadiationReal* temperature,
    const RadiationReal* diameter,const RadiationReal* incident,const RadiationReal* volume,RadiationReal heatFactor,RadiationReal elapsed,RadiationReal minimum,RadiationReal maximum,RadiationReal cutoff,
    ParticleRadiationSolveControls controls,ParticleRadiationResponseView response,const int* offsets,const int* indices,RadiationReal* next,RadiationReal* energy,diagnostics::DeviceContext* context){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=count)return;
    response=resolveParticleRadiationResponse(response);
    if(response.cellOffsets){offsets=response.cellOffsets;indices=nullptr;}
    const RadiationReal old=temperature[cell];next[cell]=old;energy[cell]=0;
    if(diagnostics::hasFatal(context)||!(epsilon[cell]>cutoff))return;
    const RadiationReal capacity=mass[cell]*heatFactor;
    const MaterialBalance balance{table,diameter[cell],incident[cell]};const MaterialPower power{balance,epsilon[cell],volume[cell]};
    if(response.enabled){
        const IndexedParticleRadiationResponse caloric{response,indices,offsets[cell],offsets[cell+1]};
        const auto root=solveCaloricParticleRadiation(caloric,power,elapsed,controls.caloricMaximumIterations,controls.caloricEnergyRelativeTolerance);
        next[cell]=root.temperature;energy[cell]=root.energy;return;
    }
    const auto root=radiationMath::solveBoundedImplicitMaterialTemperature(old,minimum,maximum,capacity,elapsed,controls.boundedMaximumIterations,controls.boundedTemperatureRelativeTolerance,controls.boundedEnergyRelativeTolerance,balance,power);
    next[cell]=root.temperatureK;energy[cell]=capacity*(root.temperatureK-old);
}

__global__ void rawBoundaryEnergy(int count,RadiationReal elapsed,const RadiationReal* area,const RadiationReal* flux,RadiationReal* energy,diagnostics::DeviceContext* context,const double* deviceElapsed=nullptr){
    if(deviceElapsed)elapsed=RadiationReal(*deviceElapsed);
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face<count)energy[face]=diagnostics::hasFatal(context)?0:elapsed*area[face]*flux[face];
}
__global__ void publishMaterialBoundary(int faces,const RadiationReal* raw,const unsigned char* coupled,
    RadiationReal* boundary,RadiationReal* escaped){
    if(threadIdx.x)return;


    RadiationReal other=0;
    for(int face=0;face<faces;++face){boundary[face]=coupled[face]?raw[face]:0;if(!coupled[face])other+=raw[face];}
    *escaped=other;
}

}
struct ParticleRadiationMaterial::Implementation {
    ParticleRadiationSolveControls controls;
    std::shared_future<void> externalCompletion;
    ParticleRadiationResponseView preparedResponse;
    RadiationParticleDeviceView preparedParticles;
    bool preparingParent=false,parentPrepared=false;
    const double* deviceElapsed=nullptr;
    std::size_t responseScanBytes=0;
    MieOpticalProperties optical;cudaStream_t stream=nullptr;cudaEvent_t inputReady=nullptr;PinnedTransferQueue transfer;
    MaterialArray<RadiationReal> epsilon,mass,temperature,diameter,incident,volume,area,flux,next,energy,raw,boundary,escaped;
    MaterialArray<unsigned char> coupled,scanScratch;
    std::vector<unsigned char> coupledHost;
    MaterialArray<RadiationEntryParticle<GpuReal>> anchors;
    MaterialArray<int> anchorOffsets;
    MaterialArray<RadiationEnergyDomain<RadiationReal>> anchorDomains;
    MaterialArray<RadiationMaximum> maximumA,maximumB;
    MaterialArray<RadiationReal> actualEnergy,entryB,slope,plusSlope,minusSlope,linearizedEmission,thermalFeedback,materialResidual,materialRelativeResidual,maximumMaterialResidual,finalFraction,searchRequest,searchActual,searchTemperature;
    MaterialArray<unsigned char> materialValid,trialInDomain,usedTemperatureGrid;
    MaterialArray<RadiationReal> temperatureGridAllowance,materialEffectiveRelative,materialAllowedResidual;
    MaterialArray<AnchorWindowDescriptor> windowDescriptor;
 const unsigned long long graphOwner=nextMaterialGraphOwner();
 unsigned long long graphRevision=0;
 std::vector<const void*> graphPointers;
 int graphCells=-1,graphWidth=0;
 RadiationDeviceView anchorState{};
    RadiationReal anchorDt=0,anchorMinimum=0,anchorMaximum=0,anchorCp=0,anchorCutoff=0;
    bool anchorParcels=false,anchorActive=false;
    AnchorArrays arrays(){AnchorArrays result{energy.ptr,actualEnergy.ptr,next.ptr,entryB.ptr,slope.ptr,plusSlope.ptr,minusSlope.ptr,linearizedEmission.ptr,thermalFeedback.ptr,materialResidual.ptr,materialRelativeResidual.ptr,maximumMaterialResidual.ptr,finalFraction.ptr,searchRequest.ptr,searchActual.ptr,searchTemperature.ptr,materialValid.ptr,trialInDomain.ptr,anchorDomains.ptr,anchorCutoff};
 result.temperatureGrid=temperatureGridAllowance.ptr;result.effectiveRelative=materialEffectiveRelative.ptr;result.allowedResidual=materialAllowedResidual.ptr;result.usedTemperatureGrid=usedTemperatureGrid.ptr;result.relativeTolerance=controls.boundedEnergyRelativeTolerance;
 return result;}
    int cells=0,faces=0;bool resultReady=false;
    diagnostics::DeviceContext* diagnostics=nullptr;
    MaterialArray<int> responseCounts,responseOffsets,responseCursor,responseIndices;
    cudaGraph_t responseGraph=nullptr;cudaGraphExec_t responseExecutable=nullptr;
    const ParticleRadiationResponseView* responseSource=nullptr;
    diagnostics::DeviceContext* responseDiagnostics=nullptr;
    int responseCells=-1,responseCapacity=-1,responseWidth=0;
    void releaseResponseGraph(){
        if(responseExecutable){checkedMaterial(cudaGraphExecDestroy(responseExecutable));responseExecutable=nullptr;}
        if(responseGraph){checkedMaterial(cudaGraphDestroy(responseGraph));responseGraph=nullptr;}
    }
    void prepareResponseIndex(ParticleRadiationResponseView response,int count){
        const auto* oldCounts=responseCounts.ptr;const auto* oldOffsets=responseOffsets.ptr;
        const auto* oldCursor=responseCursor.ptr;const auto* oldIndices=responseIndices.ptr;const auto* oldScratch=scanScratch.ptr;
        responseCounts.reserve(count+1);responseOffsets.reserve(count+1);responseCursor.reserve(count+1);responseIndices.reserve(response.capacity);
        responseScanBytes=0;
        checkedMaterial(cub::DeviceScan::ExclusiveSum(nullptr,responseScanBytes,responseCounts.ptr,responseOffsets.ptr,count+1,stream));
        scanScratch.reserve(responseScanBytes);
        if(oldCounts!=responseCounts.ptr||oldOffsets!=responseOffsets.ptr||oldCursor!=responseCursor.ptr
            ||oldIndices!=responseIndices.ptr||oldScratch!=scanScratch.ptr)releaseResponseGraph();
    }
    void enqueueResponseIndex(ParticleRadiationResponseView response,int count){
        checkedMaterial(cudaMemsetAsync(responseCounts.ptr,0,(count+1)*sizeof(int),stream));
        const int width=workPlan().radiationBlockThreads,blocks=(response.capacity+width-1)/width;
        if(blocks)countResponseParticles<<<blocks,width,0,stream>>>(response,count,responseCounts.ptr,diagnostics);
        checkedMaterial(cub::DeviceScan::ExclusiveSum(scanScratch.ptr,responseScanBytes,responseCounts.ptr,responseOffsets.ptr,count+1,stream));
        checkedMaterial(cudaMemcpyAsync(responseCursor.ptr,responseOffsets.ptr,(count+1)*sizeof(int),cudaMemcpyDeviceToDevice,stream));
        if(blocks)indexResponseParticles<<<blocks,width,0,stream>>>(response,responseCursor.ptr,responseIndices.ptr,diagnostics);
        checkedMaterial(cudaGetLastError());
    }
    ExecutionGraphNode appendResponseIndexTo(ExecutionGraph graph,ExecutionGraphNode dependency,ParticleRadiationResponseView response,int count){
        using executionGraphDetail::appendCaptured;
        const auto deps=dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{};
        if(!response.deviceView)return appendCaptured(graph,deps,stream,[&]{enqueueResponseIndex(response,count);});
        cudaGraphConditionalHandle condition;checkedMaterial(cudaGraphConditionalHandleCreate(&condition,graph,0,cudaGraphCondAssignDefault));
        const auto selected=appendCaptured(graph,deps,stream,[&]{selectResponseIndex<<<1,1,0,stream>>>(response.deviceView,condition,diagnostics);});
        cudaGraph_t body=nullptr;const auto done=executionGraphDetail::addIf(graph,condition,{selected},body);
        appendCaptured(body,{},stream,[&]{enqueueResponseIndex(response,count);});return done;
    }
    void buildResponseIndex(ParticleRadiationResponseView response,int count){
        prepareResponseIndex(response,count);
        if(preparingParent)return;
        if(!response.deviceView){enqueueResponseIndex(response,count);return;}
        if(!responseExecutable||responseSource!=response.deviceView||responseCells!=count
            ||responseCapacity!=response.capacity||responseDiagnostics!=diagnostics||responseWidth!=workPlan().radiationBlockThreads){
            releaseResponseGraph();checkedMaterial(cudaGraphCreate(&responseGraph,0));appendResponseIndexTo(responseGraph,nullptr,response,count);
            checkedMaterial(cudaGraphInstantiateWithFlags(&responseExecutable,responseGraph,0));
            responseSource=response.deviceView;responseCells=count;responseCapacity=response.capacity;responseDiagnostics=diagnostics;responseWidth=workPlan().radiationBlockThreads;
        }
        checkedMaterial(cudaGraphLaunch(responseExecutable,stream));
    }
    explicit Implementation(const MieOpticalTable& table,const ParticleRadiationSolveControls& options):controls(options),optical(table){checkedMaterial(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);checkedMaterial(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));}
    ~Implementation(){if(externalCompletion.valid())externalCompletion.wait();transfer.discard();if(responseExecutable)cudaGraphExecDestroy(responseExecutable);if(responseGraph)cudaGraphDestroy(responseGraph);if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}if(inputReady)cudaEventDestroy(inputReady);}
};
ParticleRadiationMaterial::ParticleRadiationMaterial(const MieOpticalTable& table,const ParticleRadiationSolveControls& controls):data_(new Implementation(table,controls)){}
ParticleRadiationMaterial::~ParticleRadiationMaterial()=default;
RadiationThermalCouplingDeviceView ParticleRadiationMaterial::beginAnchoredWindow(const ParticleRadiationMaterialInput& input){
 auto& d=*data_;
 if(!d.preparingParent){awaitCompletion();d.externalCompletion={};d.deviceElapsed=nullptr;d.parentPrepared=false;}
 PinnedTransferQueue::Operation operation(d.transfer);
 d.preparedResponse=input.response;d.preparedParticles=input.particleResident;
 const int n=input.particleResident.enabled?input.particleResident.cells:(input.resident.enabled?input.resident.cells:int(input.temperature.size()));
 const int faces=input.particleResident.enabled?input.particleResident.boundaryFaces:(input.resident.enabled?input.resident.boundaryFaces:int(input.boundaryArea.size()));
 auto v=input.resident;
 if(input.particleResident.enabled){const auto& r=input.particleResident;checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(r.producerStream)));checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));
  d.volume.upload(input.volume,d.transfer);d.mass.reserve(n);d.temperature.reserve(n);d.diameter.reserve(n);d.epsilon.reserve(n);
  const int width=workPlan().radiationBlockThreads;if(n&&!d.preparingParent)anchorAggregate<<<(n+width-1)/width,width,0,d.stream>>>(r,n,d.volume.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr,d.epsilon.ptr,d.optical.deviceTable(),input.activeEpsilonMinimum);
  v.enabled=true;v.cells=n;v.boundaryFaces=faces;v.epsilon=d.epsilon.ptr;v.mass=d.mass.ptr;v.temperature=d.temperature.ptr;v.diameter=d.diameter.ptr;v.volume=d.volume.ptr;v.producerStream=d.stream;v.diagnostics=r.diagnostics;
 }else if(v.enabled){checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(v.producerStream)));checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));
  if(!v.mass){d.mass.upload(input.mass,d.transfer);v.mass=d.mass.ptr;}
 }else{d.epsilon.upload(input.epsilon,d.transfer);d.mass.upload(input.mass,d.transfer);d.temperature.upload(input.temperature,d.transfer);d.diameter.upload(input.diameter,d.transfer);d.volume.upload(input.volume,d.transfer);
  v.enabled=true;v.cells=n;v.boundaryFaces=faces;v.epsilon=d.epsilon.ptr;v.mass=d.mass.ptr;v.temperature=d.temperature.ptr;v.diameter=d.diameter.ptr;v.volume=d.volume.ptr;v.producerStream=d.stream;}
 d.diagnostics=v.diagnostics;d.coupled.upload(input.coupledBoundary,d.transfer);d.coupledHost=input.coupledBoundary;
 d.next.reserve(n);d.energy.reserve(n);d.actualEnergy.reserve(n);d.entryB.reserve(n);d.slope.reserve(n);d.plusSlope.reserve(n);d.minusSlope.reserve(n);d.linearizedEmission.reserve(n);d.thermalFeedback.reserve(n);d.materialResidual.reserve(n);d.materialRelativeResidual.reserve(n);d.finalFraction.reserve(n);d.searchRequest.reserve(n);d.searchActual.reserve(n);d.searchTemperature.reserve(n);d.materialValid.reserve(n);d.trialInDomain.reserve(n);d.maximumMaterialResidual.reserve(1);d.temperatureGridAllowance.reserve(n);d.materialEffectiveRelative.reserve(n);d.materialAllowedResidual.reserve(n);d.usedTemperatureGrid.reserve(n);
 d.raw.reserve(faces);d.boundary.reserve(faces);d.escaped.reserve(1);d.anchorOffsets.reserve(n+1);d.anchorDomains.reserve(n);
 const int reductionWidth=std::max(2,workPlan().radiationBlockThreads);
 const int maximumBlocks=(n+reductionWidth-1)/reductionWidth;d.maximumA.reserve(maximumBlocks);d.maximumB.reserve(maximumBlocks);
 d.anchorCutoff=input.activeEpsilonMinimum;d.anchorParcels=input.response.enabled;d.anchorCp=input.heatFactor;d.anchorMinimum=input.minimumTemperature;d.anchorMaximum=input.maximumTemperature;d.anchorDt=RadiationReal(input.elapsed);d.anchorState=v;d.cells=n;d.faces=faces;
 const int width=workPlan().radiationBlockThreads;
 const bool borrowEntries=input.response.enabled&&d.preparingParent;
 if(input.response.enabled){
  if(borrowEntries)d.anchors.clear();else d.anchors.reserve(std::max(1,input.response.capacity));
  if(!input.response.cellOffsets||input.response.deviceView)d.buildResponseIndex(input.response,n);
  if(!d.preparingParent){
   anchorDirectory<<<(n+1+width-1)/width,width,0,d.stream>>>(input.response,n,d.responseOffsets.ptr,d.anchorOffsets.ptr);
   if(input.response.capacity)captureAnchoredRadiationEntriesKernel<<<(input.response.capacity+width-1)/width,width,0,d.stream>>>(
       input.response,d.anchorOffsets.ptr,d.responseIndices.ptr,n,d.anchors.ptr,nullptr);
  }
 }else{
  d.anchors.reserve(std::max(1,n));
  if(!d.preparingParent)anchorLumped<<<(n+1+width-1)/width,width,0,d.stream>>>(n,v.mass,v.temperature,d.anchorOffsets.ptr,d.anchors.ptr);
 }
 if(n&&!d.preparingParent){anchorInitialize<<<(n+width-1)/width,width,0,d.stream>>>(n,d.anchorParcels,d.anchorCp,d.anchors.ptr,d.anchorOffsets.ptr,d.anchorMinimum,d.anchorMaximum,d.anchorDt,d.optical.deviceTable(),v,d.arrays());}
 if(!d.preparingParent)checkedMaterial(cudaMemsetAsync(d.maximumMaterialResidual.ptr,0,sizeof(RadiationReal),d.stream));checkedMaterial(cudaGetLastError());d.anchorActive=true;d.resultReady=false;


 d.windowDescriptor.reserve(1);
 AnchorWindowDescriptor window{n,d.anchorParcels,d.anchorCp,d.anchorMinimum,d.anchorMaximum,d.anchorDt,d.anchors.ptr,d.anchorOffsets.ptr,d.optical.deviceTable(),d.arrays(),d.deviceElapsed,d.diagnostics};
 if(borrowEntries){window.residentEntries=input.response;window.residentIndices=d.responseIndices.ptr;}
 d.transfer.upload(d.windowDescriptor.ptr,&window,sizeof(window));
 const std::vector<const void*> pointers{
  d.windowDescriptor.ptr,d.anchors.ptr,d.anchorOffsets.ptr,d.anchorDomains.ptr,
  d.energy.ptr,d.actualEnergy.ptr,d.next.ptr,d.entryB.ptr,d.slope.ptr,d.plusSlope.ptr,d.minusSlope.ptr,
  d.linearizedEmission.ptr,d.thermalFeedback.ptr,d.materialResidual.ptr,d.materialRelativeResidual.ptr,
  d.maximumMaterialResidual.ptr,d.finalFraction.ptr,d.searchRequest.ptr,d.searchActual.ptr,d.searchTemperature.ptr,
  d.materialValid.ptr,d.trialInDomain.ptr,d.temperatureGridAllowance.ptr,d.materialEffectiveRelative.ptr,
  d.materialAllowedResidual.ptr,d.usedTemperatureGrid.ptr,d.maximumA.ptr,d.maximumB.ptr};
 if(pointers!=d.graphPointers||n!=d.graphCells||width!=d.graphWidth){
  ++d.graphRevision;d.graphPointers=pointers;d.graphCells=n;d.graphWidth=width;
 }
 RadiationThermalCouplingDeviceView out{};out.graphOwner=d.graphOwner;out.graphRevision=d.graphRevision;out.trialMeanTemperature=d.next.ptr;out.linearizedEmission=d.linearizedEmission.ptr;out.thermalFeedback=d.thermalFeedback.ptr;out.materialRelativeResidual=d.maximumMaterialResidual.ptr;out.materialRelativeTolerance=d.controls.boundedEnergyRelativeTolerance;out.producerStream=d.stream;return out;
}
RadiationThermalCouplingDeviceView ParticleRadiationMaterial::prepareParentWindow(const ParticleRadiationMaterialInput& input,const double* elapsed){
 awaitCompletion();auto& d=*data_;d.externalCompletion={};d.deviceElapsed=elapsed;d.preparingParent=true;
 try{const auto result=beginAnchoredWindow(input);d.preparingParent=false;d.parentPrepared=true;return result;}
 catch(...){d.preparingParent=false;throw;}
}
ExecutionGraphNode ParticleRadiationMaterial::appendWindowStartTo(ExecutionGraph graph,ExecutionGraphNode dependency){
 auto& d=*data_;if(!d.parentPrepared)throw std::logic_error("prepare parent material buffers before graph construction");
 const int width=workPlan().radiationBlockThreads,n=d.cells;auto tail=dependency;
 if(d.preparedResponse.enabled&&(!d.preparedResponse.cellOffsets||d.preparedResponse.deviceView))
  tail=d.appendResponseIndexTo(graph,tail,d.preparedResponse,n);
 return executionGraphDetail::appendCaptured(graph,tail?std::vector<cudaGraphNode_t>{tail}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
  if(d.preparedParticles.enabled&&n)anchorAggregate<<<(n+width-1)/width,width,0,d.stream>>>(d.preparedParticles,n,d.volume.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr,d.epsilon.ptr,d.optical.deviceTable(),d.anchorCutoff);
  if(d.preparedResponse.enabled){
   anchorDirectory<<<(n+1+width-1)/width,width,0,d.stream>>>(d.preparedResponse,n,d.responseOffsets.ptr,d.anchorOffsets.ptr);
  }else anchorLumped<<<(n+1+width-1)/width,width,0,d.stream>>>(n,d.anchorState.mass,d.anchorState.temperature,d.anchorOffsets.ptr,d.anchors.ptr);
  if(n)anchorInitializeWindow<<<(n+width-1)/width,width,0,d.stream>>>(d.windowDescriptor.ptr,d.anchorState);
  checkedMaterial(cudaMemsetAsync(d.maximumMaterialResidual.ptr,0,sizeof(RadiationReal),d.stream));checkedMaterial(cudaGetLastError());
 });
}
ExecutionGraphNode ParticleRadiationMaterial::appendWindowFinishTo(ExecutionGraph graph,ExecutionGraphNode dependency,const RadiationDeviceView& transport){
 auto& d=*data_;const int width=workPlan().radiationBlockThreads;
 return executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},d.stream,[&]{
  if(d.faces)rawBoundaryEnergy<<<(d.faces+width-1)/width,width,0,d.stream>>>(d.faces,d.anchorDt,transport.boundaryArea,transport.boundaryFlux,d.raw.ptr,d.diagnostics,d.deviceElapsed);
  publishMaterialBoundary<<<1,1,0,d.stream>>>(d.faces,d.raw.ptr,d.coupled.ptr,d.boundary.ptr,d.escaped.ptr);checkedMaterial(cudaGetLastError());
 });
}
RadiationMaterialDeviceView ParticleRadiationMaterial::deviceView()const noexcept{
 const auto& d=*data_;RadiationMaterialDeviceView out{d.anchorActive,d.cells,d.faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};
 if(d.anchorActive)out.publicationFraction=d.finalFraction.ptr;return out;
}
void ParticleRadiationMaterial::acceptExternalCompletion(const std::shared_future<void>& completion){data_->externalCompletion=completion;data_->resultReady=true;}
void ParticleRadiationMaterial::acceptExternalExecution(void* producerStream){
 auto& d=*data_;checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(producerStream)));checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));
}
void ParticleRadiationMaterial::awaitCompletion()const{if(data_->externalCompletion.valid())data_->externalCompletion.get();}
void* ParticleRadiationMaterial::producerStream()const noexcept{return data_->stream;}
ExecutionGraphNode ParticleRadiationMaterial::appendAnchoredIteration(ExecutionGraph graph,ExecutionGraphNode dependency,void* producer,const RadiationThermalIterationDeviceView& view){
 auto& d=*data_;const auto stream=static_cast<cudaStream_t>(producer);const int width=workPlan().radiationBlockThreads;
 return executionGraphDetail::appendCaptured(graph,dependency?std::vector<cudaGraphNode_t>{dependency}:std::vector<cudaGraphNode_t>{},stream,[&](){
  if(d.cells)anchorUpdate<<<(d.cells+width-1)/width,width,0,stream>>>(d.windowDescriptor.ptr,view);
  enqueueOrderedMaximum(d.cells,d.materialEffectiveRelative.ptr,nullptr,0,d.maximumMaterialResidual.ptr,
      d.maximumA.ptr,d.maximumB.ptr,width,stream);
  checkedMaterial(cudaGetLastError());});
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::finishAnchoredWindow(const RadiationDeviceView& transport,bool exportFields){
 auto& d=*data_;checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(transport.producerStream)));checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));const int width=workPlan().radiationBlockThreads;
 if(d.faces)rawBoundaryEnergy<<<(d.faces+width-1)/width,width,0,d.stream>>>(d.faces,d.anchorDt,transport.boundaryArea,transport.boundaryFlux,d.raw.ptr,d.diagnostics);
 publishMaterialBoundary<<<1,1,0,d.stream>>>(d.faces,d.raw.ptr,d.coupled.ptr,d.boundary.ptr,d.escaped.ptr);
 checkedMaterial(cudaGetLastError());d.resultReady=true;
 if(exportFields)return collect();ParticleRadiationMaterialResult out;out.resident={true,d.cells,d.faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};out.resident.publicationFraction=d.finalFraction.ptr;return out;
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::solve(const ParticleRadiationMaterialInput& input,bool exportFields){
    awaitCompletion();auto& d=*data_;d.externalCompletion={};d.parentPrepared=false;d.anchorActive=false;PinnedTransferQueue::Operation operation(d.transfer);
    const int cells=input.resident.enabled?input.resident.cells:input.temperature.size();
    const int faces=input.resident.enabled?input.resident.boundaryFaces:input.boundaryArea.size();
    auto state=input.resident;d.diagnostics=state.enabled?state.diagnostics:nullptr;
    if(state.enabled){
        if(input.coupledBoundary.size()!=std::size_t(faces))throw std::invalid_argument("resident radiation material boundary shape mismatch");
        checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(state.producerStream)));checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));
        if(!state.mass){d.mass.upload(input.mass,d.transfer);state.mass=d.mass.ptr;}
    }else{
        d.epsilon.upload(input.epsilon,d.transfer);d.mass.upload(input.mass,d.transfer);d.temperature.upload(input.temperature,d.transfer);
        d.diameter.upload(input.diameter,d.transfer);d.incident.upload(input.incident,d.transfer);d.volume.upload(input.volume,d.transfer);
        d.area.upload(input.boundaryArea,d.transfer);d.flux.upload(input.boundaryFlux,d.transfer);
        state={true,cells,faces,d.epsilon.ptr,d.mass.ptr,d.temperature.ptr,d.diameter.ptr,d.incident.ptr,nullptr,d.volume.ptr,d.area.ptr,d.flux.ptr,d.stream};
    }
    if(d.coupledHost!=input.coupledBoundary){d.coupled.upload(input.coupledBoundary,d.transfer);d.coupledHost=input.coupledBoundary;}
    d.next.reserve(cells);d.energy.reserve(cells);d.raw.reserve(faces);d.boundary.reserve(faces);d.escaped.reserve(1);
    const int* offsets=input.response.cellOffsets;const int* indices=nullptr;
    if(input.response.enabled&&(!offsets||input.response.deviceView)){



        d.buildResponseIndex(input.response,cells);
        offsets=d.responseOffsets.ptr;indices=d.responseIndices.ptr;
    }
    if(cells)materialRoots<<<(cells+ugkp::gpu::workPlan().radiationBlockThreads-1)/ugkp::gpu::workPlan().radiationBlockThreads,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(cells,d.optical.deviceTable(),state.epsilon,state.mass,state.temperature,state.diameter,
        state.incident,state.volume,input.heatFactor,input.elapsed,input.minimumTemperature,input.maximumTemperature,input.activeEpsilonMinimum,d.controls,input.response,offsets,indices,d.next.ptr,d.energy.ptr,d.diagnostics);
    if(faces)rawBoundaryEnergy<<<(faces+ugkp::gpu::workPlan().radiationBlockThreads-1)/ugkp::gpu::workPlan().radiationBlockThreads,ugkp::gpu::workPlan().radiationBlockThreads,0,d.stream>>>(faces,input.elapsed,state.boundaryArea,state.boundaryFlux,d.raw.ptr,d.diagnostics);
    publishMaterialBoundary<<<1,1,0,d.stream>>>(faces,d.raw.ptr,d.coupled.ptr,d.boundary.ptr,d.escaped.ptr);
    checkedMaterial(cudaGetLastError());
    if(input.resident.enabled){
        checkedMaterial(cudaEventRecord(d.inputReady,d.stream));
        checkedMaterial(cudaStreamWaitEvent(static_cast<cudaStream_t>(input.resident.producerStream),d.inputReady,0));
    }
    d.cells=cells;d.faces=faces;d.resultReady=true;
    if(exportFields)return collect();
    ParticleRadiationMaterialResult result;result.resident={true,cells,faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};if(d.anchorActive)result.resident.publicationFraction=d.finalFraction.ptr;return result;
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::collect(){
    awaitCompletion();auto& d=*data_;if(!d.resultReady)throw std::logic_error("radiation material output requested before solve");
    PinnedTransferQueue::Operation operation(d.transfer);const int cells=d.cells,faces=d.faces;
    ParticleRadiationMaterialResult result;result.temperature.resize(cells);result.energy.resize(cells);result.boundaryEnergy.resize(faces);
    d.transfer.download(result.temperature.data(),d.next.ptr,cells*sizeof(RadiationReal));d.transfer.download(result.energy.data(),d.energy.ptr,cells*sizeof(RadiationReal));
    d.transfer.download(result.boundaryEnergy.data(),d.boundary.ptr,faces*sizeof(RadiationReal));
    d.transfer.download(&result.escapedEnergy,d.escaped.ptr,sizeof(RadiationReal));d.transfer.finish();result.resident={true,cells,faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};if(d.anchorActive)result.resident.publicationFraction=d.finalFraction.ptr;return result;
}

RadiationReal ParticleRadiationMaterial::collectEscapedEnergy(){
    awaitCompletion();auto& d=*data_;
    if(!d.resultReady)throw std::logic_error("radiation energy report requested before solve");
    PinnedTransferQueue::Operation operation(d.transfer);RadiationReal result=0;
    d.transfer.download(&result,d.escaped.ptr,sizeof(result));d.transfer.finish();return result;
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::collectBoundaryEnergy(){
    awaitCompletion();auto& d=*data_;if(!d.resultReady)throw std::logic_error("radiation boundary output requested before solve");
    PinnedTransferQueue::Operation operation(d.transfer);ParticleRadiationMaterialResult result;
    result.boundaryEnergy.resize(d.faces);
    d.transfer.download(result.boundaryEnergy.data(),d.boundary.ptr,d.faces*sizeof(RadiationReal));

    d.transfer.download(&result.escapedEnergy,d.escaped.ptr,sizeof(RadiationReal));

    d.transfer.finish();result.resident={true,d.cells,d.faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};if(d.anchorActive)result.resident.publicationFraction=d.finalFraction.ptr;return result;
}
}}
