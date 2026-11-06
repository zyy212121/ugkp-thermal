#include "ParticleRadiationMaterial.H"
#include "../Diagnostics.H"
#include "RadiationImplicitMath.H"
#include "PinnedTransferQueue.cuh"
#include "ExecutionGraph.cuh"
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checkedMaterial(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(std::string("CUDA particle radiation: ")+cudaGetErrorString(status));}
template<class T> struct MaterialArray {
    T* ptr=nullptr;std::size_t capacity=0;
    ~MaterialArray(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t size){if(size<=capacity)return;T* next=nullptr;checkedMaterial(cudaMalloc(reinterpret_cast<void**>(&next),size*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=size;}
    void upload(const std::vector<T>& values,PinnedTransferQueue& transfer){reserve(values.size());transfer.upload(ptr,values.data(),values.size()*sizeof(T));}
};
constexpr double fourPiMaterial=12.56637061435917295385057353311801153679;
__host__ __device__ double materialBlack(double t,double band){const double square=t*t;return band*5.670374419e-8*square*square/(fourPiMaterial/4);}
struct MaterialBalance {
    MieOpticalTableView table;double diameter,incident;
    __host__ __device__ double operator()(double temperature)const{return incident-fourPiMaterial*materialBlack(temperature,mieBand(table,temperature));}
};
struct MaterialPower {
    MaterialBalance balance;double epsilon,volume;
    __host__ __device__ double magnitude(double temperature)const{
        const auto d=mieLogBracket(balance.table.diameter,balance.table.diameters,balance.diameter);
        const auto t=mieLinearBracket(balance.table.temperature,balance.table.temperatures,temperature);
        const double absorption=3*epsilon*mieBilinear(balance.table.absorption,balance.table.temperatures,d,t)/(2*balance.diameter);
        return fabs(volume*absorption)*(fabs(balance.incident)+fabs(fourPiMaterial*materialBlack(temperature,mieBand(balance.table,temperature))));
    }
    __host__ __device__ double operator()(double temperature)const{
        const auto d=mieLogBracket(balance.table.diameter,balance.table.diameters,balance.diameter);
        const auto t=mieLinearBracket(balance.table.temperature,balance.table.temperatures,temperature);
        const double absorption=3*epsilon*mieBilinear(balance.table.absorption,balance.table.temperatures,d,t)/(2*balance.diameter);
        // The original zero-absorption short circuit is significant for IEEE values.
        return volume*(absorption==0?0:absorption*balance(temperature));
    }
};
__global__ void countResponseParticles(ParticleRadiationResponseView view,int cells,int* counts,diagnostics::DeviceContext* context){
    if(diagnostics::hasFatal(context))return;
    view=resolveParticleRadiationResponse(view);
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<*view.count;i+=blockDim.x*gridDim.x){
        if(view.status[i]!=1||view.wallState[i]!=0)continue;
        const int c=view.cell[i];if(c<0||c>=cells){asm("trap;");return;}
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
__global__ void materialRoots(int count,MieOpticalTableView table,const double* epsilon,const double* mass,const double* temperature,
    const double* diameter,const double* incident,const double* volume,double heatFactor,double elapsed,double minimum,double maximum,double cutoff,
    ParticleRadiationResponseView response,const int* offsets,const int* indices,double* next,double* energy,double* magnitude,diagnostics::DeviceContext* context){
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;if(cell>=count)return;
    response=resolveParticleRadiationResponse(response);
    if(response.cellOffsets){offsets=response.cellOffsets;indices=nullptr;}
    const double old=temperature[cell];next[cell]=old;energy[cell]=0;magnitude[cell]=0;
    if(diagnostics::hasFatal(context)||!(epsilon[cell]>cutoff))return;
    const double capacity=mass[cell]*heatFactor;
    const MaterialBalance balance{table,diameter[cell],incident[cell]};const MaterialPower power{balance,epsilon[cell],volume[cell]};
    if(response.enabled){
        magnitude[cell]=fabs(elapsed)*power.magnitude(old);
        const IndexedParticleRadiationResponse caloric{response,indices,offsets[cell],offsets[cell+1]};
        const auto root=solveCaloricParticleRadiation(caloric,power,elapsed);
        next[cell]=root.temperature;energy[cell]=root.energy;return;
    }
    const auto root=radiationMath::solveBoundedImplicitMaterialTemperature(old,minimum,maximum,capacity,elapsed,100,1e-10,1e-9,balance,power);
    next[cell]=root.temperatureK;energy[cell]=capacity*(root.temperatureK-old);
}
__global__ void rawBoundaryEnergy(int count,double elapsed,const double* area,const double* flux,double* energy,diagnostics::DeviceContext* context){
    const int face=blockIdx.x*blockDim.x+threadIdx.x;if(face<count)energy[face]=diagnostics::hasFatal(context)?0:elapsed*area[face]*flux[face];
}
__global__ void conservativeMaterialBoundary(int cells,int faces,const double* accepted,const double* raw,const unsigned char* coupled,
    const double* magnitude,double* magnitudeTotal,double* boundary,double* escaped){
    if(threadIdx.x)return;
    // These are the native ordered transaction sums, deliberately unreordered.
    double acceptedTotal=0,gross=0,rawTotal=0,radiativeMagnitude=0;
    for(int cell=0;cell<cells;++cell){acceptedTotal+=accepted[cell];gross+=fabs(accepted[cell]);radiativeMagnitude+=magnitude[cell];}
    for(int face=0;face<faces;++face){rawTotal+=raw[face];gross+=fabs(raw[face]);}
    const auto scaling=radiationMath::conservativeBoundaryEnergyScale(acceptedTotal,rawTotal,gross,1e-10,radiativeMagnitude);
    *magnitudeTotal=radiativeMagnitude;
    double other=0;
    for(int face=0;face<faces;++face){const double value=scaling.scale*raw[face];boundary[face]=0;if(coupled[face])boundary[face]=value;else other+=value;}
    *escaped=other;
}
__global__ void settleMaterialBoundary(int cells,int faces,const double* actual,double* boundary,
    double* escaped,const double* magnitude,double* acceptedTotal,diagnostics::DeviceContext* context){
    if(threadIdx.x)return;
    if(diagnostics::hasFatal(context)){
        for(int face=0;face<faces;++face)boundary[face]=0;
        *escaped=0;*acceptedTotal=0;return;
    }
    // Preserve the transaction's ordered sums and its escaped-boundary grouping.
    double particleHeat=0,boundaryHeat=*escaped,gross=fabs(boundaryHeat);
    for(int cell=0;cell<cells;++cell){const double q=actual?actual[cell]:0;particleHeat+=q;gross+=fabs(q);}
    for(int face=0;face<faces;++face){boundaryHeat+=boundary[face];gross+=fabs(boundary[face]);}
    const auto scale=radiationMath::conservativeBoundaryEnergyScale(particleHeat,boundaryHeat,gross,1e-10,*magnitude);
    for(int face=0;face<faces;++face)boundary[face]*=scale.scale;
    *escaped*=scale.scale;*acceptedTotal=particleHeat;
}
}
struct ParticleRadiationMaterial::Implementation {
    MieOpticalProperties optical;cudaStream_t stream=nullptr;cudaEvent_t inputReady=nullptr;PinnedTransferQueue transfer;
    MaterialArray<double> epsilon,mass,temperature,diameter,incident,volume,area,flux,next,energy,raw,boundary,escaped,magnitude,magnitudeTotal,acceptedTotal;
    MaterialArray<unsigned char> coupled,scanScratch;
    std::vector<unsigned char> coupledHost;
    int cells=0,faces=0;bool resultReady=false,settled=false;
    diagnostics::DeviceContext* diagnostics=nullptr;
    MaterialArray<int> responseCounts,responseOffsets,responseCursor,responseIndices;
    cudaGraph_t responseGraph=nullptr;cudaGraphExec_t responseExecutable=nullptr;
    const ParticleRadiationResponseView* responseSource=nullptr;
    diagnostics::DeviceContext* responseDiagnostics=nullptr;
    int responseCells=-1,responseCapacity=-1;
    void releaseResponseGraph(){
        if(responseExecutable){checkedMaterial(cudaGraphExecDestroy(responseExecutable));responseExecutable=nullptr;}
        if(responseGraph){checkedMaterial(cudaGraphDestroy(responseGraph));responseGraph=nullptr;}
    }
    void buildResponseIndex(ParticleRadiationResponseView response,int count){
        const auto* oldCounts=responseCounts.ptr;const auto* oldOffsets=responseOffsets.ptr;
        const auto* oldCursor=responseCursor.ptr;const auto* oldIndices=responseIndices.ptr;const auto* oldScratch=scanScratch.ptr;
        responseCounts.reserve(count+1);responseOffsets.reserve(count+1);responseCursor.reserve(count+1);
        responseIndices.reserve(response.capacity);
        std::size_t bytes=0;
        checkedMaterial(cub::DeviceScan::ExclusiveSum(nullptr,bytes,responseCounts.ptr,responseOffsets.ptr,count+1,stream));
        scanScratch.reserve(bytes);
        if(oldCounts!=responseCounts.ptr||oldOffsets!=responseOffsets.ptr||oldCursor!=responseCursor.ptr
            ||oldIndices!=responseIndices.ptr||oldScratch!=scanScratch.ptr)releaseResponseGraph();
        const auto submit=[&](){
            checkedMaterial(cudaMemsetAsync(responseCounts.ptr,0,(count+1)*sizeof(int),stream));
            const int blocks=(response.capacity+127)/128;
            if(blocks)countResponseParticles<<<blocks,128,0,stream>>>(response,count,responseCounts.ptr,diagnostics);
            checkedMaterial(cub::DeviceScan::ExclusiveSum(scanScratch.ptr,bytes,responseCounts.ptr,responseOffsets.ptr,count+1,stream));
            checkedMaterial(cudaMemcpyAsync(responseCursor.ptr,responseOffsets.ptr,(count+1)*sizeof(int),cudaMemcpyDeviceToDevice,stream));
            if(blocks)indexResponseParticles<<<blocks,128,0,stream>>>(response,responseCursor.ptr,responseIndices.ptr,diagnostics);
            checkedMaterial(cudaGetLastError());
        };
        if(!response.deviceView){submit();return;}
        if(!responseExecutable||responseSource!=response.deviceView||responseCells!=count
            ||responseCapacity!=response.capacity||responseDiagnostics!=diagnostics){
            releaseResponseGraph();checkedMaterial(cudaGraphCreate(&responseGraph,0));
            cudaGraphConditionalHandle condition;checkedMaterial(cudaGraphConditionalHandleCreate(&condition,responseGraph,0,0));
            const auto select=executionGraphDetail::appendCaptured(responseGraph,{},stream,[&](){
                selectResponseIndex<<<1,1,0,stream>>>(response.deviceView,condition,diagnostics);
                checkedMaterial(cudaGetLastError());
            });
            cudaGraph_t body=nullptr;executionGraphDetail::addIf(responseGraph,condition,{select},body);
            executionGraphDetail::appendCaptured(body,{},stream,submit);
            checkedMaterial(cudaGraphInstantiateWithFlags(&responseExecutable,responseGraph,0));
            responseSource=response.deviceView;responseCells=count;responseCapacity=response.capacity;responseDiagnostics=diagnostics;
        }
        checkedMaterial(cudaGraphLaunch(responseExecutable,stream));
    }
    explicit Implementation(const MieOpticalTable& table):optical(table){checkedMaterial(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfer.bind(stream);checkedMaterial(cudaEventCreateWithFlags(&inputReady,cudaEventDisableTiming));}
    ~Implementation(){transfer.discard();if(responseExecutable)cudaGraphExecDestroy(responseExecutable);if(responseGraph)cudaGraphDestroy(responseGraph);if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}if(inputReady)cudaEventDestroy(inputReady);}
};
ParticleRadiationMaterial::ParticleRadiationMaterial(const MieOpticalTable& table):data_(new Implementation(table)){}
ParticleRadiationMaterial::~ParticleRadiationMaterial()=default;
ParticleRadiationMaterialResult ParticleRadiationMaterial::solve(const ParticleRadiationMaterialInput& input,bool exportFields){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfer);
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
    d.next.reserve(cells);d.energy.reserve(cells);d.raw.reserve(faces);d.boundary.reserve(faces);d.escaped.reserve(1);d.magnitude.reserve(cells);d.magnitudeTotal.reserve(1);d.acceptedTotal.reserve(1);
    const int* offsets=input.response.cellOffsets;const int* indices=nullptr;
    if(input.response.enabled&&(!offsets||input.response.deviceView)){
        // The owner may have swapped its parcel arrays since the host acquired
        // this view. A resident IF builds an index only when that current view
        // has no direct cell directory; the common packed path does no scan.
        d.buildResponseIndex(input.response,cells);
        offsets=d.responseOffsets.ptr;indices=d.responseIndices.ptr;
    }
    if(cells)materialRoots<<<(cells+127)/128,128,0,d.stream>>>(cells,d.optical.deviceTable(),state.epsilon,state.mass,state.temperature,state.diameter,
        state.incident,state.volume,input.heatFactor,input.elapsed,input.minimumTemperature,input.maximumTemperature,input.activeEpsilonMinimum,input.response,offsets,indices,d.next.ptr,d.energy.ptr,d.magnitude.ptr,d.diagnostics);
    if(faces)rawBoundaryEnergy<<<(faces+127)/128,128,0,d.stream>>>(faces,input.elapsed,state.boundaryArea,state.boundaryFlux,d.raw.ptr,d.diagnostics);
    conservativeMaterialBoundary<<<1,1,0,d.stream>>>(cells,faces,d.energy.ptr,d.raw.ptr,d.coupled.ptr,d.magnitude.ptr,d.magnitudeTotal.ptr,d.boundary.ptr,d.escaped.ptr);
    checkedMaterial(cudaGetLastError());
    if(input.resident.enabled){
        checkedMaterial(cudaEventRecord(d.inputReady,d.stream));
        checkedMaterial(cudaStreamWaitEvent(static_cast<cudaStream_t>(input.resident.producerStream),d.inputReady,0));
    }
    d.cells=cells;d.faces=faces;d.resultReady=true;d.settled=false;
    if(exportFields)return collect();
    ParticleRadiationMaterialResult result;result.resident={true,cells,faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};return result;
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::collect(){
    auto& d=*data_;if(!d.resultReady)throw std::logic_error("radiation material output requested before solve");
    PinnedTransferQueue::Operation operation(d.transfer);const int cells=d.cells,faces=d.faces;
    ParticleRadiationMaterialResult result;result.temperature.resize(cells);result.energy.resize(cells);result.boundaryEnergy.resize(faces);
    d.transfer.download(result.temperature.data(),d.next.ptr,cells*sizeof(double));d.transfer.download(result.energy.data(),d.energy.ptr,cells*sizeof(double));
    d.transfer.download(result.boundaryEnergy.data(),d.boundary.ptr,faces*sizeof(double));d.transfer.download(&result.escapedEnergy,d.escaped.ptr,sizeof(double));d.transfer.download(&result.radiativeEnergyMagnitude,d.magnitudeTotal.ptr,sizeof(double));d.transfer.finish();result.resident={true,cells,faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};return result;
}
void ParticleRadiationMaterial::settleAcceptedEnergy(const RadiationAcceptedEnergyDeviceView& actual){
    auto& d=*data_;
    if(!d.resultReady||actual.cells!=d.cells||d.settled)throw std::logic_error("invalid radiation settlement lifecycle");
    checkedMaterial(cudaEventRecord(d.inputReady,static_cast<cudaStream_t>(actual.producerStream)));
    checkedMaterial(cudaStreamWaitEvent(d.stream,d.inputReady,0));
    settleMaterialBoundary<<<1,1,0,d.stream>>>(d.cells,d.faces,actual.energy,d.boundary.ptr,d.escaped.ptr,d.magnitudeTotal.ptr,d.acceptedTotal.ptr,d.diagnostics);
    checkedMaterial(cudaGetLastError());
    checkedMaterial(cudaEventRecord(d.inputReady,d.stream));
    checkedMaterial(cudaStreamWaitEvent(static_cast<cudaStream_t>(actual.producerStream),d.inputReady,0));
    d.settled=true;
}
ParticleRadiationMaterialResult ParticleRadiationMaterial::collectBoundaryEnergy(){
    auto& d=*data_;if(!d.settled)throw std::logic_error("radiation boundary output requested before settlement");
    PinnedTransferQueue::Operation operation(d.transfer);ParticleRadiationMaterialResult result;
    result.boundaryEnergy.resize(d.faces);
    d.transfer.download(result.boundaryEnergy.data(),d.boundary.ptr,d.faces*sizeof(double));
    d.transfer.download(&result.escapedEnergy,d.escaped.ptr,sizeof(double));
    d.transfer.download(&result.radiativeEnergyMagnitude,d.magnitudeTotal.ptr,sizeof(double));
    d.transfer.download(&result.acceptedParticleEnergy,d.acceptedTotal.ptr,sizeof(double));
    d.transfer.finish();result.resident={true,d.cells,d.faces,d.next.ptr,d.energy.ptr,d.boundary.ptr,d.stream,d.diagnostics};return result;
}
}}
