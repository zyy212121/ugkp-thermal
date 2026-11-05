#include "MieOpticalProperties.H"
#include "PinnedTransferQueue.cuh"
#include "MiePhaseBalance.cuh"
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA Mie optics: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t n){if(n<=capacity)return;T* next=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&next),n*sizeof(T)));if(ptr)cudaFree(ptr);ptr=next;capacity=n;}
    void upload(const std::vector<T>& input,PinnedTransferQueue& queue){reserve(input.size());queue.upload(ptr,input.data(),input.size()*sizeof(T));}
};
__global__ void evaluateOptics(int count,MieOpticalTableView table,const MieOpticalQuery* input,MieOpticalResult* output){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=mieOpticalProperties(table,input[i]);
}

}
struct MieOpticalProperties::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;
    Array<double> temperature,diameter,mu,band,absorption,scattering,extinction,phase;
    Array<double> weights,cosines,scaling,nextScaling,rowError,columnError;
    Array<MieBracket> phaseBrackets;Array<MiePhaseMetrics> phaseMetrics;
    Array<MieOpticalQuery> queries;Array<MieOpticalResult> results;MieOpticalTableView table;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
MieOpticalProperties::MieOpticalProperties(const MieOpticalTable& table):data_(new Implementation){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    d.temperature.upload(table.temperature,d.transfers);d.diameter.upload(table.diameter,d.transfers);
    d.mu.upload(table.mu,d.transfers);d.band.upload(table.band,d.transfers);
    d.absorption.upload(table.absorption,d.transfers);d.scattering.upload(table.scattering,d.transfers);
    d.extinction.upload(table.extinction,d.transfers);d.phase.upload(table.phase,d.transfers);
    d.table={int(table.temperature.size()),int(table.diameter.size()),int(table.mu.size()),d.temperature.ptr,d.diameter.ptr,
        d.mu.ptr,d.band.ptr,d.absorption.ptr,d.scattering.ptr,d.extinction.ptr,d.phase.ptr};d.transfers.finish();
}
MieOpticalProperties::~MieOpticalProperties()=default;
MieOpticalTableView MieOpticalProperties::deviceTable()const noexcept{return data_->table;}
void MieOpticalProperties::balance(const std::vector<MieOpticalQuery>& input,const std::vector<double>& weights,
    const std::vector<double>& cosines,double tolerance,int maximumIterations,std::vector<double>& scaling,
    std::vector<MiePhaseMetrics>& metrics){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);
    const int directions=weights.size();const std::size_t entries=input.size()*weights.size();
    d.queries.upload(input,d.transfers);d.weights.upload(weights,d.transfers);d.cosines.upload(cosines,d.transfers);
    d.phaseBrackets.reserve(cosines.size());d.scaling.reserve(entries);d.nextScaling.reserve(entries);
    d.rowError.reserve(entries);d.columnError.reserve(entries);d.phaseMetrics.reserve(input.size());
    if(!cosines.empty())mieDetail::phaseBrackets<<<(cosines.size()+127)/128,128,0,d.stream>>>(cosines.size(),d.table,d.cosines.ptr,d.phaseBrackets.ptr);
    scaling.resize(entries);metrics.resize(input.size());
    if(!input.empty()){
        mieDetail::balancePhase<<<input.size(),128,0,d.stream>>>(input.size(),directions,d.table,d.queries.ptr,d.weights.ptr,
            d.phaseBrackets.ptr,tolerance,maximumIterations,d.scaling.ptr,d.nextScaling.ptr,d.rowError.ptr,d.columnError.ptr,d.phaseMetrics.ptr);
        d.transfers.download(scaling.data(),d.scaling.ptr,entries*sizeof(double));
        d.transfers.download(metrics.data(),d.phaseMetrics.ptr,metrics.size()*sizeof(MiePhaseMetrics));
    }
    d.transfers.finish();
}
void MieOpticalProperties::evaluate(const std::vector<MieOpticalQuery>& input,std::vector<MieOpticalResult>& output){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);d.queries.upload(input,d.transfers);d.results.reserve(input.size());output.resize(input.size());
    if(!input.empty()){evaluateOptics<<<(input.size()+127)/128,128,0,d.stream>>>(input.size(),d.table,d.queries.ptr,d.results.ptr);
        d.transfers.download(output.data(),d.results.ptr,output.size()*sizeof(MieOpticalResult));}
    d.transfers.finish();
}
}}
