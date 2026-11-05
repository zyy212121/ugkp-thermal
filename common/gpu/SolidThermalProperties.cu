#include "SolidThermalProperties.H"
#include <cuda_runtime.h>
#include "PinnedTransferQueue.cuh"
#include <stdexcept>
#include <string>
namespace ugkp { namespace gpu { namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA solid properties: ")+cudaGetErrorString(code));}
template<class T> struct Array {
    T* ptr=nullptr;std::size_t capacity=0;
    ~Array(){if(ptr)cudaFree(ptr);}
    void reserve(std::size_t count){if(count<=capacity)return;T* replacement=nullptr;checked(cudaMalloc(reinterpret_cast<void**>(&replacement),sizeof(T)*count));if(ptr)cudaFree(ptr);ptr=replacement;capacity=count;}
    void upload(const T* host,std::size_t count,PinnedTransferQueue& queue){reserve(count);queue.upload(ptr,host,count*sizeof(T));}
};
__global__ void fixedPointResidual(int n,const double* current,const double* previous,double* residual){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)residual[i]=current[i]-previous[i];
}
__global__ void aitkenRelaxation(int n,const double* residual,const double* previous,bool havePrevious,
    double relaxation,double minimum,double maximum,double small,double* result){
    if(blockIdx.x)return;
    if(!havePrevious){if(!threadIdx.x)*result=relaxation;return;}
    __shared__ double products[256],squares[256];
    double numerator=0,denominator=0;
    // Coalesced reads and products, followed by the unchanged ordered sum.
    for(int first=0;first<n;first+=256){
        const int i=first+threadIdx.x;
        if(i<n){const double difference=residual[i]-previous[i];
            products[threadIdx.x]=previous[i]*difference;squares[threadIdx.x]=difference*difference;}
        __syncthreads();
        if(!threadIdx.x)for(int j=0;j<256&&first+j<n;++j){numerator+=products[j];denominator+=squares[j];}
        __syncthreads();
    }
    if(threadIdx.x)return;
    double candidate=relaxation;
    if(std::isfinite(denominator)&&denominator>small){
        const double value=-relaxation*numerator/denominator;
        if(std::isfinite(value))candidate=value;
    }
    candidate=candidate<minimum?minimum:candidate;
    *result=candidate<maximum?candidate:maximum;
}
__global__ void applyRelaxation(int n,const double* previous,const double* residual,const double* relaxation,
    double* current,double* history){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){current[i]=previous[i]+*relaxation*residual[i];history[i]=residual[i];}
}
__global__ void evaluateProperties(int n,const ThermalPropertyTable* tables,const double* knots,int count,const ThermalPropertyQuery* queries,ThermalPropertyResult* results){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)results[i]=thermalProperties(tables,knots,count,queries[i]);
}
}
struct SolidThermalProperties::Implementation {
    cudaStream_t stream=nullptr;PinnedTransferQueue transfers;int knotCount=0;
    Array<double> temperatures[4],values[4],knots;Array<ThermalPropertyTable> tables;
    Array<double> current,previous,residual,residualHistory,relaxation;
    Array<ThermalPropertyQuery> queries;Array<ThermalPropertyResult> results;
    Implementation(){checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));transfers.bind(stream);}
    ~Implementation(){transfers.discard();if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}}
};
SolidThermalProperties::SolidThermalProperties(const ThermalPropertyTable* tables,const double* knots,int count):data_(new Implementation){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);ThermalPropertyTable device[4];
    for(int i=0;i<4;++i){
        device[i]=tables[i];d.temperatures[i].upload(tables[i].temperatures,tables[i].count,d.transfers);
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
        d.transfers.download(current,d.current.ptr,count*sizeof(double));}
    d.transfers.download(&relaxation,d.relaxation.ptr,sizeof(double));
    d.transfers.finish();return relaxation;
}
void SolidThermalProperties::evaluate(const std::vector<ThermalPropertyQuery>& queries,std::vector<ThermalPropertyResult>& results){
    auto& d=*data_;PinnedTransferQueue::Operation operation(d.transfers);const int n=queries.size();d.queries.upload(queries.data(),n,d.transfers);d.results.reserve(n);results.resize(n);
    if(n){evaluateProperties<<<(n+127)/128,128,0,d.stream>>>(n,d.tables.ptr,d.knots.ptr,d.knotCount,d.queries.ptr,d.results.ptr);
        d.transfers.download(results.data(),d.results.ptr,n*sizeof(ThermalPropertyResult));}
    d.transfers.finish();
}
}}
