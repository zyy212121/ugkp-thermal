#ifndef CHMT_GPU_BUFFER_H
#define CHMT_GPU_BUFFER_H
#include <cuda_runtime.h>
#include <limits>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
namespace chmt {
struct CudaFault {
    std::string message;
    bool check(cudaError_t code,const char* operation){if(code==cudaSuccess)return true;
        if(message.empty()){message=std::string(operation)+": "+cudaGetErrorString(code);}
        std::fprintf(stderr,"CHMT CUDA error during %s: %s\n",operation,cudaGetErrorString(code));
        // OOM is a checked, recoverable allocation failure. Its error remains
        // recorded in this owner, but must not survive as a thread-local stale
        // launch error after an independent replacement owner is discarded.
        // Never clear a different (possibly asynchronous) runtime error here.
        if(code==cudaErrorMemoryAllocation&&cudaPeekAtLastError()==code)(void)cudaGetLastError();
        return false;}
};
class DeviceStream {
    cudaStream_t value_=nullptr;CudaFault* fault_=nullptr;
public:
    DeviceStream()=default;DeviceStream(const DeviceStream&)=delete;DeviceStream& operator=(const DeviceStream&)=delete;
    bool create(CudaFault& fault){fault_=&fault;return fault.check(cudaStreamCreateWithFlags(&value_,cudaStreamNonBlocking),"create CUDA stream");}
    bool close()noexcept{if(!value_)return true;const auto e=cudaStreamDestroy(value_);value_=nullptr;return !fault_||fault_->check(e,"destroy CUDA stream");}
    ~DeviceStream(){close();}
    cudaStream_t get()const{return value_;}
    bool sync()const{return fault_&&fault_->check(cudaStreamSynchronize(value_),"synchronize CUDA stream");}
};
template<class T>class Buffer {
    T* pointer_=nullptr;std::size_t size_=0;CudaFault* fault_=nullptr;
public:
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& other)noexcept{swap(other);}Buffer& operator=(Buffer&& other)noexcept{if(this!=&other){release();swap(other);}return *this;}
    ~Buffer(){release();}
    void swap(Buffer& other)noexcept{std::swap(pointer_,other.pointer_);std::swap(size_,other.size_);std::swap(fault_,other.fault_);}
    bool release()noexcept{bool ok=true;if(pointer_){const auto code=cudaFree(pointer_);if(fault_)ok=fault_->check(code,"free CUDA buffer");}pointer_=nullptr;size_=0;return ok;}
    bool resize(std::size_t count,CudaFault& fault){if(count==size_){fault_=&fault;return true;}if(!release())return false;fault_=&fault;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T)){fault.message="CUDA buffer size overflow";return false;}
        if(count&&!fault.check(cudaMalloc(reinterpret_cast<void**>(&pointer_),count*sizeof(T)),"allocate CUDA buffer"))return false;
        size_=count;return true;}
    bool upload(const std::vector<T>& values,CudaFault& fault){if(!resize(values.size(),fault))return false;
        return !size_||fault.check(cudaMemcpy(pointer_,values.data(),size_*sizeof(T),cudaMemcpyHostToDevice),"upload CUDA buffer");}
    bool uploadOne(const T& value,CudaFault& fault){if(!resize(1,fault))return false;return fault.check(cudaMemcpy(pointer_,&value,sizeof(T),cudaMemcpyHostToDevice),"upload CUDA scalar");}
    bool zero(cudaStream_t stream){return !size_||fault_->check(cudaMemsetAsync(pointer_,0,size_*sizeof(T),stream),"zero CUDA buffer");}
    bool copy(const Buffer& source,CudaFault& fault,cudaStream_t stream){if(!resize(source.size_,fault))return false;
        return !size_||fault.check(cudaMemcpyAsync(pointer_,source.pointer_,size_*sizeof(T),cudaMemcpyDeviceToDevice,stream),"copy CUDA candidate");}
    bool download(std::vector<T>& out,cudaStream_t stream)const{std::vector<T> value(size_);
        if(size_&&(!fault_->check(cudaMemcpyAsync(value.data(),pointer_,size_*sizeof(T),cudaMemcpyDeviceToHost,stream),"download CUDA buffer")||!fault_->check(cudaStreamSynchronize(stream),"synchronize CUDA download")))return false;
        out=std::move(value);return true;}
    bool downloadOne(T& out,cudaStream_t stream)const{if(size_!=1)return false;std::vector<T> values;if(!download(values,stream))return false;out=values[0];return true;}
    void rebindFault(CudaFault& fault)noexcept{fault_=&fault;}
    T* data()const{return pointer_;}std::size_t size()const{return size_;}
};
}
#endif
