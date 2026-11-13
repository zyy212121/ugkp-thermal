#ifndef UGKP_GPU_PINNED_TRANSFER_QUEUE_CUH
#define UGKP_GPU_PINNED_TRANSFER_QUEUE_CUH
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
namespace ugkp { namespace gpu {
// Reusable pinned staging keeps pageable host mirrors out of CUDA's implicit
// transfer synchronization. Download publication follows stream completion.
class PinnedTransferQueue {
    struct Slot {void* data=nullptr;std::size_t capacity=0;};
    struct Read {void* destination;const void* source;std::size_t bytes;std::function<void()> convert;};
    cudaStream_t stream_=nullptr;std::vector<Slot> slots_;std::vector<Read> reads_;
    std::size_t used_=0;
    static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA pinned transfer: ")+cudaGetErrorString(code));}
    void* acquire(std::size_t bytes){
        if(used_==slots_.size())slots_.push_back({});
        auto& slot=slots_[used_];
        if(slot.capacity<bytes){void* next=nullptr;check(cudaMallocHost(&next,bytes));
            if(slot.data)cudaFreeHost(slot.data);slot.data=next;slot.capacity=bytes;}
        ++used_;return slot.data;
    }
public:
    PinnedTransferQueue()=default;
    PinnedTransferQueue(const PinnedTransferQueue&)=delete;
    PinnedTransferQueue& operator=(const PinnedTransferQueue&)=delete;
    ~PinnedTransferQueue(){discard();for(auto slot:slots_)if(slot.data)cudaFreeHost(slot.data);}
    void bind(cudaStream_t stream){stream_=stream;}
    void discard()noexcept{reads_.clear();if(used_)cudaStreamSynchronize(stream_);used_=0;}
    class Operation {
        PinnedTransferQueue& queue_;
    public:
        explicit Operation(PinnedTransferQueue& queue):queue_(queue){}
        ~Operation(){queue_.discard();}
    };
    void upload(void* destination,const void* source,std::size_t bytes){
        if(!bytes)return;void* stage=acquire(bytes);std::memcpy(stage,source,bytes);
        check(cudaMemcpyAsync(destination,stage,bytes,cudaMemcpyHostToDevice,stream_));
    }
    void download(void* destination,const void* source,std::size_t bytes){
        if(!bytes)return;void* stage=acquire(bytes);reads_.push_back({destination,stage,bytes});
        check(cudaMemcpyAsync(stage,source,bytes,cudaMemcpyDeviceToHost,stream_));
    }
    // Host input/output representation is independent of the device scalar.
    // Staging owns conversion inputs until stream completion, including graphs'
    // asynchronous producer handoffs.
    template<class To,class From> void uploadConverted(To* destination,const From* source,std::size_t count){
        if(!count)return;
        auto* stage=static_cast<To*>(acquire(count*sizeof(To)));
        for(std::size_t i=0;i<count;++i)stage[i]=static_cast<To>(source[i]);
        check(cudaMemcpyAsync(destination,stage,count*sizeof(To),cudaMemcpyHostToDevice,stream_));
    }
    template<class To,class From> void downloadConverted(To* destination,const From* source,std::size_t count){
        if(!count)return;
        auto* stage=static_cast<From*>(acquire(count*sizeof(From)));
        reads_.push_back({nullptr,stage,0,[destination,stage,count]{
            for(std::size_t i=0;i<count;++i)destination[i]=static_cast<To>(stage[i]);
        }});
        check(cudaMemcpyAsync(stage,source,count*sizeof(From),cudaMemcpyDeviceToHost,stream_));
    }
    void finish(){
        check(cudaStreamSynchronize(stream_));
        for(const auto& read:reads_){if(read.convert)read.convert();else std::memcpy(read.destination,read.source,read.bytes);}
        reads_.clear();used_=0;
    }
};
}}
#endif
