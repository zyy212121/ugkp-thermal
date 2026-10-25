#ifndef CHMT_GPU_HOST_TRANSFER_QUEUE_H
#define CHMT_GPU_HOST_TRANSFER_QUEUE_H
#include "gpu/Buffer.H"
#include <cstring>

namespace chmt {
// One owner's stream and reusable pinned staging. Upload callers may release
// their source immediately; download destinations must live through finish().

class HostTransferQueue {
    struct Slot {void* data=nullptr;std::size_t capacity=0;};
    struct Readback {void* destination;const void* staging;std::size_t bytes;};
    CudaFault& fault_;
    cudaStream_t stream_=nullptr;
    std::vector<Slot> slots_;
    std::vector<Readback> readbacks_;
    std::size_t used_=0;
    bool failed_=false;
    void* acquire(std::size_t bytes){
        if(failed_&&!drain())return nullptr;
        if(used_==slots_.size())slots_.push_back({});
        auto& slot=slots_[used_];
        if(bytes>slot.capacity){
            void* replacement=nullptr;
            if(!fault_.check(cudaMallocHost(&replacement,bytes),"allocate pinned transfer staging"))return nullptr;
            if(slot.data&&!fault_.check(cudaFreeHost(slot.data),"release pinned transfer staging")){
                (void)cudaFreeHost(replacement);failed_=true;return nullptr;
            }
            slot.data=replacement;slot.capacity=bytes;
        }
        ++used_;return slot.data;
    }
public:
    explicit HostTransferQueue(CudaFault& fault):fault_(fault){}
    HostTransferQueue(const HostTransferQueue&)=delete;
    HostTransferQueue& operator=(const HostTransferQueue&)=delete;
    ~HostTransferQueue(){
        // Destinations may already have unwound after an error. Drain DMA only;
        // never publish readbacks from a destructor or a failed operation.
        if(used_)(void)cudaStreamSynchronize(stream_);
        for(const auto& slot:slots_)if(slot.data)fault_.check(cudaFreeHost(slot.data),"free pinned transfer staging");
    }
    // Bind only before enqueueing work; configuration candidates share their
    // resident owner's stream, never a new stream with implicit dependencies.
    void bindStream(cudaStream_t stream){stream_=stream;}
    cudaStream_t stream()const{return stream_;}
    bool uploadBytes(void* destination,const void* source,std::size_t bytes){
        try{
        if(!bytes)return !failed_;
        void* staged=acquire(bytes);if(!staged)return false;
        std::memcpy(staged,source,bytes);
        if(!fault_.check(cudaMemcpyAsync(destination,staged,bytes,cudaMemcpyHostToDevice,stream_),"queue CUDA upload")){failed_=true;return false;}
        return true;
        }catch(...){
            // Earlier readbacks may target another local in the unwinding
            // caller. Complete DMA but discard every pending publication.
            drain();throw;
        }
    }
    bool downloadBytes(void* destination,const void* source,std::size_t bytes){
        try{
        if(!bytes)return !failed_;
        void* staged=acquire(bytes);if(!staged)return false;
        readbacks_.push_back({destination,staged,bytes});
        if(!fault_.check(cudaMemcpyAsync(staged,source,bytes,cudaMemcpyDeviceToHost,stream_),"queue CUDA download")){failed_=true;return false;}
        return true;
        }catch(...){
            // Earlier readbacks may target another local in the unwinding
            // caller. Complete DMA but discard every pending publication.
            drain();throw;
        }
    }
    template<class T>bool upload(T* destination,const T* source,std::size_t count){
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T)){fault_.message="CUDA transfer size overflow";return false;}
        return uploadBytes(destination,source,count*sizeof(T));
    }
    template<class T>bool download(T* destination,const T* source,std::size_t count){
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T)){fault_.message="CUDA transfer size overflow";return false;}
        return downloadBytes(destination,source,count*sizeof(T));
    }
    bool finish(){
        const bool completed=fault_.check(cudaStreamSynchronize(stream_),"complete CUDA owner transfers");
        if(!completed||failed_){failed_=!completed;readbacks_.clear();if(completed)used_=0;return false;}
        for(const auto& readback:readbacks_)std::memcpy(readback.destination,readback.staging,readback.bytes);
        readbacks_.clear();used_=0;return true;
    }
    // Failure cleanup never writes into destinations whose scope is ending.
    bool drain(){
        readbacks_.clear();
        const bool completed=fault_.check(cudaStreamSynchronize(stream_),"drain CUDA owner transfers");
        if(completed){used_=0;failed_=false;}
        return completed;
    }
};
}
#endif
