#ifndef UGKP_GPU_PINNED_TRANSFER_QUEUE_CUH
#define UGKP_GPU_PINNED_TRANSFER_QUEUE_CUH
#include <cuda_runtime.h>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace ugkp { namespace gpu {




class PinnedTransferQueue {
    struct Slot {void* data=nullptr;std::size_t capacity=0;};
    struct Read {void* destination;const void* source;std::size_t bytes;std::function<void()> convert;};
    struct Batch {
        std::vector<Slot> slots;cudaEvent_t retired=nullptr;bool pending=false;
        ~Batch(){for(auto slot:slots)if(slot.data)cudaFreeHost(slot.data);if(retired)cudaEventDestroy(retired);}
    };
    cudaStream_t stream_=nullptr;std::vector<std::unique_ptr<Batch>> batches_;
    std::vector<Read> reads_;Batch* active_=nullptr;std::size_t used_=0;
    std::size_t maximumBatches_=8;cudaError_t deferredError_=cudaSuccess;
    static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(std::string("CUDA pinned transfer: ")+cudaGetErrorString(code));}
    void acquireBatch(){
        check(deferredError_);
        if(active_)return;
        for(auto& batch:batches_){
            if(batch->pending){
                const auto status=cudaEventQuery(batch->retired);
                if(status==cudaErrorNotReady)continue;
                check(status);batch->pending=false;
            }
            active_=batch.get();return;
        }
        if(batches_.size()<maximumBatches_){
            auto batch=std::make_unique<Batch>();
            check(cudaEventCreateWithFlags(&batch->retired,cudaEventDisableTiming|cudaEventBlockingSync));
            batches_.push_back(std::move(batch));active_=batches_.back().get();return;
        }


        auto& batch=*batches_.front();check(cudaEventSynchronize(batch.retired));
        batch.pending=false;active_=&batch;
    }
    void* acquire(std::size_t bytes){
        acquireBatch();
        if(used_==active_->slots.size())active_->slots.push_back({});
        auto& slot=active_->slots[used_];
        if(slot.capacity<bytes){void* next=nullptr;check(cudaMallocHost(&next,bytes));
            if(slot.data)cudaFreeHost(slot.data);slot.data=next;slot.capacity=bytes;}
        ++used_;return slot.data;
    }
    void transferred(cudaError_t status){
        if(status==cudaSuccess)status=cudaEventRecord(active_->retired,stream_);
        if(status!=cudaSuccess){


            deferredError_=status;cudaStreamSynchronize(stream_);active_->pending=false;check(status);
        }
        active_->pending=true;
    }
public:
    explicit PinnedTransferQueue(std::size_t maximumBatches=8):maximumBatches_(maximumBatches){
        if(!maximumBatches)throw std::invalid_argument("pinned transfer batch limit must be positive");
    }
    PinnedTransferQueue(const PinnedTransferQueue&)=delete;
    PinnedTransferQueue& operator=(const PinnedTransferQueue&)=delete;
    ~PinnedTransferQueue(){drain();}
    void bind(cudaStream_t stream){
        if(stream!=stream_){drain();check(deferredError_);stream_=stream;}
    }
    void discard()noexcept{
        reads_.clear();


        active_=nullptr;used_=0;
    }


    void drain()noexcept{
        discard();for(auto& batch:batches_)if(batch->pending){
            const auto status=cudaEventSynchronize(batch->retired);
            if(status!=cudaSuccess)deferredError_=status;
            batch->pending=false;
        }
    }
    class Operation {
        PinnedTransferQueue& queue_;
    public:
        explicit Operation(PinnedTransferQueue& queue):queue_(queue){}
        Operation(const Operation&)=delete;
        Operation& operator=(const Operation&)=delete;
        ~Operation(){queue_.discard();}
    };
    void upload(void* destination,const void* source,std::size_t bytes){
        if(!bytes)return;void* stage=acquire(bytes);std::memcpy(stage,source,bytes);
        transferred(cudaMemcpyAsync(destination,stage,bytes,cudaMemcpyHostToDevice,stream_));
    }
    void download(void* destination,const void* source,std::size_t bytes){
        if(!bytes)return;void* stage=acquire(bytes);reads_.push_back({destination,stage,bytes,{}});
        transferred(cudaMemcpyAsync(stage,source,bytes,cudaMemcpyDeviceToHost,stream_));
    }
    template<class To,class From> void uploadConverted(To* destination,const From* source,std::size_t count){
        if(!count)return;auto* stage=static_cast<To*>(acquire(count*sizeof(To)));
        for(std::size_t i=0;i<count;++i)stage[i]=static_cast<To>(source[i]);
        transferred(cudaMemcpyAsync(destination,stage,count*sizeof(To),cudaMemcpyHostToDevice,stream_));
    }
    template<class To,class From> void downloadConverted(To* destination,const From* source,std::size_t count){
        if(!count)return;auto* stage=static_cast<From*>(acquire(count*sizeof(From)));
        reads_.push_back({nullptr,stage,0,[destination,stage,count]{
            for(std::size_t i=0;i<count;++i)destination[i]=static_cast<To>(stage[i]);
        }});
        transferred(cudaMemcpyAsync(stage,source,count*sizeof(From),cudaMemcpyDeviceToHost,stream_));
    }
    void finish(){
        check(deferredError_);
        check(cudaStreamSynchronize(stream_));
        for(const auto& read:reads_){if(read.convert)read.convert();else std::memcpy(read.destination,read.source,read.bytes);}
        reads_.clear();used_=0;active_=nullptr;
        for(auto& batch:batches_)batch->pending=false;
    }
};
}}
#endif
