#pragma once
#include "gpu/CoupledSurfaceHistory.H"
#include "gpu/ResidentMaterialDrive.H"
#include "../../../common/DiagnosticsHost.H"
#include <cuda_runtime.h>
#include <array>
#include <condition_variable>
#include <cerrno>
#include <unistd.h>
#include <cstdio>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace chmt {



class CoupledHistoryArchive {
public:
    static constexpr std::size_t slabFrames=256;
    struct Frame {
        CoupledSurfaceDownload* surface=nullptr;
        CoupledWallDownload* wall=nullptr;
        ResidentMaterialDriveInterval* interval=nullptr;
    };
    struct Chunk {
        std::vector<CoupledSurfaceDownload> surface;
        std::vector<CoupledWallDownload> wall;
        std::vector<ResidentMaterialDriveInterval> interval;
        std::size_t first=0,frames=0;
    };
private:
    struct Slab {
        CoupledSurfaceDownload *surface=nullptr,*hostSurface=nullptr;
        CoupledWallDownload *wall=nullptr,*hostWall=nullptr;
        ResidentMaterialDriveInterval *interval=nullptr,*hostInterval=nullptr;
        cudaEvent_t produced=nullptr,copied=nullptr;
        bool busy=false;
    };
public:
    class TransferStorage {
        friend class CoupledHistoryArchive;
        std::array<Slab,2> slabs{};
        cudaStream_t stream=nullptr;
        int device=0;
        std::size_t faces=0;
        bool matching=false,reusable=false;
    public:
        TransferStorage()=default;
        TransferStorage(const TransferStorage&)=delete;
        TransferStorage& operator=(const TransferStorage&)=delete;
        ~TransferStorage(){
            if(stream)cudaStreamSynchronize(stream);
            for(auto& slab:slabs){
                if(slab.surface)cudaFree(slab.surface);
                if(slab.wall)cudaFree(slab.wall);
                if(slab.interval)cudaFree(slab.interval);
                if(slab.hostSurface)cudaFreeHost(slab.hostSurface);
                if(slab.hostWall)cudaFreeHost(slab.hostWall);
                if(slab.hostInterval)cudaFreeHost(slab.hostInterval);
                if(slab.produced)cudaEventDestroy(slab.produced);
                if(slab.copied)cudaEventDestroy(slab.copied);
            }
            if(stream)cudaStreamDestroy(stream);
        }
    };
    using TransferPool=std::array<std::shared_ptr<TransferStorage>,2>;
private:
    struct Job {std::size_t slot=0,first=0,frames=0;};
    struct Offset {off_t offset=0;std::size_t first=0,frames=0;};
    std::shared_ptr<TransferStorage> transfers_;
    cudaStream_t lastProducer_=nullptr;
    std::size_t faces_=0,frames_=0,queuedFrames_=0;
    bool matching_=false,retain_=true,sealed_=false,stopping_=false,recycled_=false;
    std::FILE* file_=nullptr;
    std::thread writer_;std::mutex mutex_;std::condition_variable changed_;
    std::deque<Job> jobs_;std::exception_ptr failure_;
    std::vector<Offset> offsets_;
    mutable std::deque<std::shared_ptr<const Chunk>> cache_;
    mutable std::mutex readMutex_;
    [[noreturn]] static void ioFailure(const std::string& message,
        ugkp::diagnostics::Code code=ugkp::diagnostics::Code::InputData){
        using namespace ugkp::diagnostics;
        throw RecordedFailure(makeRecord(code,38001,26),{38001,__FILE__,"CoupledHistoryArchive",__LINE__,message});
    }
    static void checked(cudaError_t code,const char* operation){
        if(code==cudaSuccess)return;
        using namespace ugkp::diagnostics;
        auto record=makeRecord(Code::CudaRuntime,38002,26,0,EntityKind::None,-1,-1,int(code),0,0,Quantity::CudaStatus);
        throw RecordedFailure(record,{38002,__FILE__,"CoupledHistoryArchive",__LINE__,
            std::string(operation)+": "+cudaGetErrorString(code)});
    }
    template<class T> static void allocate(T*& pointer,std::size_t count,bool pinned){
        if(pointer||!count)return;
        if(count>std::numeric_limits<std::size_t>::max()/sizeof(T))ioFailure("history slab byte count overflow");
        auto address=reinterpret_cast<void**>(&pointer);
        checked(pinned?cudaMallocHost(address,count*sizeof(T)):cudaMalloc(address,count*sizeof(T)),
            pinned?"allocate pinned history slab":"allocate device history slab");
    }
    void prepareSlab(Slab& slab,bool pinned){
        allocate(pinned?slab.hostSurface:slab.surface,slabFrames*faces_,pinned);
        allocate(pinned?slab.hostWall:slab.wall,matching_?slabFrames*faces_:0,pinned);
        allocate(pinned?slab.hostInterval:slab.interval,slabFrames,pinned);
        if(!slab.produced)checked(cudaEventCreateWithFlags(&slab.produced,cudaEventDisableTiming),"create history producer event");
        if(pinned&&!slab.copied)checked(cudaEventCreateWithFlags(&slab.copied,cudaEventDisableTiming|cudaEventBlockingSync),"create history copy event");
    }
    void openSpool(){
        if(file_)return;
        file_=std::tmpfile();
        if(!file_)ioFailure("cannot create accepted history backing file",ugkp::diagnostics::Code::OutputData);
        writer_=std::thread([this]{worker();});
    }
    template<class T> void download(std::vector<T>& target,const T* source,std::size_t count)const{
        target.resize(count);
        if(count)checked(cudaMemcpyAsync(target.data(),source,count*sizeof(T),cudaMemcpyDeviceToHost,transfers_->stream),"read resident history");
    }
    template<class T> static void write(std::FILE* file,const T* data,std::size_t count){
        if(count&&std::fwrite(data,sizeof(T),count,file)!=count)ioFailure("cannot write accepted history payload",ugkp::diagnostics::Code::OutputData);
    }
    template<class T> static void read(std::FILE* file,std::vector<T>& output,std::size_t count){
        output.resize(count);
        if(count&&std::fread(output.data(),sizeof(T),count,file)!=count)ioFailure("accepted history payload is incomplete");
    }
    void worker()noexcept{
        try{
            checked(cudaSetDevice(transfers_->device),"select history copy device");
            for(;;){
                Job job;
                {std::unique_lock<std::mutex> lock(mutex_);
                    changed_.wait(lock,[&]{return stopping_||!jobs_.empty();});
                    if(jobs_.empty())return;job=jobs_.front();jobs_.pop_front();}
                auto& slab=transfers_->slabs[job.slot];
                checked(cudaEventSynchronize(slab.copied),"complete accepted history copy");
                const auto offset=::ftello(file_);
                if(offset<0)ioFailure("cannot locate accepted history payload",ugkp::diagnostics::Code::OutputData);
                write(file_,slab.hostSurface,job.frames*faces_);
                write(file_,slab.hostWall,matching_?job.frames*faces_:0);
                write(file_,slab.hostInterval,job.frames);
                offsets_.push_back({offset,job.first,job.frames});
                {std::lock_guard<std::mutex> lock(mutex_);slab.busy=false;}
                changed_.notify_all();
            }
        }catch(...){std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();changed_.notify_all();}
    }
    void checkFailure()const {if(failure_)std::rethrow_exception(failure_);}
    void queue(std::size_t first,std::size_t count,cudaStream_t producer){
        const auto slot=(first/slabFrames)%transfers_->slabs.size();auto& slab=transfers_->slabs[slot];
        openSpool();
        prepareSlab(slab,true);
        {std::lock_guard<std::mutex> lock(mutex_);checkFailure();slab.busy=true;}
        checked(cudaEventRecord(slab.produced,producer),"mark accepted history slab");
        checked(cudaStreamWaitEvent(transfers_->stream,slab.produced,0),"order accepted history copy");
        if(faces_)checked(cudaMemcpyAsync(slab.hostSurface,slab.surface,count*faces_*sizeof(CoupledSurfaceDownload),
            cudaMemcpyDeviceToHost,transfers_->stream),"copy accepted surface history");
        if(matching_&&faces_)checked(cudaMemcpyAsync(slab.hostWall,slab.wall,count*faces_*sizeof(CoupledWallDownload),
            cudaMemcpyDeviceToHost,transfers_->stream),"copy accepted wall matching history");
        checked(cudaMemcpyAsync(slab.hostInterval,slab.interval,count*sizeof(ResidentMaterialDriveInterval),
            cudaMemcpyDeviceToHost,transfers_->stream),"copy accepted history clock");
        checked(cudaEventRecord(slab.copied,transfers_->stream),"mark completed history copy");
        {std::lock_guard<std::mutex> lock(mutex_);jobs_.push_back({slot,first,count});}
        changed_.notify_all();queuedFrames_+=count;
    }
    void stopWriter()noexcept{
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
        changed_.notify_all();
        if(writer_.joinable())writer_.join();
    }
    void release()noexcept{
        if(writer_.joinable())stopWriter();
        if(file_){std::fclose(file_);file_=nullptr;}
    }
public:
    CoupledHistoryArchive(std::size_t faces,bool matching,TransferPool* pool=nullptr,bool retain=true)
        :faces_(faces),matching_(matching),retain_(retain){
        if(faces>std::numeric_limits<std::size_t>::max()/slabFrames)ioFailure("history face count overflow");
        int device;
        checked(cudaGetDevice(&device),"read history device");
        if(pool)for(auto& entry:*pool){
            if(entry.use_count()==1 && entry->reusable && entry->device==device && entry->faces==faces && entry->matching==matching){
                transfers_=entry;entry->reusable=false;recycled_=true;break;
            }
        }
        if(!transfers_){
            transfers_=std::make_shared<TransferStorage>();
            transfers_->device=device;transfers_->faces=faces;transfers_->matching=matching;
            checked(cudaStreamCreateWithFlags(&transfers_->stream,cudaStreamNonBlocking),"create history copy stream");
            prepareSlab(transfers_->slabs.front(),false);
            if(pool)for(auto& entry:*pool)if(entry.use_count()<=1){entry=transfers_;break;}
        }
    }
    CoupledHistoryArchive(const CoupledHistoryArchive&)=delete;
    CoupledHistoryArchive& operator=(const CoupledHistoryArchive&)=delete;
    ~CoupledHistoryArchive(){release();}
    bool retainsHistory()const{return retain_;}
    std::size_t writeSlot()const{return retain_?(frames_/slabFrames)%transfers_->slabs.size():0;}
    Frame next(cudaStream_t producer=nullptr){
        if(sealed_)ioFailure("cannot append to sealed accepted history");
        const auto slot=writeSlot();
        if(recycled_){checked(cudaStreamWaitEvent(producer,transfers_->slabs[0].produced,0),"order recycled history storage");recycled_=false;}
        if(retain_ && frames_==slabFrames && !file_){
            queue(0,slabFrames,lastProducer_);
            prepareSlab(transfers_->slabs[1],false);
        }
        if(file_){
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock,[&]{return !transfers_->slabs[slot].busy||failure_;});checkFailure();
        }
        const auto offset=(frames_%slabFrames)*faces_;auto& slab=transfers_->slabs[slot];
        return {faces_?slab.surface+offset:nullptr,matching_&&faces_?slab.wall+offset:nullptr,
            slab.interval+frames_%slabFrames};
    }

    struct Batch {Frame storage{};std::size_t first=0,capacity=0,slot=0;};
    Batch reserveBatch(cudaStream_t producer){return {next(producer),frames_,slabFrames-frames_%slabFrames,writeSlot()};}
    void acceptedBatch(std::size_t count,cudaStream_t producer){
        if(sealed_||count>slabFrames-frames_%slabFrames)
            ioFailure("accepted history batch exceeds its reserved storage");
        if(count)lastProducer_=producer;
        frames_+=count;
        if(file_&&count&&frames_%slabFrames==0)queue(frames_-slabFrames,slabFrames,producer);
    }
    void accepted(cudaStream_t producer){
        acceptedBatch(1,producer);
    }
    void seal(cudaStream_t producer){
        if(sealed_)return;
        if(!file_){
            checked(cudaEventRecord(transfers_->slabs[0].produced,producer),"mark resident history ready");
            sealed_=true;transfers_->reusable=true;
            return;
        }
        if(queuedFrames_!=frames_)queue(queuedFrames_,frames_-queuedFrames_,producer);
        {std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock,[&]{return failure_||(!transfers_->slabs[0].busy&&!transfers_->slabs[1].busy);});checkFailure();}
        if(std::fflush(file_)!=0)ioFailure("cannot flush accepted history payload",ugkp::diagnostics::Code::OutputData);
        sealed_=true;transfers_->reusable=true;
    }


    void releaseProducer(cudaStream_t producer){
        seal(producer);
        if(file_){stopWriter();transfers_.reset();}
    }
    bool spilled()const noexcept{return file_!=nullptr;}

    std::shared_ptr<const Chunk> chunk(std::size_t frame)const{
        std::lock_guard<std::mutex> lock(readMutex_);
        if(!retain_||!sealed_||frame>=frames_)ioFailure("requested accepted history frame is unavailable");
        const auto index=frame/slabFrames;
        for(auto it=cache_.begin();it!=cache_.end();++it)if(frame>=(*it)->first&&frame<(*it)->first+(*it)->frames){
            auto value=*it;cache_.erase(it);cache_.push_back(value);return value;
        }
        auto value=std::make_shared<Chunk>();
        if(!file_){
            const auto& slab=transfers_->slabs[0];
            checked(cudaSetDevice(transfers_->device),"select resident history device");
            checked(cudaStreamWaitEvent(transfers_->stream,slab.produced,0),"order resident history read");
            value->frames=frames_;
            try {
                download(value->surface,slab.surface,frames_*faces_);
                download(value->wall,slab.wall,matching_?frames_*faces_:0);
                download(value->interval,slab.interval,frames_);
                checked(cudaStreamSynchronize(transfers_->stream),"complete resident history read");
            } catch(...) {cudaStreamSynchronize(transfers_->stream);throw;}
            cache_.push_back(value);return value;
        }
        const auto& offset=offsets_.at(index);
        value->first=offset.first;value->frames=offset.frames;
        if(::fseeko(file_,offset.offset,SEEK_SET)!=0)ioFailure("cannot seek accepted history payload");
        read(file_,value->surface,offset.frames*faces_);read(file_,value->wall,matching_?offset.frames*faces_:0);
        read(file_,value->interval,offset.frames);
        if(cache_.size()==2)cache_.pop_front();cache_.push_back(value);return value;
    }
    ResidentMaterialDriveHistory residentView(void* producer)const{
        if(!transfers_||frames_>slabFrames)return {};
        return {transfers_->slabs[0].surface,transfers_->slabs[0].wall,transfers_->slabs[0].interval,frames_,faces_,producer};
    }
    std::size_t frames()const noexcept{return frames_;}
};
}
