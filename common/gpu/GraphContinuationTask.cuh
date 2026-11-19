#ifndef UGKP_GPU_GRAPH_CONTINUATION_TASK_CUH
#define UGKP_GPU_GRAPH_CONTINUATION_TASK_CUH
#include <cuda_runtime.h>
#include <cstddef>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <functional>
#include <atomic>
#include <vector>
namespace ugkp { namespace gpu {
namespace graphContinuationDetail {
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(std::string("CUDA graph continuation: ")+cudaGetErrorString(e));}
struct Observer;
struct Work {
    std::shared_ptr<Observer> observer;
    cudaStream_t stream=nullptr;cudaGraphExec_t graph=nullptr;
    std::function<bool()> service;std::function<void()> finished;std::promise<void> completion,submitted;
    bool completed=false,submissionPublished=false;
};
class DeviceService;
struct Batch {
        DeviceService* owner=nullptr;std::shared_ptr<Work> work;cudaEvent_t event=nullptr;
        Batch* next=nullptr;bool queued=false,callbackReady=false,observed=false,processed=false;
        cudaError_t status=cudaSuccess;std::exception_ptr failure;
};
struct Observer {
    cudaStream_t stream=nullptr;std::size_t users=0,active=0;bool stopping=false;
    std::deque<Batch*> batches;std::condition_variable changed;std::thread thread;
};
class DeviceService {
    friend class DeviceServiceRegistry;
    inline static std::atomic<std::size_t> liveServices_{0};
    int device_;std::mutex mutex_;std::condition_variable changed_,idle_;bool workerBusy_=false;
    std::deque<std::shared_ptr<Work>> pending_;
    std::map<Observer*,std::shared_ptr<Observer>> observers_;
    std::map<Batch*,std::shared_ptr<Batch>> active_;
    Batch* readyHead_=nullptr;Batch* readyTail_=nullptr;
    bool stopping_=false;std::size_t launching_=0,retiringObservers_=0;std::thread worker_;

    void queue(Batch* batch) noexcept {
        if(batch->queued)return;
        batch->queued=true;batch->next=nullptr;
        if(readyTail_)readyTail_->next=batch;else readyHead_=batch;
        readyTail_=batch;
    }
    static void CUDART_CB ready(void* opaque) noexcept {
        auto* batch=static_cast<Batch*>(opaque);auto* owner=batch->owner;
        {std::lock_guard<std::mutex> lock(owner->mutex_);batch->callbackReady=true;owner->queue(batch);}
        owner->changed_.notify_one();
    }
    void submission(const std::shared_ptr<Work>& work,std::exception_ptr failure={}){
        if(work->submissionPublished)return;work->submissionPublished=true;
        if(failure)work->submitted.set_exception(failure);else work->submitted.set_value();
    }
    void fail(const std::shared_ptr<Work>& work,std::exception_ptr failure){
        submission(work,failure);
        if(work->completed)return;cudaStreamSynchronize(work->stream);work->completed=true;work->completion.set_exception(failure);
    }
    void launch(const std::shared_ptr<Work>& work){
        auto batch=std::make_shared<Batch>();batch->owner=this;batch->work=work;
        {std::lock_guard<std::mutex> lock(mutex_);active_.emplace(batch.get(),batch);++work->observer->active;}
        bool launched=false;
        try{
            check(cudaSetDevice(device_));
            check(cudaEventCreateWithFlags(&batch->event,cudaEventDisableTiming|cudaEventBlockingSync));
            check(cudaGraphLaunch(work->graph,work->stream));launched=true;
            check(cudaLaunchHostFunc(work->stream,&DeviceService::ready,batch.get()));

            check(cudaEventRecord(batch->event,work->stream));
            auto observer=work->observer;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                observer->batches.push_back(batch.get());
            }
            observer->changed.notify_one();
            submission(work);
        }catch(...){
            const auto failure=std::current_exception();


            if(launched)cudaStreamSynchronize(work->stream);
            submission(work,failure);
            {std::lock_guard<std::mutex> lock(mutex_);batch->failure=failure;batch->observed=true;queue(batch.get());}
            changed_.notify_one();
        }
    }
    void observe(Observer* observer){
        const auto deviceStatus=cudaSetDevice(device_);
        for(;;){
            Batch* batch=nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                observer->changed.wait(lock,[&]{return !observer->batches.empty()||observer->stopping;});
                if(observer->batches.empty())return;
                batch=observer->batches.front();observer->batches.pop_front();
            }




            const auto status=deviceStatus==cudaSuccess?cudaEventSynchronize(batch->event):deviceStatus;
            {std::lock_guard<std::mutex> lock(mutex_);batch->status=status;batch->observed=true;queue(batch);}
            changed_.notify_one();
        }
    }
    bool reclaimable(const std::shared_ptr<Observer>& observer)const{
        return observer->active==0&&(observer->users==0||(stopping_&&pending_.empty()&&!launching_));
    }
    bool hasRetiredObserver()const{
        for(const auto& item:observers_)if(reclaimable(item.second))return true;
        return false;
    }
    bool idleLocked()const{
        return !workerBusy_&&pending_.empty()&&active_.empty()&&!launching_&&!readyHead_&&!hasRetiredObserver();
    }
    void awaitIdle(){std::unique_lock<std::mutex> lock(mutex_);idle_.wait(lock,[&]{return idleLocked();});}
    void run(){
        for(;;){
            std::shared_ptr<Work> pending;std::shared_ptr<Batch> batch;std::shared_ptr<Observer> retiredObserver;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workerBusy_=false;idle_.notify_all();
                changed_.wait(lock,[&]{return readyHead_||!pending_.empty()||hasRetiredObserver()||(stopping_&&active_.empty());});
                for(auto it=observers_.begin();it!=observers_.end();++it)if(reclaimable(it->second)){
                    retiredObserver=it->second;retiredObserver->stopping=true;observers_.erase(it);++retiringObservers_;break;
                }
                if(retiredObserver){}
                else if(readyHead_){
                    auto* next=readyHead_;readyHead_=next->next;if(!readyHead_)readyTail_=nullptr;
                    next->queued=false;next->next=nullptr;batch=active_.at(next);
                }else if(!pending_.empty()){pending=std::move(pending_.front());pending_.pop_front();++launching_;}
                else return;
                workerBusy_=true;
            }
            if(retiredObserver){


                retiredObserver->changed.notify_one();retiredObserver->thread.join();
                {std::lock_guard<std::mutex> lock(mutex_);--retiringObservers_;}continue;
            }
            if(pending){
                try{launch(pending);}catch(...){fail(pending,std::current_exception());}
                {
                    std::lock_guard<std::mutex> lock(mutex_);--launching_;
                    if(stopping_)for(auto& item:observers_)item.second->changed.notify_one();
                }
                continue;
            }
            bool process=false;cudaError_t status=cudaSuccess;std::exception_ptr failure;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                status=batch->status;failure=batch->failure;
                process=!batch->processed&&(batch->callbackReady||batch->observed);
                if(process)batch->processed=true;
            }
            try{
                check(cudaSetDevice(device_));check(status);if(failure)std::rethrow_exception(failure);
                if(process&&!batch->work->completed){
                    if(batch->work->service())launch(batch->work);
                    else{
                        batch->work->finished();batch->work->completed=true;
                        batch->work->completion.set_value();
                    }
                }
            }catch(...){fail(batch->work,std::current_exception());}
            bool retire=false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                retire=batch->observed&&batch->processed&&!batch->queued;
                if(retire){active_.erase(batch.get());--batch->work->observer->active;}
            }
            if(retire&&batch->event)cudaEventDestroy(batch->event);
            if(retire){
                std::lock_guard<std::mutex> lock(mutex_);
                for(auto& item:observers_)item.second->changed.notify_one();
                changed_.notify_one();
            }
        }
    }
public:
    explicit DeviceService(int device):device_(device),worker_([this]{run();}){++liveServices_;}
    ~DeviceService(){
        {
            std::lock_guard<std::mutex> lock(mutex_);stopping_=true;
            for(auto& item:observers_)item.second->changed.notify_one();
        }
        changed_.notify_one();worker_.join();--liveServices_;
    }
    bool onWorkerThread()const noexcept{return std::this_thread::get_id()==worker_.get_id();}
    static std::size_t liveServiceCount()noexcept{return liveServices_.load();}
    std::shared_ptr<Observer> registerStream(cudaStream_t stream){
        std::lock_guard<std::mutex> lock(mutex_);
        for(auto& entry:observers_)if(entry.second->stream==stream&&entry.second->users){
            ++entry.second->users;return entry.second;
        }
        auto observer=std::make_shared<Observer>();observer->stream=stream;observer->users=1;
        observers_.emplace(observer.get(),observer);
        try{observer->thread=std::thread([this,observer]{observe(observer.get());});}
        catch(...){observers_.erase(observer.get());throw;}
        return observer;
    }
    void unregisterStream(const std::shared_ptr<Observer>& observer)noexcept{
        {std::lock_guard<std::mutex> lock(mutex_);--observer->users;}
        changed_.notify_one();
    }

    std::size_t observerCount(){std::lock_guard<std::mutex> lock(mutex_);return observers_.size()+retiringObservers_;}
    void submit(std::shared_ptr<Work> work){


        if(std::this_thread::get_id()==worker_.get_id()){
            try{launch(work);}catch(...){fail(work,std::current_exception());}
            return;
        }
        {std::lock_guard<std::mutex> lock(mutex_);pending_.push_back(std::move(work));}
        changed_.notify_one();
    }
};




class DeviceServiceRegistry {
    std::mutex mutex_;std::map<int,std::shared_ptr<DeviceService>> services_;
public:
    std::shared_ptr<DeviceService> get(int device){
        std::lock_guard<std::mutex> lock(mutex_);
        auto found=services_.find(device);if(found!=services_.end())return found->second;
        auto service=std::make_shared<DeviceService>(device);
        services_.emplace(device,service);return service;
    }


    ~DeviceServiceRegistry(){
        for(;;){
            std::vector<std::shared_ptr<DeviceService>> snapshot;
            {std::lock_guard<std::mutex> lock(mutex_);for(auto& item:services_)snapshot.push_back(item.second);}
            for(auto& service:snapshot)service->awaitIdle();


            std::lock_guard<std::mutex> registryLock(mutex_);
            std::vector<std::unique_lock<std::mutex>> locks;
            for(auto& item:services_)locks.emplace_back(item.second->mutex_);
            bool idle=true;for(auto& item:services_)idle=idle&&item.second->idleLocked();
            if(idle)break;
        }
    }
};
inline std::shared_ptr<DeviceService> sharedServiceFor(int device){
    static DeviceServiceRegistry registry;return registry.get(device);
}
inline DeviceService& serviceFor(int device){return *sharedServiceFor(device);}
}
class GraphContinuationTask {
    std::shared_future<void> result_;
    std::shared_ptr<graphContinuationDetail::DeviceService> executor_;
    std::shared_ptr<graphContinuationDetail::Observer> observer_;
public:
    GraphContinuationTask()=default;
    GraphContinuationTask(const GraphContinuationTask&)=delete;
    GraphContinuationTask& operator=(const GraphContinuationTask&)=delete;
    ~GraphContinuationTask(){reset();}


    void reset()noexcept{
        if(result_.valid())result_.wait();
        if(observer_){executor_->unregisterStream(observer_);observer_.reset();executor_=nullptr;}
        result_={};
    }
    void wait()const{if(result_.valid())result_.get();}
    std::shared_future<void> result()const{return result_;}
    template<class Service,class Finished> void start(cudaStream_t stream,cudaGraphExec_t graph,Service service,Finished finished){
        using namespace graphContinuationDetail;
        wait();int device=0;check(cudaGetDevice(&device));auto executor=sharedServiceFor(device);
        if(!observer_||executor_!=executor||observer_->stream!=stream){
            reset();observer_=executor->registerStream(stream);executor_=executor;
        }
        auto work=std::make_shared<Work>();work->observer=observer_;work->stream=stream;work->graph=graph;
        work->service=std::move(service);work->finished=std::move(finished);
        result_=work->completion.get_future().share();auto submitted=work->submitted.get_future();
        try{executor->submit(work);}
        catch(...){work->completion.set_exception(std::current_exception());throw;}


        submitted.get();
    }
};
}}
#endif
