#include "GraphContinuationTask.cuh"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
extern thread_local bool failNextAllocation;
extern std::atomic<bool> failNextThread;
using ugkp::gpu::GraphContinuationTask;

static GraphContinuationTask earlyStaticTask;
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
__device__ unsigned long long deviceTimeNanoseconds(){
    unsigned long long value;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));return value;
}
__global__ void addOne(int* value,unsigned long long durationNanoseconds){
    const auto begin=deviceTimeNanoseconds();while(deviceTimeNanoseconds()-begin<durationNanoseconds){}atomicAdd(value,1);
}
__global__ void invalidStore(int* address){
    const auto begin=deviceTimeNanoseconds();while(deviceTimeNanoseconds()-begin<20000000ULL){}*address=1;
}
__global__ void scaleByTen(int* address){*address*=10;}
struct Fixture {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;int* count=nullptr;
    Fixture(unsigned long long ticks=0,bool invalid=false){
        checked(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        checked(cudaMallocManaged(&count,sizeof(int)));*count=0;
        checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(invalid)invalidStore<<<1,1,0,stream>>>(nullptr);else addOne<<<1,1,0,stream>>>(count,ticks);
        checked(cudaStreamEndCapture(stream,&graph));checked(cudaGraphInstantiateWithFlags(&executable,graph,0));
    }
    ~Fixture(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(count)cudaFree(count);if(stream)cudaStreamDestroy(stream);}
};
static void normal(){
    Fixture longStream(250000000ULL),shortStream;
    std::atomic<int> finishLong{0},finishShort{0};GraphContinuationTask longTask,shortTask;
    longTask.start(longStream.stream,longStream.executable,[]{return false;},[&]{++finishLong;});
    shortTask.start(shortStream.stream,shortStream.executable,[]{return false;},[&]{++finishShort;});
    auto shortFuture=shortTask.result();
    require(shortFuture.wait_for(std::chrono::seconds(5))==std::future_status::ready,"short stream completion stalled");
    shortFuture.get();require(finishShort==1,"short completion duplicated or missing");


    require(longTask.result().wait_for(std::chrono::seconds(0))!=std::future_status::ready,"long/short overlap not exercised");
    longTask.result().get();require(finishLong==1,"long completion duplicated or missing");
    require(*longStream.count==1&&*shortStream.count==1,"initial launch count");
    for(int submission=0;submission<3;++submission){
        int batches=0,finishes=0;
        shortTask.start(shortStream.stream,shortStream.executable,[&]{return ++batches<3;},[&]{++finishes;});
        shortTask.result().get();
        require(batches==3&&finishes==1,"continuation batches/completion not exactly once");
    }
    require(*shortStream.count==10,"resubmission lost or duplicated work");
    shortTask.start(shortStream.stream,shortStream.executable,[]{return false;},[]{});
    scaleByTen<<<1,1,0,shortStream.stream>>>(shortStream.count);
    shortTask.result().get();checked(cudaStreamSynchronize(shortStream.stream));
    require(*shortStream.count==110,"start returned before graph submission, breaking stream order");
    std::shared_future<void> final;
    {
        GraphContinuationTask scoped;
        scoped.start(shortStream.stream,shortStream.executable,[]{return false;},[]{});
        final=scoped.result();
    }
    final.get();require(*shortStream.count==111,"destructor did not await callback completion");
    GraphContinuationTask parent,child;
    parent.start(shortStream.stream,shortStream.executable,[]{return false;},[&]{
        child.start(shortStream.stream,shortStream.executable,[]{return false;},[]{});
    });
    parent.result().get();child.result().get();
    require(*shortStream.count==113,"service-thread reentrant submission");
    std::puts("graph continuation callback multi-stream/relaunch/shared-future/destructor PASS");
}
static void awaitObservers(std::size_t expected){
    auto& executor=ugkp::gpu::graphContinuationDetail::serviceFor(0);
    const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(executor.observerCount()!=expected&&std::chrono::steady_clock::now()<limit)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(executor.observerCount()==expected,"observer registration was not reclaimed");
}
static void lifecycle(){
    auto& executor=ugkp::gpu::graphContinuationDetail::serviceFor(0);
    awaitObservers(0);


    auto first=executor.registerStream(nullptr);auto shared=executor.registerStream(nullptr);
    require(first==shared,"same-stream users did not share observer");
    executor.unregisterStream(first);require(shared->users==1,"first user retired shared observer");
    executor.unregisterStream(shared);
    auto next=executor.registerStream(nullptr);require(next!=first,"released handle generation reused");
    executor.unregisterStream(next);first.reset();shared.reset();next.reset();awaitObservers(0);
    {Fixture fixture;GraphContinuationTask task;
        task.start(nullptr,fixture.executable,[]{return false;},[]{});task.reset();awaitObservers(0);
        require(*fixture.count==1,"default-stream graph did not complete");}
    for(int i=0;i<32;++i){
        Fixture fixture;GraphContinuationTask task;
        task.start(fixture.stream,fixture.executable,[]{return false;},[]{});
        task.reset();
        awaitObservers(0);
    }
    {
        Fixture fixture;std::atomic<int> finished{0};std::exception_ptr errors[4];std::thread threads[4];
        for(int i=0;i<4;++i)threads[i]=std::thread([&,i]{try{
            checked(cudaSetDevice(0));GraphContinuationTask task;
            for(int j=0;j<8;++j){
                task.start(fixture.stream,fixture.executable,[]{return false;},[&]{++finished;});task.wait();
            }
        }catch(...){errors[i]=std::current_exception();}});
        for(auto& thread:threads)thread.join();
        for(auto& error:errors)if(error)std::rethrow_exception(error);
        require(finished==32,"concurrent same-stream work lost");awaitObservers(0);
        GraphContinuationTask complete,parent;
        complete.start(fixture.stream,fixture.executable,[]{return false;},[]{});complete.wait();
        parent.start(fixture.stream,fixture.executable,[]{return false;},[&]{complete.reset();});
        parent.wait();parent.reset();awaitObservers(0);
    }
    std::puts("graph continuation observer lifecycle/generation/shared-stream/concurrent/reentrant-reset PASS");
}
static void workerLastOwner(){
    using namespace ugkp::gpu::graphContinuationDetail;
    Fixture fixture(20000000ULL);const auto baseline=DeviceService::liveServiceCount();
    auto registry=std::make_unique<DeviceServiceRegistry>();
    auto executor=registry->get(0);auto observer=executor->registerStream(fixture.stream);
    auto work=std::make_shared<Work>();work->stream=fixture.stream;work->graph=fixture.executable;work->observer=observer;
    work->service=[]{return false;};


    work->finished=[executor,observer]{executor->unregisterStream(observer);};
    auto done=work->completion.get_future();auto submitted=work->submitted.get_future();
    executor->submit(work);submitted.get();executor.reset();observer.reset();work.reset();registry.reset();done.get();
    const auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(DeviceService::liveServiceCount()!=baseline&&std::chrono::steady_clock::now()<limit)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(DeviceService::liveServiceCount()==baseline,"worker-last-owner service was not joined");
    std::puts("graph continuation worker last-owner destruction PASS");
}
static void registryErrors(){
    using namespace ugkp::gpu::graphContinuationDetail;
    const auto baseline=DeviceService::liveServiceCount();
    for(int kind=0;kind<2;++kind){
        DeviceServiceRegistry registry;bool caught=false;
        if(kind==0)failNextAllocation=true;else failNextThread=true;
        try{registry.get(0);}catch(const std::exception&){caught=true;}
        failNextAllocation=false;failNextThread=false;
        require(caught,"registry construction failure was not injected");
        auto retry=registry.get(0);require(bool(retry),"failed creation left a null registry entry");
    }
    require(DeviceService::liveServiceCount()==baseline,"failed registry construction leaked service");
    std::puts("graph continuation allocation/thread creation exception safety PASS");
}
static void asynchronousError(){
    Fixture longFirst(400000000ULL),failure(0,true);
    GraphContinuationTask longTask,task;
    longTask.start(longFirst.stream,longFirst.executable,[]{return false;},[]{});
    task.start(failure.stream,failure.executable,[]{return false;},[]{});
    bool caught=false;
    try{task.result().get();}catch(const std::exception& e){caught=true;std::printf("expected async error: %s\n",e.what());}
    require(caught,"missing asynchronous CUDA exception");
    try{longTask.result().get();}catch(const std::exception&){}
    task.reset();longTask.reset();awaitObservers(0);
    std::puts("graph continuation callback omitted-on-error shared-future PASS");
}
static void submissionError(){
    Fixture fixture;GraphContinuationTask task;bool submitted=false,completed=false;
    try{task.start(fixture.stream,nullptr,[]{return false;},[]{});}catch(const std::exception&){submitted=true;}
    try{task.result().get();}catch(const std::exception&){completed=true;}
    require(submitted&&completed,"submission error did not resolve both acknowledgements");
    task.reset();awaitObservers(0);
    std::puts("graph continuation submission failure PASS");
}
int main(int argc,char** argv){
    try{
        const std::string mode=argc>1?argv[1]:"normal";
        if(mode=="registry-errors")registryErrors();
        else if(mode=="static-lifetime"){Fixture fixture;earlyStaticTask.start(fixture.stream,fixture.executable,[]{return false;},[]{});earlyStaticTask.wait();std::puts("static task service lifetime body PASS; destructor checked at process exit");}
        else if(mode=="async-error")asynchronousError();else if(mode=="submission-error")submissionError();else {normal();lifecycle();workerLastOwner();}
        return 0;
    }
    catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
