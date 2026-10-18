#ifndef UGKP_GPU_DIAGNOSTICS_OWNER_CUH
#define UGKP_GPU_DIAGNOSTICS_OWNER_CUH
#include "DiagnosticsHost.H"
#include "DiagnosticsCatalogue.H"
#include <cuda_runtime.h>
#include <atomic>
#include <algorithm>
namespace ugkp { namespace diagnostics {
// Host-only owner, one independent resident lane. First fatal and first warning
// samples stay sticky across asynchronous operations until an existing host
// completion point. No per-operation reset, synchronization, or download.
class GpuOwner {
    DeviceContext* device_=nullptr;
    DeviceContext staging_{};
    std::uint64_t instance_;
    std::uint64_t sequence_=0;
    std::int64_t step_=-1;
    double time_=0;
    Snapshot snapshot_;
    HostSink sink_;
    std::uint32_t seen_[warningCategoryCount]{};
    bool overflowSeen_[warningCategoryCount]{};
    static std::uint64_t nextInstance() {
        static std::atomic<std::uint64_t> next{1};
        return next.fetch_add(1,std::memory_order_relaxed);
    }
public:
    GpuOwner():instance_(nextInstance()) {}
    ~GpuOwner(){if(device_)cudaFree(device_);}
    GpuOwner(const GpuOwner&)=delete;
    GpuOwner& operator=(const GpuOwner&)=delete;
    cudaError_t initialise() {
        auto rc=cudaMalloc(reinterpret_cast<void**>(&device_),sizeof(DeviceContext));
        if(rc!=cudaSuccess)return rc;
        reset(staging_,0,instance_);
        return cudaMemcpy(device_,&staging_,sizeof(staging_),cudaMemcpyHostToDevice);
    }
    DeviceContext* device() const{return device_;}
    std::uint64_t instanceId() const noexcept{return instance_;}
    bool terminal() const{return snapshot_.hasFatal;}
    void setSink(const HostSink& sink){sink_=sink;}
    void setOperationContext(std::uint64_t sequence,std::int64_t step,double time) {
        sequence_=sequence;step_=step;time_=time;
    }
    // Call ONLY after an already existing blocking readback or synchronization.
    // This is one bounded companion copy at the same completion point, not a
    // new synchronization cadence. Asynchronous advance never calls this.
    cudaError_t observeCompletedWork() {
        const auto rc=cudaMemcpy(&staging_,device_,sizeof(staging_),cudaMemcpyDeviceToHost);
        if(rc!=cudaSuccess) {
            recordFatal(&staging_,withOperation(makeRecord(Code::CudaRuntime,1,0,0,EntityKind::None,-1,-1,static_cast<double>(rc)),sequence_,step_,time_));
        }
        auto next=takeSnapshot(staging_,sourceCatalogue(),"ugkp-diagnostics-v1");
        for(auto& warning:next.warnings) {
            const auto index=static_cast<std::uint32_t>(warning.sample.code)-101u;
            const auto total=warning.count;warning.count=total-seen_[index];seen_[index]=total;
            const bool overflow=warning.overflow;
            warning.overflow=overflow&&!overflowSeen_[index];overflowSeen_[index]=overflow;
        }
        next.warnings.erase(std::remove_if(next.warnings.begin(),next.warnings.end(),
            [](const WarningSummary& warning){return warning.count==0&&!warning.overflow;}),next.warnings.end());
        next.warnings.insert(next.warnings.begin(),snapshot_.warnings.begin(),snapshot_.warnings.end());
        if(snapshot_.hasFatal) {
            next.hasFatal=true;next.fatal=snapshot_.fatal;
            next.operationSequence=snapshot_.operationSequence;
            next.instanceId=snapshot_.instanceId;
        }
        next.sources=sourceCatalogue();
        snapshot_=std::move(next);
        return rc;
    }
    void recordHostFatal(Record record) {
        if(snapshot_.hasFatal)return;
        record.severity=Severity::Fatal;
        if(!(record.metadata&explicitOperationMetadata))record=withOperation(record,sequence_,step_,time_);
        if(!record.instanceId)record.instanceId=instance_;
        snapshot_.hasFatal=true;snapshot_.fatal=record;
        snapshot_.operationSequence=record.operationSequence;
        snapshot_.instanceId=instance_;
        snapshot_.sources=sourceCatalogue();snapshot_.catalogueId="ugkp-diagnostics-v1";
    }
    cudaError_t finish(Snapshot& result) {
        result=snapshot_;
        result.sources.erase(std::remove_if(result.sources.begin(),result.sources.end(),
            [&](const SourceDescriptor& source) {
                if(result.hasFatal&&result.fatal.sourceSiteId==source.siteId)return false;
                for(const auto& warning:result.warnings)
                    if(warning.sample.sourceSiteId==source.siteId)return false;
                return true;
            }),result.sources.end());
        snapshot_.warnings.clear();return cudaSuccess;
    }
    int completeReadback(int status=0,bool completed=true) {
        if(!completed)return status;
        const auto rc=observeCompletedWork();
        return dispatchPending() || rc!=cudaSuccess || status!=0 ? 1 : 0;
    }
    bool dispatchPending() {
        const bool fatal=snapshot_.hasFatal;
        if(sink_.fatal||sink_.warning) {
            dispatch(snapshot_,sink_);
            snapshot_.warnings.clear();
        }
        return fatal;
    }
    void dispatchSnapshot(const Snapshot& snapshot) const {
        if(sink_.fatal||sink_.warning)dispatch(snapshot,sink_);
    }
};
}}
#endif
