#pragma once
#include <cuda_runtime.h>
#include "../DiagnosticsHost.H"
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ugkp { namespace gpu {
inline void checkSnapshotCopy(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return;
    using namespace ugkp::diagnostics;
    auto record=makeRecord(Code::CudaRuntime,42002,42,0,EntityKind::None,-1,-1,
        static_cast<double>(result),0,0,Quantity::CudaStatus);
    throw RecordedFailure(record,{42002,__FILE__,__func__,__LINE__,
        std::string(operation)+": "+cudaGetErrorString(result)});
}
[[noreturn]] inline void snapshotDataFailure(const char* reason) {
    using namespace ugkp::diagnostics;
    throw RecordedFailure(makeRecord(Code::InputData,42003,42),
        {42003,__FILE__,__func__,__LINE__,reason});
}

template<class T> struct SnapshotRange {
    std::size_t offset = 0, count = 0;
    T* at(unsigned char* bytes) const {
        return count ? reinterpret_cast<T*>(bytes + offset) : nullptr;
    }
    const T* at(const unsigned char* bytes) const {
        return count ? reinterpret_cast<const T*>(bytes + offset) : nullptr;
    }
    void copy(unsigned char* bytes, const T* source, cudaStream_t producer) const {
        if (!count) return;
        if (!source) snapshotDataFailure("missing device field in accepted snapshot");
        checkSnapshotCopy(cudaMemcpyAsync(at(bytes), source, count*sizeof(T),
            cudaMemcpyDeviceToDevice, producer), "freeze accepted field");
    }
};

struct SnapshotLayout {
    std::size_t bytes = 0;
    template<class T> SnapshotRange<T> allocate(std::size_t count) {
        if (!count) return {};
        const auto limit = std::numeric_limits<std::size_t>::max();
        const auto padding = (alignof(T) - (bytes % alignof(T))) % alignof(T);
        if (padding > limit - bytes || count > (limit - bytes - padding)/sizeof(T))
            throw std::length_error("accepted snapshot byte count overflow");
        SnapshotRange<T> range{bytes + padding, count};
        bytes = range.offset + count*sizeof(T);
        return range;
    }
};




class FrozenReadback {
    unsigned char *device_ = nullptr, *host_ = nullptr;
    cudaStream_t copy_ = nullptr;
    cudaEvent_t frozen_ = nullptr, ready_ = nullptr;
    std::size_t capacity_ = 0;
    bool pending_ = false;
    int deviceOrdinal_ = 0;

    void reserve(std::size_t bytes) {
        if (!copy_) {
            checkSnapshotCopy(cudaGetDevice(&deviceOrdinal_), "get snapshot device");
            checkSnapshotCopy(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "create snapshot copy stream");
        }
        if (!frozen_) checkSnapshotCopy(cudaEventCreateWithFlags(&frozen_, cudaEventDisableTiming), "create freeze event");
        if (!ready_) checkSnapshotCopy(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming | cudaEventBlockingSync), "create snapshot ready event");
        if (bytes <= capacity_) return;
        capacity_ = 0;
        if (device_) {
            checkSnapshotCopy(cudaFree(device_), "release previous device snapshot");
            device_ = nullptr;
        }
        if (host_) {
            checkSnapshotCopy(cudaFreeHost(host_), "release previous pinned snapshot");
            host_ = nullptr;
        }
        checkSnapshotCopy(cudaMalloc(reinterpret_cast<void**>(&device_), bytes), "allocate device snapshot");
        checkSnapshotCopy(cudaMallocHost(reinterpret_cast<void**>(&host_), bytes), "allocate pinned snapshot");
        capacity_ = bytes;
    }
public:
    FrozenReadback() = default;
    FrozenReadback(const FrozenReadback&) = delete;
    FrozenReadback& operator=(const FrozenReadback&) = delete;
    ~FrozenReadback() {
        if (pending_) cudaEventSynchronize(ready_);
        if (device_) cudaFree(device_);
        if (host_) cudaFreeHost(host_);
        if (frozen_) cudaEventDestroy(frozen_);
        if (ready_) cudaEventDestroy(ready_);
        if (copy_) cudaStreamDestroy(copy_);
    }

    template<class Pack>
    void freeze(std::size_t bytes, cudaStream_t producer, Pack pack) {
        if (pending_) throw std::logic_error("accepted snapshot still leased");
        if (!bytes) return;
        reserve(bytes);
        try {
            pack(device_);
            checkSnapshotCopy(cudaEventRecord(frozen_, producer), "record accepted snapshot");
            checkSnapshotCopy(cudaStreamWaitEvent(copy_, frozen_, 0), "order snapshot download");
            checkSnapshotCopy(cudaMemcpyAsync(host_, device_, bytes, cudaMemcpyDeviceToHost, copy_), "download frozen snapshot");
            checkSnapshotCopy(cudaEventRecord(ready_, copy_), "record completed snapshot download");
            pending_ = true;
        } catch (...) {

            cudaStreamSynchronize(producer);
            cudaStreamSynchronize(copy_);
            throw;
        }
    }

    const unsigned char* collect() {
        if (pending_) {
            checkSnapshotCopy(cudaSetDevice(deviceOrdinal_), "select output snapshot device");
            checkSnapshotCopy(cudaEventSynchronize(ready_), "collect frozen snapshot");
            pending_ = false;
        }
        return host_;
    }
    void orderNextWriter(cudaStream_t writer) const {
        if (pending_)
            checkSnapshotCopy(cudaStreamWaitEvent(writer, frozen_, 0), "order next update after snapshot freeze");
    }
};
}}
