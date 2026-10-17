#ifndef UGKP_DIAGNOSTICS_H
#define UGKP_DIAGNOSTICS_H

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#ifdef __CUDACC__
#include <cuda/atomic>
#define UGKP_DIAGNOSTIC_HD __host__ __device__
#else
#define UGKP_DIAGNOSTIC_HD
#endif

namespace ugkp { namespace diagnostics {

enum class Severity : std::uint32_t { Fatal = 1, Warning = 2 };
enum class Code : std::uint32_t {
    NegativeTemperature = 1, NonFinite = 2, InvalidIndex = 3,
    MissingOperator = 4, ZeroDenominator = 5, CudaRuntime = 6, InvalidPhysicalState = 7,
    ConservationResidual = 101, AccuracyResidual = 102, FiniteYClip = 103
};
namespace Quantity {
constexpr std::uint32_t Unspecified=0, Temperature=1, MassFraction=2,
    Density=3, Pressure=4, Velocity=5, InternalEnergy=6, Index=7,
    Denominator=8, CudaStatus=9, Conservation=10, Accuracy=11, LinearSystem=12;
}
enum class EntityKind : std::uint32_t { None = 0, Cell = 1, Face = 2, Particle = 3, Species = 4 };

// Fixed-width fields only. Wire encoding is explicit, never a memcpy of this POD.
struct Record {
    Severity severity;
    Code code;
    std::uint32_t sourceSiteId, operatorId, stageId;
    EntityKind entityKind;
    std::int64_t entityIndex;
    std::int32_t component;
    std::uint32_t metadata, quantityId;
    std::uint64_t operationSequence, instanceId;
    std::int64_t step;
    double simulationTime;
    double observed, reference, corrected;
};
struct WarningSlot {
    std::uint32_t sampleClaim, count, overflow;
    Record sample;
};
constexpr std::uint32_t warningCategoryCount = 3;
struct DeviceContext {
    std::uint32_t fatalClaim;
    Record firstFatal;
    WarningSlot warnings[warningCategoryCount];
    std::uint64_t operationSequence, instanceId;
    std::int64_t step;
    double simulationTime;
};
static_assert(std::is_trivial<DeviceContext>::value && std::is_standard_layout<DeviceContext>::value,
              "diagnostic storage must remain transferable POD");

UGKP_DIAGNOSTIC_HD inline bool isWarning(Code code) {
    return code == Code::ConservationResidual || code == Code::AccuracyResidual || code == Code::FiniteYClip;
}
UGKP_DIAGNOSTIC_HD inline Record makeRecord(
    Code code, std::uint32_t sourceSiteId = 0, std::uint32_t operatorId = 0,
    std::uint32_t stageId = 0, EntityKind entityKind = EntityKind::None,
    std::int64_t entityIndex = -1, std::int32_t component = -1,
    double observed = 0, double reference = 0, double corrected = 0, std::uint32_t quantityId = 0) {
    Record record{};
    record.severity = isWarning(code) ? Severity::Warning : Severity::Fatal;
    record.code = code; record.sourceSiteId = sourceSiteId; record.operatorId = operatorId;
    record.stageId = stageId; record.entityKind = entityKind; record.entityIndex = entityIndex;
    record.component = component; record.observed = observed;
    record.reference = reference; record.corrected = corrected; record.quantityId = quantityId;
    return record;
}
// Use per-launch state/arguments to stamp provenance without concurrent writes
// to shared context metadata. The first-fatal publisher preserves this stamp.
constexpr std::uint32_t explicitOperationMetadata = 0x80000000u;
constexpr std::uint32_t detailCodeMask = 0x7fffffffu;
UGKP_DIAGNOSTIC_HD inline Record withOperation(Record record, std::uint64_t sequence,
    std::int64_t step, double simulationTime) {
    record.operationSequence=sequence;record.step=step;record.simulationTime=simulationTime;
    record.metadata |= explicitOperationMetadata;return record;
}
namespace detail {
UGKP_DIAGNOSTIC_HD inline std::uint32_t read(const std::uint32_t* p) {
#ifdef __CUDA_ARCH__
    cuda::atomic_ref<std::uint32_t,cuda::thread_scope_device> flag(*const_cast<std::uint32_t*>(p));
    return flag.load(cuda::std::memory_order_relaxed);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}
UGKP_DIAGNOSTIC_HD inline std::uint32_t compareExchange(std::uint32_t* p, std::uint32_t old, std::uint32_t value) {
#ifdef __CUDA_ARCH__
    return atomicCAS(p, old, value);
#else
    __atomic_compare_exchange_n(p, &old, value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    return old;
#endif
}
UGKP_DIAGNOSTIC_HD inline void saturatingIncrement(WarningSlot& slot) {
    std::uint32_t old = read(&slot.count);
    for (;;) {
        if (old == 0xffffffffu) { compareExchange(&slot.overflow, 0u, 1u); return; }
        const std::uint32_t seen = compareExchange(&slot.count, old, old + 1u);
        if (seen == old) return;
        old = seen;
    }
}
}

// The owner calls reset only after all prior producers/consumers have completed.
// A fatal operation must not be reset to resume partially computed physics state.
inline void reset(DeviceContext& context, std::uint64_t operationSequence, std::uint64_t instanceId,
    std::int64_t step = -1, double simulationTime = 0) {
    context = DeviceContext{};
    context.operationSequence = operationSequence; context.instanceId = instanceId;
    context.step = step; context.simulationTime = simulationTime;
}
// Metadata advance is ordered with producers by the owner. It never clears
// unread warnings or the first fatal, including when reporting is deferred.
UGKP_DIAGNOSTIC_HD inline void beginOperation(DeviceContext& context,
    std::uint64_t operationSequence, std::int64_t step, double simulationTime) {
    context.operationSequence = operationSequence;
    context.step = step; context.simulationTime = simulationTime;
}
UGKP_DIAGNOSTIC_HD inline bool hasFatal(const DeviceContext* context) {
    return context && detail::read(&context->fatalClaim) != 0;
}
UGKP_DIAGNOSTIC_HD inline void recordFatal(DeviceContext* context, const Record& input);
UGKP_DIAGNOSTIC_HD inline void recordWarning(DeviceContext* context, const Record& input) {
    if (!isWarning(input.code)) { recordFatal(context,input); return; }
    if (!context) return;
    const std::uint32_t index = static_cast<std::uint32_t>(input.code) - 101u;
    WarningSlot& slot = context->warnings[index];
    detail::saturatingIncrement(slot);
    if (detail::compareExchange(&slot.sampleClaim, 0u, 1u) == 0u) {
        Record record = input; record.severity = Severity::Warning;
        record.instanceId = context->instanceId;
        if (!(record.metadata & explicitOperationMetadata)) {
            record.operationSequence = context->operationSequence;
            record.step = context->step; record.simulationTime = context->simulationTime;
        }
        slot.sample = record;
    }
}
UGKP_DIAGNOSTIC_HD inline void recordFatal(DeviceContext* context, const Record& input) {
    // A residual or finite-Y correction must never latch a fatal, even if a
    // caller accidentally submits it through this entry point.
    if (isWarning(input.code)) { recordWarning(context, input); return; }
    if (!context) {
#ifdef __CUDA_ARCH__
        asm volatile("trap;");
#else
        std::abort();
#endif
        return;
    }
    if (detail::compareExchange(&context->fatalClaim, 0u, 1u) == 0u) {
        Record record = input; record.severity = Severity::Fatal;
        record.instanceId = context->instanceId;
        if (!(record.metadata & explicitOperationMetadata)) {
            record.operationSequence = context->operationSequence;
            record.step = context->step; record.simulationTime = context->simulationTime;
        }
        context->firstFatal = record;
    }
}
#ifdef __CUDACC__
// One ordered setup launch per logical operation; no copy or synchronization.
// Launch outside a captured graph before replay, or update captured arguments
// explicitly, so step/time are never frozen to the graph's capture operation.
static __global__ void beginOperationKernel(DeviceContext* context, std::uint64_t sequence,
    std::int64_t step, double simulationTime) {
    if (blockIdx.x==0 && threadIdx.x==0 && context)
        beginOperation(*context,sequence,step,simulationTime);
}
// All threads in an ordinary block must call at the same program point. This
// entry guard is unsuitable for kernels with grid-wide/cooperative barriers.
__device__ inline bool blockMayProceed(const DeviceContext* context) {
    __shared__ unsigned int stopped;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) stopped = hasFatal(context) ? 1u : 0u;
    __syncthreads();
    return stopped == 0u;
}
#endif

}} // namespace ugkp::diagnostics
#undef UGKP_DIAGNOSTIC_HD
#endif
