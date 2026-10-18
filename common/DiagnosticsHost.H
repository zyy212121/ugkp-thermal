#ifndef UGKP_DIAGNOSTICS_HOST_H
#define UGKP_DIAGNOSTICS_HOST_H
#include "Diagnostics.H"
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ugkp { namespace diagnostics {
struct SourceDescriptor {
    std::uint32_t siteId;
    std::string file, function;
    std::uint32_t line;
    std::string description;
};
struct WarningSummary { Record sample; std::uint32_t count; bool overflow; };
struct Snapshot {
    std::uint64_t operationSequence = 0, instanceId = 0;
    bool hasFatal = false;
    Record fatal{};
    std::vector<WarningSummary> warnings;
    std::vector<SourceDescriptor> sources;
    std::string catalogueId;
};
// Attach only descriptors referenced by the complete pending snapshot.
inline void selectSnapshotSources(Snapshot& result,const std::vector<SourceDescriptor>& sources) {
    result.sources.clear();
    if(!result.hasFatal&&result.warnings.empty())return;
    // Send descriptors only for sampled sites in this bounded snapshot, not
    // the full application catalogue on every successful IPC reply.
    for(const auto& source:sources) {
        bool used=result.hasFatal && result.fatal.sourceSiteId==source.siteId;
        for(const auto& warning:result.warnings)used=used||warning.sample.sourceSiteId==source.siteId;
        if(used)result.sources.push_back(source);
    }
}
// Read only after all producers complete (join/event + ordered device copy).
inline Snapshot takeSnapshot(const DeviceContext& context,
    const std::vector<SourceDescriptor>& sources = {}, const std::string& catalogueId = {}) {
    Snapshot result;
    result.operationSequence = context.operationSequence; result.instanceId = context.instanceId;
    result.hasFatal = context.fatalClaim != 0;
    if (result.hasFatal) result.fatal = context.firstFatal;
    for (std::uint32_t i=0; i<warningCategoryCount; ++i) {
        const WarningSlot& slot = context.warnings[i];
        if (slot.count) result.warnings.push_back({slot.sample, slot.count, slot.overflow != 0});
    }
    selectSnapshotSources(result,sources);
    result.catalogueId = catalogueId;
    return result;
}
inline const SourceDescriptor* findSource(const Snapshot& snapshot, std::uint32_t siteId) {
    for (const auto& source : snapshot.sources) if (source.siteId == siteId) return &source;
    return nullptr;
}
inline const char* codeName(Code code) {
    switch (code) {
        case Code::NegativeTemperature: return "negative/nonphysical temperature";
        case Code::NonFinite: return "nonfinite value";
        case Code::InvalidIndex: return "invalid index";
        case Code::MissingOperator: return "missing operator/model";
        case Code::ZeroDenominator: return "zero denominator";
        case Code::CudaRuntime: return "CUDA runtime failure";
        case Code::InvalidPhysicalState: return "invalid physical state";
        case Code::ConservationResidual: return "conservation residual";
        case Code::AccuracyResidual: return "accuracy residual";
        case Code::FiniteYClip: return "finite mass fraction clipped to [0,1]";
    }
    return "unknown diagnostic";
}
inline const char* quantityName(std::uint32_t quantity) {
    switch(quantity) {
        case Quantity::Temperature:return "temperature";
        case Quantity::MassFraction:return "mass fraction";
        case Quantity::Density:return "density";
        case Quantity::Pressure:return "pressure";
        case Quantity::Velocity:return "velocity";
        case Quantity::InternalEnergy:return "internal energy";
        case Quantity::Index:return "index";
        case Quantity::Denominator:return "denominator";
        case Quantity::CudaStatus:return "CUDA status";
        case Quantity::Conservation:return "conservation residual";
        case Quantity::Accuracy:return "accuracy residual";
        case Quantity::LinearSystem:return "linear system";
        case Quantity::HeatCapacity:return "heat capacity";
        default:return "unspecified";
    }
}
inline std::string formatRecord(const Record& record, const SourceDescriptor* source) {
    std::ostringstream out;
    out << codeName(record.code) << " [code=" << static_cast<std::uint32_t>(record.code)
        << ", detailCode=" << (record.metadata & detailCodeMask) << ", site=" << record.sourceSiteId << ", operator=" << record.operatorId
        << ", stage=" << record.stageId << ", instance=" << record.instanceId
        << ", operation=" << record.operationSequence << ", step=" << record.step
        << ", time=" << std::setprecision(17) << record.simulationTime << ", quantity=" << quantityName(record.quantityId) << "(" << record.quantityId << ")" << ", entityKind=" << static_cast<std::uint32_t>(record.entityKind)
        << ", entity=" << record.entityIndex << ", component=" << record.component << "]"
        << std::setprecision(17) << " observed=" << record.observed << " reference=" << record.reference
        << " corrected=" << record.corrected;
    if (source) out << " at " << source->file << ':' << source->line << " in " << source->function << ": " << source->description;
    return out.str();
}
struct HostSink {
    void* user = nullptr;
    void (*fatal)(void*, const Record&, const SourceDescriptor*) = nullptr;
    void (*warning)(void*, const WarningSummary&, const SourceDescriptor*) = nullptr;
    bool warningsEnabled = true;
};
// Dispatch from the owning host thread, never a CUDA callback. Tests may throw.
// Warning suppression affects output only; the immutable snapshot keeps counts.
inline void dispatch(const Snapshot& snapshot, const HostSink& sink) {
    if (sink.warningsEnabled && sink.warning)
        for (const auto& warning : snapshot.warnings)
            sink.warning(sink.user, warning, findSource(snapshot, warning.sample.sourceSiteId));
    if (snapshot.hasFatal) {
        if (sink.fatal) sink.fatal(sink.user, snapshot.fatal, findSource(snapshot, snapshot.fatal.sourceSiteId));
        // A missing or erroneously returning fatal callback cannot permit the
        // solver to continue with a physically failed operation.
        std::abort();
    }
}
}} // namespace ugkp::diagnostics
#endif
