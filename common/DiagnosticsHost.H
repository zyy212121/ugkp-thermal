#ifndef UGKP_DIAGNOSTICS_HOST_H
#define UGKP_DIAGNOSTICS_HOST_H
#include "Diagnostics.H"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

inline void selectSnapshotSources(Snapshot& result,const std::vector<SourceDescriptor>& sources) {
    result.sources.clear();
    if(!result.hasFatal&&result.warnings.empty())return;


    for(const auto& source:sources) {
        bool used=result.hasFatal && result.fatal.sourceSiteId==source.siteId;
        for(const auto& warning:result.warnings)used=used||warning.sample.sourceSiteId==source.siteId;
        if(used)result.sources.push_back(source);
    }
}

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
        case Code::OutputData: return "accepted output could not be published";
        case Code::InputData: return "required input data unavailable or invalid";
        case Code::ConservationResidual: return "conservation residual";
        case Code::AccuracyResidual: return "accuracy residual";
        case Code::FiniteYClip: return "finite mass fraction clipped to [0,1]";
        case Code::RestartMetadata: return "restart metadata advisory";
        case Code::MethodFallback: return "bounded algorithm safeguard";
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
        case Quantity::RadiationIntensity:return "radiation intensity";
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

struct RecordedFailure : std::runtime_error {
    Record record;
    SourceDescriptor source;
    RecordedFailure(Record value,SourceDescriptor location)
        :std::runtime_error(formatRecord(value,&location)),record(value),source(std::move(location)) {}
};
struct HostSink {
    void* user = nullptr;
    void (*fatal)(void*, const Record&, const SourceDescriptor*) = nullptr;
    void (*warning)(void*, const WarningSummary&, const SourceDescriptor*) = nullptr;
    bool warningsEnabled = true;
};


inline void dispatch(const Snapshot& snapshot, const HostSink& sink) {
    if (sink.warningsEnabled && sink.warning)
        for (const auto& warning : snapshot.warnings)
            sink.warning(sink.user, warning, findSource(snapshot, warning.sample.sourceSiteId));
    if (snapshot.hasFatal) {
        if (sink.fatal) sink.fatal(sink.user, snapshot.fatal, findSource(snapshot, snapshot.fatal.sourceSiteId));


        std::abort();
    }
}


inline bool hostWarningsEnabled() {
    const char* value=std::getenv("UGKP_DIAGNOSTIC_WARNINGS");
    if(!value)return true;
    const std::string setting(value);
    return setting!="0" && setting!="off" && setting!="false";
}


class ScopedHostWarningCapture {
    inline static thread_local ScopedHostWarningCapture* current_=nullptr;
    ScopedHostWarningCapture* previous_;
    DeviceContext& context_;
    std::vector<SourceDescriptor>& sources_;
public:
    ScopedHostWarningCapture(DeviceContext& context,std::vector<SourceDescriptor>& sources)
      :previous_(current_),context_(context),sources_(sources){current_=this;}
    ScopedHostWarningCapture(const ScopedHostWarningCapture&)=delete;
    ScopedHostWarningCapture& operator=(const ScopedHostWarningCapture&)=delete;
    ~ScopedHostWarningCapture(){current_=previous_;}
    static bool capture(const Record& report,const SourceDescriptor& source) {
        if (!current_ || !isWarning(report.code)) return false;
        auto& context=current_->context_;
        const auto index=static_cast<std::uint32_t>(report.code)-101u;
        if (!context.warnings[index].count) current_->sources_.push_back(source);
        recordWarning(&context,report);
        return true;
    }
};
inline void hostWarning(Code code,const SourceDescriptor& source,double observed=0,
    double reference=0,double corrected=0) {
    const auto record=makeRecord(code,source.siteId,0,0,EntityKind::None,-1,-1,
        observed,reference,corrected);
    if(ScopedHostWarningCapture::capture(record,source)||!hostWarningsEnabled())return;
    std::cerr << "Warning: " << formatRecord(record,&source) << '\n';
}
inline std::string inputError(const SourceDescriptor& source) {
    return formatRecord(makeRecord(Code::InputData,source.siteId),&source);
}
}}
#endif
