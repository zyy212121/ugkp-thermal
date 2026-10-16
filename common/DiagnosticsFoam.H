#ifndef UGKP_DIAGNOSTICS_FOAM_H
#define UGKP_DIAGNOSTICS_FOAM_H
// This adapter belongs exclusively in the OpenFOAM host frontend. CUDA helpers
// exchange DiagnosticsCodec.H envelopes and do not link OpenFOAM.
#include "DiagnosticsHost.H"
#include "error.H"
#include "Pstream.H"
#include "IOstreams.H"

namespace ugkp { namespace diagnostics {
inline void foamFatal(void*, const Record& record, const SourceDescriptor* source) {
    Foam::FatalError(source ? source->function.c_str() : "ugkp::diagnostics::dispatch",
        source ? source->file.c_str() : "unknown diagnostic site", source ? static_cast<int>(source->line) : 0)
        << formatRecord(record,source).c_str() << Foam::exit(Foam::FatalError);
}
inline void foamWarning(void*, const WarningSummary& warning, const SourceDescriptor* source) {
    std::ostringstream message;
    message << formatRecord(warning.sample,source) << " occurrences=" << warning.count;
    if(warning.overflow)message << " (counter saturated; additional occurrences were not counted)";
    if (!Foam::Pstream::master()) {
        // Foam::Warning filters non-master ranks. Use OpenFOAM's rank-prefixed
        // local output stream with the same warning title instead of adding a
        // collective or delaying the warning indefinitely.
        Foam::Pout << Foam::Warning.title().c_str() << message.str().c_str() << Foam::endl;
        return;
    }
    Foam::Warning(source ? source->function.c_str() : "ugkp::diagnostics::dispatch",
        source ? source->file.c_str() : "unknown diagnostic site",source ? static_cast<int>(source->line) : 0)
        << message.str().c_str() << Foam::endl;
}
inline bool environmentWarningsEnabled() {
    const char* value=std::getenv("UGKP_DIAGNOSTIC_WARNINGS");
    if(!value)return true;
    const std::string setting(value);
    return setting!="0" && setting!="off" && setting!="false";
}
inline HostSink makeFoamSink(bool warningsEnabled = environmentWarningsEnabled()) {
    return HostSink{nullptr,foamFatal,foamWarning,warningsEnabled};
}
}} // namespace ugkp::diagnostics
#endif
