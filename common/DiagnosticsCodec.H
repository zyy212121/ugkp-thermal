#ifndef UGKP_DIAGNOSTICS_CODEC_H
#define UGKP_DIAGNOSTICS_CODEC_H
#include "DiagnosticsHost.H"
#include <cstring>
#include <limits>

namespace ugkp { namespace diagnostics {
namespace wire {
// Little-endian fixed widths, IEEE binary64; independent of compiler padding.
constexpr std::uint32_t magic = 0x444b4755u; // UGKD
constexpr std::uint32_t version = 1u;
inline void u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned i=0;i<4;++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8*i)));
}
inline void u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
    for (unsigned i=0;i<8;++i) bytes.push_back(static_cast<std::uint8_t>(value >> (8*i)));
}
inline void real(std::vector<std::uint8_t>& bytes, double value) {
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"binary64 required");
    std::uint64_t bits; std::memcpy(&bits,&value,8); u64(bytes,bits);
}
inline void string(std::vector<std::uint8_t>& bytes, const std::string& value) {
    u64(bytes,static_cast<std::uint64_t>(value.size())); bytes.insert(bytes.end(),value.begin(),value.end());
}
inline void record(std::vector<std::uint8_t>& bytes, const Record& r) {
    u32(bytes,static_cast<std::uint32_t>(r.severity)); u32(bytes,static_cast<std::uint32_t>(r.code));
    u32(bytes,r.sourceSiteId); u32(bytes,r.operatorId); u32(bytes,r.stageId); u32(bytes,static_cast<std::uint32_t>(r.entityKind));
    u64(bytes,static_cast<std::uint64_t>(r.entityIndex)); u32(bytes,static_cast<std::uint32_t>(r.component)); u32(bytes,r.metadata); u32(bytes,r.quantityId);
    u64(bytes,r.operationSequence); u64(bytes,r.instanceId); u64(bytes,static_cast<std::uint64_t>(r.step)); real(bytes,r.simulationTime);
    real(bytes,r.observed); real(bytes,r.reference); real(bytes,r.corrected);
}
struct Reader {
    const std::uint8_t* data; std::size_t size, offset = 0; bool good = true;
    bool available(std::size_t n) { if (!good || n > size-offset) { good=false; return false; } return true; }
    std::uint32_t u32() { if(!available(4)) return 0; std::uint32_t value=0; for(unsigned i=0;i<4;++i)value|=std::uint32_t(data[offset++])<<(8*i); return value; }
    std::uint64_t u64() { if(!available(8)) return 0; std::uint64_t value=0; for(unsigned i=0;i<8;++i)value|=std::uint64_t(data[offset++])<<(8*i); return value; }
    double real() { const std::uint64_t bits=u64(); double value; std::memcpy(&value,&bits,8); return value; }
    std::string string() { const auto n=u64(); if(n>size-offset || !good) {good=false;return {};}
        std::string value(reinterpret_cast<const char*>(data+offset),static_cast<std::size_t>(n));offset+=static_cast<std::size_t>(n);return value; }
    Record record() {
        Record r{};r.severity=static_cast<Severity>(u32());r.code=static_cast<Code>(u32());
        r.sourceSiteId=u32();r.operatorId=u32();r.stageId=u32();r.entityKind=static_cast<EntityKind>(u32());
        const auto index=u64();std::memcpy(&r.entityIndex,&index,8);
        const auto component=u32();std::memcpy(&r.component,&component,4);r.metadata=u32();r.quantityId=u32();
        r.operationSequence=u64();r.instanceId=u64();const auto step=u64();std::memcpy(&r.step,&step,8);r.simulationTime=real();r.observed=real();r.reference=real();r.corrected=real();return r;
    }
};
}
inline std::vector<std::uint8_t> encode(const Snapshot& snapshot) {
    std::vector<std::uint8_t> bytes;
    wire::u32(bytes,wire::magic);wire::u32(bytes,wire::version);
    wire::u64(bytes,snapshot.operationSequence);wire::u64(bytes,snapshot.instanceId);
    wire::u32(bytes,snapshot.hasFatal?1u:0u);
    wire::string(bytes,snapshot.catalogueId);
    if(snapshot.hasFatal)wire::record(bytes,snapshot.fatal);
    wire::u64(bytes,static_cast<std::uint64_t>(snapshot.warnings.size()));
    for(const auto& w:snapshot.warnings) {wire::record(bytes,w.sample);wire::u32(bytes,w.count);wire::u32(bytes,w.overflow?1u:0u);}
    wire::u64(bytes,static_cast<std::uint64_t>(snapshot.sources.size()));
    for(const auto& source:snapshot.sources) {
        wire::u32(bytes,source.siteId);wire::string(bytes,source.file);wire::string(bytes,source.function);
        wire::u32(bytes,source.line);wire::string(bytes,source.description);
    }
    return bytes;
}
inline bool decode(const std::uint8_t* data, std::size_t size, Snapshot& output, std::string& error) {
    if (!data && size) {error="null diagnostic envelope";return false;}
    wire::Reader reader{data,size}; Snapshot snapshot;
    if(reader.u32()!=wire::magic || reader.u32()!=wire::version) {error="diagnostic envelope magic/version mismatch";return false;}
    snapshot.operationSequence=reader.u64();snapshot.instanceId=reader.u64();
    const auto hasFatal=reader.u32();if(hasFatal>1) {error="invalid diagnostic fatal flag";return false;}
    snapshot.hasFatal=hasFatal!=0;snapshot.catalogueId=reader.string();
    if(snapshot.hasFatal)snapshot.fatal=reader.record();
    const auto warnings=reader.u64();
    if(!reader.good || warnings>(size-reader.offset)/108u) {error="truncated diagnostic warning records";return false;}
    for(std::uint64_t i=0;i<warnings;++i) { WarningSummary w{};w.sample=reader.record();w.count=reader.u32();const auto overflow=reader.u32();
        if(overflow>1) {error="invalid diagnostic overflow flag";return false;}w.overflow=overflow!=0;snapshot.warnings.push_back(w); }
    const auto sources=reader.u64();
    if(!reader.good || sources>(size-reader.offset)/32u) {error="truncated diagnostic source descriptors";return false;}
    for(std::uint64_t i=0;i<sources && reader.good;++i) {SourceDescriptor source{};source.siteId=reader.u32();source.file=reader.string();
        source.function=reader.string();source.line=reader.u32();source.description=reader.string();snapshot.sources.push_back(source);}
    if(!reader.good || reader.offset!=size) {error="truncated or trailing diagnostic envelope bytes";return false;}
    output=std::move(snapshot);error.clear();return true;
}
}} // namespace ugkp::diagnostics
#endif
