#ifndef UGKP_THERMAL_PARTICLE_RESTART_CODEC_H
#define UGKP_THERMAL_PARTICLE_RESTART_CODEC_H
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace Foam { namespace gpuThermalRestart {
static constexpr std::uint32_t maximumChunkParticles = 262144u;
static constexpr const char* currentChtMarker = "UGKP_CHT_PARTICLES_SCHEMA9_FLOAT_COLD_PROFILES_BIN";
static constexpr const char* currentFshMarker = "UGKP_FSH_PARTICLES_SCHEMA8_FP64_FIELDS_BIN";
template<class DiskTime> inline const char* marker();
template<> inline const char* marker<float>() { return "UGKP_THERMAL_PARTICLES_SCHEMA7_T32_BIN"; }
template<> inline const char* marker<double>() { return "UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN"; }
template<class View, class Fn>
inline void fields(const View& v, std::size_t begin, std::size_t count, Fn fn)
{
#include "GpuParticleRestartFields.inl"
}
// Scalar object bytes are stored least-significant-byte first. Matching hosts
// use one block transfer; other hosts reverse each scalar object byte sequence.
// This does not translate an unrelated host floating-point representation.
inline bool littleEndian()
{ const std::uint16_t one=1; return *reinterpret_cast<const unsigned char*>(&one)==1; }
template<class T> inline void writeArray(std::ostream& os,const T* p,std::size_t begin,std::size_t n)
{
    if (!n) return;
    if (!p) throw std::runtime_error("missing restart source field");
    if (littleEndian() || sizeof(T)==1) os.write(reinterpret_cast<const char*>(p+begin),n*sizeof(T));
    else for (std::size_t i=begin;i<begin+n;++i)
    {
        const char* bytes=reinterpret_cast<const char*>(p+i);
        for (std::size_t j=sizeof(T);j;--j) os.put(bytes[j-1]);
    }
    if (!os.good()) throw std::runtime_error("failed writing restart field");
}
template<class T> inline void readArray(std::istream& is,T* p,std::size_t begin,std::size_t n)
{
    if (!n) return;
    if (!p) throw std::runtime_error("missing restart destination field");
    if (littleEndian() || sizeof(T)==1) is.read(reinterpret_cast<char*>(p+begin),n*sizeof(T));
    else for (std::size_t i=begin;i<begin+n;++i)
    {
        char* bytes=reinterpret_cast<char*>(p+i);
        for (std::size_t j=sizeof(T);j;--j) is.get(bytes[j-1]);
    }
    if (!is.good()) throw std::runtime_error("truncated restart field");
}
template<class View> inline void validate(const View& v)
{
    fields(v,0u,v.count,[](const auto* p,std::size_t,std::size_t n)
    { if (n && !p) throw std::runtime_error("missing restart field"); });
}
inline void representation() {} // Kept for existing callers; endian conversion is at IO.
template<class DiskTime, class View> inline void timeType(const View&) {}
template<class Disk,class Memory>
inline void readConvertedImpl(std::istream& is,Memory* p,std::size_t b,std::size_t n,std::true_type)
{
    readArray(is,p,b,n);
}
template<class Disk,class Memory>
inline void readConvertedImpl(std::istream& is,Memory* p,std::size_t b,std::size_t n,std::false_type)
{
    if (n && !p) throw std::runtime_error("missing restart destination field");
    std::vector<Disk> values(n); readArray(is,values.data(),0,n);
    for (std::size_t i=0;i<n;++i)
    {
        const Memory value=static_cast<Memory>(values[i]);
        if (std::isfinite(values[i]) && !std::isfinite(value))
            throw std::runtime_error("restart time conversion overflow");
        p[b+i]=value;
    }
}
template<class Disk,class Memory> inline void readConverted(std::istream& is,Memory* p,std::size_t b,std::size_t n)
{
    readConvertedImpl<Disk>(is,p,b,n,std::is_same<Disk,Memory>{});
}
// Historical schema7 disk widths are independent of current memory storage.
template<class DiskTime,class View,class Fn>
inline void wireFields(const View& v,std::size_t begin,std::size_t count,Fn fn)
{
    fn(v.px, begin*1u, count*1u, double{});
    fn(v.py, begin*1u, count*1u, double{});
    fn(v.pz, begin*1u, count*1u, double{});
    fn(v.pux, begin*1u, count*1u, double{});
    fn(v.puy, begin*1u, count*1u, double{});
    fn(v.puz, begin*1u, count*1u, double{});
    fn(v.pT, begin*1u, count*1u, double{});
    fn(v.pTheta, begin*1u, count*1u, double{});
    fn(v.pd, begin*1u, count*1u, double{});
    fn(v.pm, begin*1u, count*1u, double{});
    fn(v.pCellId, begin*1u, count*1u, std::int32_t{});
    fn(v.pStatus, begin*1u, count*1u, std::int32_t{});
    fn(v.pRng, begin*1u, count*1u, std::uint64_t{});
    fn(v.pOrigId, begin*1u, count*1u, std::uint64_t{});
    fn(v.pStuck, begin*1u, count*1u, std::uint8_t{});
    fn(v.pStuckFaceId, begin*1u, count*1u, std::int32_t{});
    fn(v.pDepositionArea, begin*1u, count*1u, float{});
    fn(v.pContactDuration, begin*1u, count*1u, DiskTime{});
    fn(v.pContactMaximumArea, begin*1u, count*1u, float{});
    fn(v.pContactPeakFraction, begin*1u, count*1u, float{});
    fn(v.pColdNodeSpecificEnthalpy, begin*8u, count*8u, float{});
    fn(v.pColdRingSolidMass, begin*8u, count*8u, float{});
    fn(v.pColdFrozenArea, begin*1u, count*1u, float{});
    fn(v.pColdContactAge, begin*1u, count*1u, DiskTime{});
    fn(v.pCold2DNodeSpecificEnthalpy, begin*64u, count*64u, float{});
    fn(v.pCold2DRingContactAge, begin*8u, count*8u, DiskTime{});
    fn(v.pCold2DFrozenArea, begin*1u, count*1u, float{});
    fn(v.puxOld, begin*1u, count*1u, double{});
    fn(v.puyOld, begin*1u, count*1u, double{});
    fn(v.puzOld, begin*1u, count*1u, double{});
}
template<class DiskTime, class View>
inline void readPayload(std::istream& is,std::uint32_t bound,const View& v)
{
    timeType<DiskTime>(v);
    (void)bound;
    char sep=0; is.get(sep); if (sep=='\r') is.get(sep);
    if (sep!='\n') throw std::runtime_error("invalid schema7 header terminator");
    for (std::size_t begin=0;begin<v.count;)
    {
        std::uint32_t chunk=0;readArray(is,&chunk,0,1);
        if (!is.good() || !chunk || chunk>v.count-begin)
            throw std::runtime_error("invalid schema7 chunk size");
        wireFields<DiskTime>(v,begin,chunk,[&](auto* p,std::size_t first,std::size_t n,auto disk)
        {
            readConverted<decltype(disk)>(is,p,first,n);
        });
        begin+=chunk;
    }
}
// Current layouts keep the established field order, with explicit owner markers.
// Only CHT's two cold-profile fields remain binary32; its carry is binary64.
template<class View, class Fn>
inline void appendCurrentCarry(const View& v,std::size_t begin,std::size_t count,Fn& fn,std::true_type)
{
    fn(v.pThermalEnthalpyResidual,begin,count);
}
template<class View, class Fn>
inline void appendCurrentCarry(const View&,std::size_t,std::size_t,Fn&,std::false_type)
{}
template<bool HasCarry, class View, class Fn>
inline void currentFields(const View& v,std::size_t begin,std::size_t count,Fn fn)
{
    fields(v,begin,count,fn);
    appendCurrentCarry(v,begin,count,fn,std::integral_constant<bool,HasCarry>{});
}
template<bool HasCarry, class View>
inline void writeCurrent(std::ostream& os,const char* name,const View& v)
{
    os << name << ' ' << v.count << ' ' << maximumChunkParticles << '\n';
    for (std::size_t begin=0;begin<v.count;)
    {
        const std::uint32_t chunk=std::min<std::size_t>(maximumChunkParticles,v.count-begin);
        writeArray(os,&chunk,0,1);
        currentFields<HasCarry>(v,begin,chunk,[&](const auto* p,std::size_t b,std::size_t n)
        { writeArray(os,p,b,n); });
        begin+=chunk;
    }
    if (!os.good()) throw std::runtime_error("failed writing particle restart header");
}
template<bool HasCarry, class View>
inline void readCurrentPayload(std::istream& is,std::uint32_t bound,const View& v)
{
    (void)bound;
    char sep=0; is.get(sep); if (sep=='\r') is.get(sep);
    if (sep!='\n') throw std::runtime_error("invalid particle restart header terminator");
    for (std::size_t begin=0;begin<v.count;)
    {
        std::uint32_t chunk=0;readArray(is,&chunk,0,1);
        if (!is.good() || !chunk || chunk>v.count-begin)
            throw std::runtime_error("invalid particle restart chunk size");
        currentFields<HasCarry>(v,begin,chunk,[&](auto* p,std::size_t b,std::size_t n)
        { readArray(is,p,b,n); });
        begin+=chunk;
    }
}

} }
#endif
