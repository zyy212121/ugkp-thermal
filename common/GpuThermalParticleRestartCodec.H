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
namespace Foam { namespace gpuThermalRestart {
static constexpr std::uint32_t maximumChunkParticles = 262144u;
template<class DiskTime> inline const char* marker();
template<> inline const char* marker<float>() { return "UGKP_THERMAL_PARTICLES_SCHEMA7_T32_BIN"; }
template<> inline const char* marker<double>() { return "UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN"; }
template<class View, class Fn>
inline void fields(const View& v, std::size_t begin, std::size_t count, Fn fn)
{
#include "GpuParticleRestartFields.inl"
}
template<class View> inline void validate(const View& v)
{
    fields(v, 0u, v.count, [](const auto* values, std::size_t begin, std::size_t count)
    {
        if (count && !values) throw std::runtime_error("missing schema7 particle field");
        typedef typename std::remove_cv<typename std::remove_pointer<decltype(values)>::type>::type T;
        if (std::is_floating_point<T>::value)
            for (std::size_t i=begin; i<begin+count; ++i)
                if (!std::isfinite(static_cast<double>(values[i])))
                    throw std::runtime_error("nonfinite schema7 particle field");
    });
}
inline void representation()
{
    const std::uint16_t one=1;
    if (*reinterpret_cast<const unsigned char*>(&one)!=1 || sizeof(float)!=4
        || sizeof(double)!=8 || !std::numeric_limits<float>::is_iec559
        || !std::numeric_limits<double>::is_iec559)
        throw std::runtime_error("schema7 requires little-endian IEEE binary storage");
}
template<class DiskTime, class View> inline void timeType(const View& v)
{
    typedef typename std::remove_cv<typename std::remove_pointer<decltype(v.pContactDuration)>::type>::type T;
    static_assert(std::is_same<T,DiskTime>::value,"restart time marker must match native time type");
}
template<class DiskTime, class View> inline void write(std::ostream& os, const View& v)
{
    representation(); timeType<DiskTime>(v); validate(v);
    os << marker<DiskTime>() << ' ' << v.count << ' ' << maximumChunkParticles << '\n';
    for (std::size_t begin=0; begin<v.count;)
    {
        const std::uint32_t chunk=std::min<std::size_t>(maximumChunkParticles,v.count-begin);
        os.write(reinterpret_cast<const char*>(&chunk),sizeof(chunk));
        fields(v,begin,chunk,[&](const auto* values,std::size_t first,std::size_t n)
        {
            os.write(reinterpret_cast<const char*>(values+first),n*sizeof(*values));
            if (!os.good()) throw std::runtime_error("failed writing schema7 chunk");
        });
        begin+=chunk;
    }
    if (!os.good()) throw std::runtime_error("failed writing schema7 header");
}
template<class DiskTime, class View>
inline void readPayload(std::istream& is,std::uint32_t bound,const View& v)
{
    representation(); timeType<DiskTime>(v);
    if (!bound || bound>maximumChunkParticles) throw std::runtime_error("invalid schema7 chunk bound");
    char sep=0; is.get(sep); if (sep=='\r') is.get(sep);
    if (sep!='\n') throw std::runtime_error("invalid schema7 header terminator");
    fields(v,0u,v.count,[](auto* p,std::size_t,std::size_t n)
    { if (n && !p) throw std::runtime_error("missing schema7 destination"); });
    for (std::size_t begin=0;begin<v.count;)
    {
        std::uint32_t chunk=0;is.read(reinterpret_cast<char*>(&chunk),sizeof(chunk));
        if (!is.good() || !chunk || chunk>bound || chunk>v.count-begin)
            throw std::runtime_error("invalid schema7 chunk size");
        fields(v,begin,chunk,[&](auto* p,std::size_t first,std::size_t n)
        {
            is.read(reinterpret_cast<char*>(p+first),n*sizeof(*p));
            if (!is.good()) throw std::runtime_error("truncated schema7 chunk");
        });
        begin+=chunk;
    }
    if (is.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing schema7 data");
    validate(v);
}
template<class View> inline void initializeLegacySavedVelocity(const View& v)
{
    for (std::size_t i=0;i<v.count;++i)
        if (v.pStuck[i]==2u || v.pStuck[i]==3u)
            throw std::runtime_error("legacy transient restart lacks saved impact velocity; use schema7 or a fresh state");
    for (std::size_t i=0;i<v.count;++i) v.puxOld[i]=v.puyOld[i]=v.puzOld[i]=0.0;
}
} }
#endif
