#pragma once
// Test-only producer for historical bytes; not included by solver code.
#include "common/GpuThermalParticleRestartCodec.H"
namespace Foam { namespace gpuThermalRestart {
template<class Disk,class Memory> inline void writeConverted(std::ostream& os,const Memory* p,std::size_t b,std::size_t n)
{
    if constexpr (std::is_same<Disk,Memory>::value) writeArray(os,p,b,n);
    else
    {
        if (n && !p) throw std::runtime_error("missing restart source field");
        std::vector<Disk> values(n);
        for (std::size_t i=0;i<n;++i)
        {
            values[i]=static_cast<Disk>(p[b+i]);
            if (std::isfinite(p[b+i]) && !std::isfinite(values[i]))
                throw std::runtime_error("restart time conversion overflow");
        }
        writeArray(os,values.data(),0,n);
    }
}
template<class DiskTime, class View> inline void write(std::ostream& os, const View& v)
{
    timeType<DiskTime>(v);
    os << marker<DiskTime>() << ' ' << v.count << ' ' << maximumChunkParticles << '\n';
    for (std::size_t begin=0; begin<v.count;)
    {
        const std::uint32_t chunk=std::min<std::size_t>(maximumChunkParticles,v.count-begin);
        writeArray(os,&chunk,0,1);
        wireFields<DiskTime>(v,begin,chunk,[&](const auto* values,std::size_t first,std::size_t n,auto disk)
        {
            writeConverted<decltype(disk)>(os,values,first,n);
        });
        begin+=chunk;
    }
    if (!os.good()) throw std::runtime_error("failed writing schema7 header");
}
} }
