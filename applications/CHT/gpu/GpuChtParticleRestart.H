#ifndef GpuFshParticleRestart_H
#define GpuFshParticleRestart_H

#include <algorithm>
#include "../../../common/GpuThermalParticleRestartCodec.H"
#include "../../../common/GpuParticleCaloric.H"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <vector>
#include <stdexcept>

namespace Foam
{
namespace gpuFshRestart
{

static const char* const formatName = "UGKP_FSH_PARTICLES_SCHEMA6_BIN";
static const char* const schema5FormatName = "UGKP_FSH_PARTICLES_SCHEMA5_BIN";
static const char* const schema4FormatName = "UGKP_FSH_PARTICLES_SCHEMA4_BIN";
static const char* const schema3FormatName = "UGKP_FSH_PARTICLES_SCHEMA3_BIN";
static const char* const schema2FormatName = "UGKP_FSH_PARTICLES_SCHEMA2_BIN";
static const char* const legacyFormatName = "UGKP_FSH_PARTICLES_SCHEMA1_BIN";
static constexpr std::uint32_t maximumChunkParticles = 262144u;

struct ConstView
{
    std::size_t count;
    const double* px;
    const double* py;
    const double* pz;
    const double* pux;
    const double* puy;
    const double* puz;
    const double* pT;
    const double* pTheta;
    const double* pd;
    const double* pm;
    const std::int32_t* pCellId;
    const std::int32_t* pStatus;
    const std::uint64_t* pRng;
    const std::uint64_t* pOrigId;
    const std::uint8_t* pStuck;
    const std::int32_t* pStuckFaceId;
    const double* pDepositionArea;
    const double* pContactDuration;
    const double* pContactMaximumArea;
    const double* pContactPeakFraction;
    const float* pColdNodeSpecificEnthalpy;
    const double* pColdRingSolidMass;
    const double* pColdFrozenArea;
    const double* pColdContactAge;
    const float* pCold2DNodeSpecificEnthalpy;
    const double* pCold2DRingContactAge;
    const double* pCold2DFrozenArea;
    const double* puxOld;
    const double* puyOld;
    const double* puzOld;
    const double* pThermalEnthalpyResidual = nullptr;
};

struct MutableView
{
    std::size_t count;
    double* px;
    double* py;
    double* pz;
    double* pux;
    double* puy;
    double* puz;
    double* pT;
    double* pTheta;
    double* pd;
    double* pm;
    std::int32_t* pCellId;
    std::int32_t* pStatus;
    std::uint64_t* pRng;
    std::uint64_t* pOrigId;
    std::uint8_t* pStuck;
    std::int32_t* pStuckFaceId;
    double* pDepositionArea;
    double* pContactDuration;
    double* pContactMaximumArea;
    double* pContactPeakFraction;
    float* pColdNodeSpecificEnthalpy;
    double* pColdRingSolidMass;
    double* pColdFrozenArea;
    double* pColdContactAge;
    float* pCold2DNodeSpecificEnthalpy;
    double* pCold2DRingContactAge;
    double* pCold2DFrozenArea;
    double* puxOld;
    double* puyOld;
    double* puzOld;
    double* pThermalEnthalpyResidual = nullptr;
};

static constexpr std::size_t coldWallAxialNodeCount = 8u;
static constexpr std::size_t coldWallRadialRingCount = 8u;
static constexpr std::size_t coldWall2DRadialNodeCount = 8u;
static constexpr std::size_t coldWall2DAxialNodeCount = 8u;
static constexpr std::size_t coldWall2DNodeCount =
    coldWall2DRadialNodeCount*coldWall2DAxialNodeCount;

inline void clearColdWall2DRange
(
    const MutableView& v,
    const std::size_t begin,
    const std::size_t count
)
{
    std::fill
    (
        v.pCold2DNodeSpecificEnthalpy + begin*coldWall2DNodeCount,
        v.pCold2DNodeSpecificEnthalpy + (begin + count)*coldWall2DNodeCount,
        0.0f
    );
    std::fill
    (
        v.pCold2DRingContactAge + begin*coldWall2DRadialNodeCount,
        v.pCold2DRingContactAge
          + (begin + count)*coldWall2DRadialNodeCount,
        0.0f
    );
    std::fill
    (
        v.pCold2DFrozenArea + begin,
        v.pCold2DFrozenArea + begin + count,
        0.0f
    );
}

template<class T>
inline void readArray
(
    std::istream& is,
    T* values,
    const std::size_t begin,
    const std::size_t count
)
{
    if (count == 0u)
    {
        return;
    }
    gpuThermalRestart::readArray(is, values, begin, count);
}

template<class Disk, class Memory>
inline void readConvertedArray(std::istream& is, Memory* values, std::size_t begin, std::size_t count)
{
    std::vector<Disk> staging(count);
    readArray(is, staging.data(), 0u, count);
    for (std::size_t i=0; i<count; ++i) values[begin+i] = static_cast<Memory>(staging[i]);
}
inline void readV1Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;

    char separator = '\0';
    is.get(separator);
    if (separator == '\r')
    {
        is.get(separator);
    }
    if (separator != '\n')
    {
        throw std::runtime_error("invalid UGKP FSH restart header terminator");
    }

    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u

         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error("invalid UGKP FSH restart chunk size");
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        std::fill
        (
            v.pContactDuration + begin,
            v.pContactDuration + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pContactMaximumArea + begin,
            v.pContactMaximumArea + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pContactPeakFraction + begin,
            v.pContactPeakFraction + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pColdNodeSpecificEnthalpy + begin*coldWallAxialNodeCount,
            v.pColdNodeSpecificEnthalpy
              + (begin + chunk)*coldWallAxialNodeCount,
            0.0f
        );
        std::fill
        (
            v.pColdRingSolidMass + begin*coldWallRadialRingCount,
            v.pColdRingSolidMass
              + (begin + chunk)*coldWallRadialRingCount,
            0.0f
        );
        std::fill
        (
            v.pColdFrozenArea + begin,
            v.pColdFrozenArea + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pColdContactAge + begin,
            v.pColdContactAge + begin + chunk,
            0.0f
        );
        clearColdWall2DRange(v, begin, chunk);
        begin += chunk;
    }

}

inline void readV2Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;
    char separator = '\0';
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n')
    {
        throw std::runtime_error("invalid gasUGKP5 FSH restart header terminator");
    }
    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u
         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error("invalid gasUGKP5 FSH restart chunk size");
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        readConvertedArray<float>(is, v.pContactDuration, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactMaximumArea, begin, chunk);
        for (std::size_t i = begin; i < begin + chunk; ++i)
        {
            v.pContactPeakFraction[i] =
                v.pStuck[i] == 2u || v.pStuck[i] == 3u ? 0.5f : 0.0f;
        }
        std::fill
        (
            v.pColdNodeSpecificEnthalpy + begin*coldWallAxialNodeCount,
            v.pColdNodeSpecificEnthalpy
              + (begin + chunk)*coldWallAxialNodeCount,
            0.0f
        );
        std::fill
        (
            v.pColdRingSolidMass + begin*coldWallRadialRingCount,
            v.pColdRingSolidMass
              + (begin + chunk)*coldWallRadialRingCount,
            0.0f
        );
        std::fill
        (
            v.pColdFrozenArea + begin,
            v.pColdFrozenArea + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pColdContactAge + begin,
            v.pColdContactAge + begin + chunk,
            0.0f
        );
        clearColdWall2DRange(v, begin, chunk);
        begin += chunk;
    }
}

inline void readV3Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;
    char separator = '\0';
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n')
    {
        throw std::runtime_error("invalid UGKP FSH schema-3 restart header terminator");
    }
    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u
         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error("invalid UGKP FSH schema-3 restart chunk size");
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        readConvertedArray<float>(is, v.pContactDuration, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactMaximumArea, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactPeakFraction, begin, chunk);
        std::fill
        (
            v.pColdNodeSpecificEnthalpy + begin*coldWallAxialNodeCount,
            v.pColdNodeSpecificEnthalpy
              + (begin + chunk)*coldWallAxialNodeCount,
            0.0f
        );
        std::fill
        (
            v.pColdRingSolidMass + begin*coldWallRadialRingCount,
            v.pColdRingSolidMass
              + (begin + chunk)*coldWallRadialRingCount,
            0.0f
        );
        std::fill
        (
            v.pColdFrozenArea + begin,
            v.pColdFrozenArea + begin + chunk,
            0.0f
        );
        std::fill
        (
            v.pColdContactAge + begin,
            v.pColdContactAge + begin + chunk,
            0.0f
        );
        clearColdWall2DRange(v, begin, chunk);
        begin += chunk;
    }
}

inline void readV4Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;
    char separator = '\0';
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n')
    {
        throw std::runtime_error("invalid UGKP FSH schema-4 restart header terminator");
    }
    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u
         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error("invalid UGKP FSH schema-4 restart chunk size");
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        readConvertedArray<float>(is, v.pContactDuration, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactMaximumArea, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactPeakFraction, begin, chunk);
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        gpuThermalRestart::readConverted<float>(is, v.pColdFrozenArea, begin, chunk);
        readConvertedArray<float>(is, v.pColdContactAge, begin, chunk);
        clearColdWall2DRange(v, begin, chunk);
        begin += chunk;
    }
}

inline void readV5Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;
    char separator = '\0';
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n')
    {
        throw std::runtime_error
        (
            "invalid UGKP FSH schema-5 restart header terminator"
        );
    }
    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u
         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error
            (
                "invalid UGKP FSH schema-5 restart chunk size"
            );
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        readConvertedArray<float>(is, v.pContactDuration, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactMaximumArea, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactPeakFraction, begin, chunk);
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        gpuThermalRestart::readConverted<float>(is, v.pColdFrozenArea, begin, chunk);
        readConvertedArray<float>(is, v.pColdContactAge, begin, chunk);
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pCold2DNodeSpecificEnthalpy,
            begin*coldWall2DNodeCount,
            chunk*coldWall2DNodeCount
        );
        readConvertedArray<float>
        (
            is,
            v.pCold2DRingContactAge,
            begin*coldWall2DRadialNodeCount,
            chunk*coldWall2DRadialNodeCount
        );
        gpuThermalRestart::readConverted<float>(is, v.pCold2DFrozenArea, begin, chunk);
        begin += chunk;
    }
}

inline void readV6Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    (void)declaredMaximumChunk;
    char separator = '\0';
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n')
    {
        throw std::runtime_error
        (
            "invalid UGKP FSH schema-6 restart header terminator"
        );
    }
    std::size_t begin = 0u;
    while (begin < v.count)
    {
        std::uint32_t chunk = 0u;
        readArray(is, &chunk, 0u, 1u);
        if
        (
            chunk == 0u
         || static_cast<std::size_t>(chunk) > v.count - begin
        )
        {
            throw std::runtime_error
            (
                "invalid UGKP FSH schema-6 restart chunk size"
            );
        }
        readArray(is, v.px, begin, chunk);
        readArray(is, v.py, begin, chunk);
        readArray(is, v.pz, begin, chunk);
        readArray(is, v.pux, begin, chunk);
        readArray(is, v.puy, begin, chunk);
        readArray(is, v.puz, begin, chunk);
        readArray(is, v.pT, begin, chunk);
        readArray(is, v.pTheta, begin, chunk);
        readArray(is, v.pd, begin, chunk);
        readArray(is, v.pm, begin, chunk);
        readArray(is, v.pCellId, begin, chunk);
        readArray(is, v.pStatus, begin, chunk);
        readArray(is, v.pRng, begin, chunk);
        readArray(is, v.pOrigId, begin, chunk);
        readArray(is, v.pStuck, begin, chunk);
        readArray(is, v.pStuckFaceId, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pDepositionArea, begin, chunk);
        readArray(is, v.pContactDuration, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactMaximumArea, begin, chunk);
        gpuThermalRestart::readConverted<float>(is, v.pContactPeakFraction, begin, chunk);
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        gpuThermalRestart::readConverted<float>(is, v.pColdFrozenArea, begin, chunk);
        readArray(is, v.pColdContactAge, begin, chunk);
        gpuThermalRestart::readConverted<float>
        (
            is,
            v.pCold2DNodeSpecificEnthalpy,
            begin*coldWall2DNodeCount,
            chunk*coldWall2DNodeCount
        );
        readArray
        (
            is,
            v.pCold2DRingContactAge,
            begin*coldWall2DRadialNodeCount,
            chunk*coldWall2DRadialNodeCount
        );
        gpuThermalRestart::readConverted<float>(is, v.pCold2DFrozenArea, begin, chunk);
        begin += chunk;
    }
}

static const char* const schema7FormatName = gpuThermalRestart::marker<double>();
inline void readV7Payload(std::istream& is,std::uint32_t bound,const MutableView& v)
{
    gpuThermalRestart::readPayload<double>(is,bound,v);
    if (v.pThermalEnthalpyResidual != nullptr)
        std::fill(v.pThermalEnthalpyResidual, v.pThermalEnthalpyResidual + v.count, 0.0);
}

// Schema 8 retains all schema 7 fields and appends one binary64 carry per
// particle to each chunk. No old field type, order or caloric meaning changes.
static const char* const schema8FormatName = "UGKP_CHT_PARTICLES_SCHEMA8_THERMAL_CARRY_BIN";
inline void readV8Payload(std::istream& is, std::uint32_t bound, const MutableView& v)
{
    gpuThermalRestart::timeType<double>(v);
    (void)bound;
    char separator = 0;
    is.get(separator);
    if (separator == '\r') is.get(separator);
    if (separator != '\n') throw std::runtime_error("invalid schema8 header terminator");
    for (std::size_t begin = 0; begin < v.count;)
    {
        std::uint32_t chunk = 0;
        gpuThermalRestart::readArray(is, &chunk, 0, 1);
        if (!is.good() || !chunk || chunk > v.count - begin)
            throw std::runtime_error("invalid schema8 chunk size");
        gpuThermalRestart::wireFields<double>(v, begin, chunk, [&](auto* values, std::size_t first, std::size_t n, auto disk)
        {
            gpuThermalRestart::readConverted<decltype(disk)>(is, values, first, n);
        });
        gpuThermalRestart::readArray(is, v.pThermalEnthalpyResidual, begin, chunk);
        begin += chunk;
    }

}


static const char* const currentFormatName = gpuThermalRestart::currentChtMarker;
inline void writeCurrent(std::ostream& os,const ConstView& v)
{ gpuThermalRestart::writeCurrent<true>(os,currentFormatName,v); }
inline void readCurrentPayload(std::istream& is,std::uint32_t bound,const MutableView& v)
{ gpuThermalRestart::readCurrentPayload<true>(is,bound,v); }

}
}

#endif
