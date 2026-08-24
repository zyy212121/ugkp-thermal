#ifndef GpuFshParticleRestart_H
#define GpuFshParticleRestart_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace Foam
{
namespace gpuFshRestart
{

static const char* const formatName = "UGKP_FSH_PARTICLES_SCHEMA5_BIN";
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
    const float* pDepositionArea;
    const float* pContactDuration;
    const float* pContactMaximumArea;
    const float* pContactPeakFraction;
    const float* pColdNodeSpecificEnthalpy;
    const float* pColdRingSolidMass;
    const float* pColdFrozenArea;
    const float* pColdContactAge;
    const float* pCold2DNodeSpecificEnthalpy;
    const float* pCold2DRingContactAge;
    const float* pCold2DFrozenArea;
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
    float* pDepositionArea;
    float* pContactDuration;
    float* pContactMaximumArea;
    float* pContactPeakFraction;
    float* pColdNodeSpecificEnthalpy;
    float* pColdRingSolidMass;
    float* pColdFrozenArea;
    float* pColdContactAge;
    float* pCold2DNodeSpecificEnthalpy;
    float* pCold2DRingContactAge;
    float* pCold2DFrozenArea;
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

inline bool hostIsLittleEndian()
{
    const std::uint16_t marker = 1u;
    return *reinterpret_cast<const std::uint8_t*>(&marker) == 1u;
}

inline void validateHostRepresentation()
{
    if
    (
        !hostIsLittleEndian()
     || sizeof(double) != 8u
     || sizeof(float) != 4u
     || sizeof(std::int32_t) != 4u
     || sizeof(std::uint64_t) != 8u
    )
    {
        throw std::runtime_error
        (
            "UGKP FSH restart requires little-endian IEEE binary64/binary32 "
            "and 32-bit integer storage"
        );
    }
}

template<class T>
inline void writeArray
(
    std::ostream& os,
    const T* values,
    const std::size_t begin,
    const std::size_t count
)
{
    if (count == 0u)
    {
        return;
    }
    os.write
    (
        reinterpret_cast<const char*>(values + begin),
        static_cast<std::streamsize>(count*sizeof(T))
    );
    if (!os.good())
    {
        throw std::runtime_error("failed writing UGKP FSH restart chunk");
    }
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
    is.read
    (
        reinterpret_cast<char*>(values + begin),
        static_cast<std::streamsize>(count*sizeof(T))
    );
    if (!is.good())
    {
        throw std::runtime_error("truncated UGKP FSH restart chunk");
    }
}

inline void writeV4(std::ostream& os, const ConstView& v)
{
    validateHostRepresentation();
    os << schema4FormatName << ' ' << v.count << ' '
       << maximumChunkParticles << '\n';
    if (!os.good())
    {
        throw std::runtime_error("failed writing UGKP FSH restart header");
    }

    for (std::size_t begin = 0u; begin < v.count;)
    {
        const std::uint32_t chunk = static_cast<std::uint32_t>
        (
            std::min<std::size_t>(maximumChunkParticles, v.count - begin)
        );
        writeArray(os, &chunk, 0u, 1u);
        writeArray(os, v.px, begin, chunk);
        writeArray(os, v.py, begin, chunk);
        writeArray(os, v.pz, begin, chunk);
        writeArray(os, v.pux, begin, chunk);
        writeArray(os, v.puy, begin, chunk);
        writeArray(os, v.puz, begin, chunk);
        writeArray(os, v.pT, begin, chunk);
        writeArray(os, v.pTheta, begin, chunk);
        writeArray(os, v.pd, begin, chunk);
        writeArray(os, v.pm, begin, chunk);
        writeArray(os, v.pCellId, begin, chunk);
        writeArray(os, v.pStatus, begin, chunk);
        writeArray(os, v.pRng, begin, chunk);
        writeArray(os, v.pOrigId, begin, chunk);
        writeArray(os, v.pStuck, begin, chunk);
        writeArray(os, v.pStuckFaceId, begin, chunk);
        writeArray(os, v.pDepositionArea, begin, chunk);
        writeArray(os, v.pContactDuration, begin, chunk);
        writeArray(os, v.pContactMaximumArea, begin, chunk);
        writeArray(os, v.pContactPeakFraction, begin, chunk);
        writeArray
        (
            os,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        writeArray
        (
            os,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        writeArray(os, v.pColdFrozenArea, begin, chunk);
        writeArray(os, v.pColdContactAge, begin, chunk);
        begin += chunk;
    }
}

inline void writeV5(std::ostream& os, const ConstView& v)
{
    validateHostRepresentation();
    os << formatName << ' ' << v.count << ' '
       << maximumChunkParticles << '\n';
    if (!os.good())
    {
        throw std::runtime_error("failed writing UGKP FSH restart header");
    }

    for (std::size_t begin = 0u; begin < v.count;)
    {
        const std::uint32_t chunk = static_cast<std::uint32_t>
        (
            std::min<std::size_t>(maximumChunkParticles, v.count - begin)
        );
        writeArray(os, &chunk, 0u, 1u);
        writeArray(os, v.px, begin, chunk);
        writeArray(os, v.py, begin, chunk);
        writeArray(os, v.pz, begin, chunk);
        writeArray(os, v.pux, begin, chunk);
        writeArray(os, v.puy, begin, chunk);
        writeArray(os, v.puz, begin, chunk);
        writeArray(os, v.pT, begin, chunk);
        writeArray(os, v.pTheta, begin, chunk);
        writeArray(os, v.pd, begin, chunk);
        writeArray(os, v.pm, begin, chunk);
        writeArray(os, v.pCellId, begin, chunk);
        writeArray(os, v.pStatus, begin, chunk);
        writeArray(os, v.pRng, begin, chunk);
        writeArray(os, v.pOrigId, begin, chunk);
        writeArray(os, v.pStuck, begin, chunk);
        writeArray(os, v.pStuckFaceId, begin, chunk);
        writeArray(os, v.pDepositionArea, begin, chunk);
        writeArray(os, v.pContactDuration, begin, chunk);
        writeArray(os, v.pContactMaximumArea, begin, chunk);
        writeArray(os, v.pContactPeakFraction, begin, chunk);
        writeArray
        (
            os,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        writeArray
        (
            os,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        writeArray(os, v.pColdFrozenArea, begin, chunk);
        writeArray(os, v.pColdContactAge, begin, chunk);
        writeArray
        (
            os,
            v.pCold2DNodeSpecificEnthalpy,
            begin*coldWall2DNodeCount,
            chunk*coldWall2DNodeCount
        );
        writeArray
        (
            os,
            v.pCold2DRingContactAge,
            begin*coldWall2DRadialNodeCount,
            chunk*coldWall2DRadialNodeCount
        );
        writeArray(os, v.pCold2DFrozenArea, begin, chunk);
        begin += chunk;
    }
}

inline void readV1Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    validateHostRepresentation();
    if
    (
        declaredMaximumChunk == 0u
     || declaredMaximumChunk > maximumChunkParticles
    )
    {
        throw std::runtime_error("invalid UGKP FSH restart chunk bound");
    }

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
         || chunk > declaredMaximumChunk
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
        readArray(is, v.pDepositionArea, begin, chunk);
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

    char trailing = '\0';
    if (is.get(trailing))
    {
        throw std::runtime_error("trailing data in UGKP FSH restart file");
    }
}

inline void readV2Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    validateHostRepresentation();
    if
    (
        declaredMaximumChunk == 0u
     || declaredMaximumChunk > maximumChunkParticles
    )
    {
        throw std::runtime_error("invalid gasUGKP5 FSH restart chunk bound");
    }
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
            chunk == 0u || chunk > declaredMaximumChunk
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
        readArray(is, v.pDepositionArea, begin, chunk);
        readArray(is, v.pContactDuration, begin, chunk);
        readArray(is, v.pContactMaximumArea, begin, chunk);
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
    char trailing = '\0';
    if (is.get(trailing))
    {
        throw std::runtime_error("trailing data in gasUGKP5 FSH restart file");
    }
}

inline void readV3Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    validateHostRepresentation();
    if
    (
        declaredMaximumChunk == 0u
     || declaredMaximumChunk > maximumChunkParticles
    )
    {
        throw std::runtime_error("invalid UGKP FSH schema-3 restart chunk bound");
    }
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
            chunk == 0u || chunk > declaredMaximumChunk
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
        readArray(is, v.pDepositionArea, begin, chunk);
        readArray(is, v.pContactDuration, begin, chunk);
        readArray(is, v.pContactMaximumArea, begin, chunk);
        readArray(is, v.pContactPeakFraction, begin, chunk);
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
    char trailing = '\0';
    if (is.get(trailing))
    {
        throw std::runtime_error("trailing data in UGKP FSH schema-3 restart file");
    }
}

inline void readV4Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    validateHostRepresentation();
    if
    (
        declaredMaximumChunk == 0u
     || declaredMaximumChunk > maximumChunkParticles
    )
    {
        throw std::runtime_error("invalid UGKP FSH schema-4 restart chunk bound");
    }
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
            chunk == 0u || chunk > declaredMaximumChunk
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
        readArray(is, v.pDepositionArea, begin, chunk);
        readArray(is, v.pContactDuration, begin, chunk);
        readArray(is, v.pContactMaximumArea, begin, chunk);
        readArray(is, v.pContactPeakFraction, begin, chunk);
        readArray
        (
            is,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        readArray
        (
            is,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        readArray(is, v.pColdFrozenArea, begin, chunk);
        readArray(is, v.pColdContactAge, begin, chunk);
        clearColdWall2DRange(v, begin, chunk);
        begin += chunk;
    }
    char trailing = '\0';
    if (is.get(trailing))
    {
        throw std::runtime_error("trailing data in UGKP FSH schema-4 restart file");
    }
}

inline void readV5Payload
(
    std::istream& is,
    const std::uint32_t declaredMaximumChunk,
    const MutableView& v
)
{
    validateHostRepresentation();
    if
    (
        declaredMaximumChunk == 0u
     || declaredMaximumChunk > maximumChunkParticles
    )
    {
        throw std::runtime_error("invalid UGKP FSH schema-5 restart chunk bound");
    }
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
            chunk == 0u || chunk > declaredMaximumChunk
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
        readArray(is, v.pDepositionArea, begin, chunk);
        readArray(is, v.pContactDuration, begin, chunk);
        readArray(is, v.pContactMaximumArea, begin, chunk);
        readArray(is, v.pContactPeakFraction, begin, chunk);
        readArray
        (
            is,
            v.pColdNodeSpecificEnthalpy,
            begin*coldWallAxialNodeCount,
            chunk*coldWallAxialNodeCount
        );
        readArray
        (
            is,
            v.pColdRingSolidMass,
            begin*coldWallRadialRingCount,
            chunk*coldWallRadialRingCount
        );
        readArray(is, v.pColdFrozenArea, begin, chunk);
        readArray(is, v.pColdContactAge, begin, chunk);
        readArray
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
        readArray(is, v.pCold2DFrozenArea, begin, chunk);
        begin += chunk;
    }
    char trailing = '\0';
    if (is.get(trailing))
    {
        throw std::runtime_error
        (
            "trailing data in UGKP FSH schema-5 restart file"
        );
    }
}

}                           
}                  

#endif
