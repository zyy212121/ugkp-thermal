#include "GpuPrecisionTypes.H"
#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define UGKP_CHARACTERISTIC_HD __host__ __device__ __forceinline__
#else
#define UGKP_CHARACTERISTIC_HD inline
#endif

namespace ugkpcharacteristic
{

struct Increment
{
    GpuReal rho;
    GpuReal ux;
    GpuReal uy;
    GpuReal uz;
    GpuReal p;
};

UGKP_CHARACTERISTIC_HD GpuReal clamp01(const GpuReal value)
{
    return value < GPU_R(0.0) ? GPU_R(0.0) : (value > GPU_R(1.0) ? GPU_R(1.0) : value);
}

UGKP_CHARACTERISTIC_HD GpuReal boundedTowardAdjacent
(
    const GpuReal raw,
    const GpuReal adjacentDifference,
    const GpuReal faceFraction
)
{
    if
    (
        !::isfinite(raw)
     || !::isfinite(adjacentDifference)
     || raw*adjacentDifference <= GPU_R(0.0)
    )
    {
        return GPU_R(0.0);
    }
    const GpuReal bound =
        clamp01(faceFraction)*::fabs(adjacentDifference);
    return ::copysign(::fmin(::fabs(raw), bound), raw);
}

UGKP_CHARACTERISTIC_HD Increment limitOneSide
(
    const Increment& raw,
    const Increment& adjacentDifference,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal roeDensity,
    const GpuReal roeSoundSpeed,
    const GpuReal faceFraction
)
{
    const GpuReal soundSquared = roeSoundSpeed*roeSoundSpeed;
    const GpuReal impedance = roeDensity*roeSoundSpeed;
    if
    (
        !::isfinite(soundSquared)
     || !::isfinite(impedance)
     || soundSquared <= GPU_R(0.0)
     || impedance <= GPU_R(0.0)
    )
    {
        return Increment{GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0), GPU_R(0.0)};
    }

    const GpuReal rawNormal = raw.ux*nx + raw.uy*ny + raw.uz*nz;
    const GpuReal adjacentNormal =
        adjacentDifference.ux*nx
      + adjacentDifference.uy*ny
      + adjacentDifference.uz*nz;
    const GpuReal rawMinus = raw.p - impedance*rawNormal;
    const GpuReal rawContact = raw.rho - raw.p/soundSquared;
    const GpuReal rawPlus = raw.p + impedance*rawNormal;
    const GpuReal adjacentMinus =
        adjacentDifference.p - impedance*adjacentNormal;
    const GpuReal adjacentContact =
        adjacentDifference.rho
      - adjacentDifference.p/soundSquared;
    const GpuReal adjacentPlus =
        adjacentDifference.p + impedance*adjacentNormal;

    const GpuReal limitedMinus = boundedTowardAdjacent
    (
        rawMinus, adjacentMinus, faceFraction
    );
    const GpuReal limitedContact = boundedTowardAdjacent
    (
        rawContact, adjacentContact, faceFraction
    );
    const GpuReal limitedPlus = boundedTowardAdjacent
    (
        rawPlus, adjacentPlus, faceFraction
    );

    const GpuReal rawTangentialX = raw.ux - rawNormal*nx;
    const GpuReal rawTangentialY = raw.uy - rawNormal*ny;
    const GpuReal rawTangentialZ = raw.uz - rawNormal*nz;
    const GpuReal adjacentTangentialX =
        adjacentDifference.ux - adjacentNormal*nx;
    const GpuReal adjacentTangentialY =
        adjacentDifference.uy - adjacentNormal*ny;
    const GpuReal adjacentTangentialZ =
        adjacentDifference.uz - adjacentNormal*nz;
    const GpuReal limitedTangentialX = boundedTowardAdjacent
    (
        rawTangentialX, adjacentTangentialX, faceFraction
    );
    const GpuReal limitedTangentialY = boundedTowardAdjacent
    (
        rawTangentialY, adjacentTangentialY, faceFraction
    );
    const GpuReal limitedTangentialZ = boundedTowardAdjacent
    (
        rawTangentialZ, adjacentTangentialZ, faceFraction
    );

    const GpuReal limitedPressure = GPU_R(0.5)*(limitedPlus + limitedMinus);
    const GpuReal limitedNormal =
        (limitedPlus - limitedMinus)/(GPU_R(2.0)*impedance);
    return Increment
    {
        limitedContact + limitedPressure/soundSquared,
        limitedTangentialX + limitedNormal*nx,
        limitedTangentialY + limitedNormal*ny,
        limitedTangentialZ + limitedNormal*nz,
        limitedPressure
    };
}

UGKP_CHARACTERISTIC_HD void limitFacePair
(
    Increment& leftIncrement,
    Increment& rightIncrement,
    const Increment& centreDifference,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal roeDensity,
    const GpuReal roeSoundSpeed,
    const GpuReal ownerWeight
)
{
    leftIncrement = limitOneSide
    (
        leftIncrement,
        centreDifference,
        nx,
        ny,
        nz,
        roeDensity,
        roeSoundSpeed,
        GPU_R(1.0) - clamp01(ownerWeight)
    );
    const Increment reverseDifference
    {
        -centreDifference.rho,
        -centreDifference.ux,
        -centreDifference.uy,
        -centreDifference.uz,
        -centreDifference.p
    };
    rightIncrement = limitOneSide
    (
        rightIncrement,
        reverseDifference,
        nx,
        ny,
        nz,
        roeDensity,
        roeSoundSpeed,
        clamp01(ownerWeight)
    );
}

}                                

#undef UGKP_CHARACTERISTIC_HD
