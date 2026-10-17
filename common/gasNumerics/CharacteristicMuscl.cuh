#include "GpuPrecisionTypes.H"
#pragma once

#include <cmath>
#include <limits>

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

// A failure is arithmetic evidence for the production caller, never a zero slope.
struct Failure
{
    bool failed = false;
    bool zeroDenominator = false;
    int component = -1;
    GpuReal value = GPU_R(0.0);
};
UGKP_CHARACTERISTIC_HD bool checkFinite(GpuReal value, int component, Failure* failure)
{
    if (::isfinite(value)) return true;
    if (failure && !failure->failed) *failure = Failure{true, false, component, value};
    return false;
}
UGKP_CHARACTERISTIC_HD Increment failedIncrement()
{
    const GpuReal value = std::numeric_limits<GpuReal>::quiet_NaN();
    return Increment{value,value,value,value,value};
}

UGKP_CHARACTERISTIC_HD GpuReal clamp01(const GpuReal value)
{
    return value < GPU_R(0.0) ? GPU_R(0.0) : (value > GPU_R(1.0) ? GPU_R(1.0) : value);
}

UGKP_CHARACTERISTIC_HD GpuReal boundedTowardAdjacent
(
    const GpuReal raw,
    const GpuReal adjacentDifference,
    const GpuReal faceFraction,
    Failure* failure = nullptr
)
{
    if (!checkFinite(raw, 9, failure) || !checkFinite(adjacentDifference, 10, failure)
        || !checkFinite(faceFraction, 7, failure))
        return std::numeric_limits<GpuReal>::quiet_NaN();
    if (raw*adjacentDifference <= GPU_R(0.0))
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
    const GpuReal faceFraction,
    Failure* failure = nullptr
)
{
    const GpuReal inputs[] = {raw.rho,raw.ux,raw.uy,raw.uz,raw.p,
        adjacentDifference.rho,adjacentDifference.ux,adjacentDifference.uy,
        adjacentDifference.uz,adjacentDifference.p,nx,ny,nz,faceFraction};
    for (int i=0;i<14;++i)
        if (!checkFinite(inputs[i], i, failure)) return failedIncrement();
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
        const GpuReal bad = (!::isfinite(soundSquared) || soundSquared<=GPU_R(0.0)) ? soundSquared : impedance;
        if (failure && !failure->failed)
            *failure = Failure{true,bad==GPU_R(0.0),14,bad};
        return failedIncrement();
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
        rawMinus, adjacentMinus, faceFraction, failure
    );
    const GpuReal limitedContact = boundedTowardAdjacent
    (
        rawContact, adjacentContact, faceFraction, failure
    );
    const GpuReal limitedPlus = boundedTowardAdjacent
    (
        rawPlus, adjacentPlus, faceFraction, failure
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
        rawTangentialX, adjacentTangentialX, faceFraction, failure
    );
    const GpuReal limitedTangentialY = boundedTowardAdjacent
    (
        rawTangentialY, adjacentTangentialY, faceFraction, failure
    );
    const GpuReal limitedTangentialZ = boundedTowardAdjacent
    (
        rawTangentialZ, adjacentTangentialZ, faceFraction, failure
    );

    const GpuReal limitedPressure = GPU_R(0.5)*(limitedPlus + limitedMinus);
    const GpuReal normalDenominator=GPU_R(2.0)*impedance;
    if(!checkFinite(normalDenominator,15,failure)) return failedIncrement();
    const GpuReal limitedNormal =
        (limitedPlus - limitedMinus)/normalDenominator;
    const Increment result
    {
        limitedContact + limitedPressure/soundSquared,
        limitedTangentialX + limitedNormal*nx,
        limitedTangentialY + limitedNormal*ny,
        limitedTangentialZ + limitedNormal*nz,
        limitedPressure
    };
    const GpuReal values[] = {result.rho,result.ux,result.uy,result.uz,result.p};
    for(int i=0;i<5;++i) if(!checkFinite(values[i],i,failure)) return failedIncrement();
    return result;
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
    const GpuReal ownerWeight,
    Failure* failure = nullptr
)
{
    if(!checkFinite(ownerWeight,13,failure))
    {leftIncrement=failedIncrement();rightIncrement=failedIncrement();return;}
    leftIncrement = limitOneSide
    (
        leftIncrement,
        centreDifference,
        nx,
        ny,
        nz,
        roeDensity,
        roeSoundSpeed,
        GPU_R(1.0) - clamp01(ownerWeight), failure
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
        clamp01(ownerWeight), failure
    );
}

}                                

#undef UGKP_CHARACTERISTIC_HD
