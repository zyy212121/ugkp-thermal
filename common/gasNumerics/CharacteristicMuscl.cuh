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
    double rho;
    double ux;
    double uy;
    double uz;
    double p;
};

UGKP_CHARACTERISTIC_HD double clamp01(const double value)
{
    return value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
}

UGKP_CHARACTERISTIC_HD double boundedTowardAdjacent
(
    const double raw,
    const double adjacentDifference,
    const double faceFraction
)
{
    if
    (
        !::isfinite(raw)
     || !::isfinite(adjacentDifference)
     || raw*adjacentDifference <= 0.0
    )
    {
        return 0.0;
    }
    const double bound =
        clamp01(faceFraction)*::fabs(adjacentDifference);
    return ::copysign(::fmin(::fabs(raw), bound), raw);
}

UGKP_CHARACTERISTIC_HD Increment limitOneSide
(
    const Increment& raw,
    const Increment& adjacentDifference,
    const double nx,
    const double ny,
    const double nz,
    const double roeDensity,
    const double roeSoundSpeed,
    const double faceFraction
)
{
    const double soundSquared = roeSoundSpeed*roeSoundSpeed;
    const double impedance = roeDensity*roeSoundSpeed;
    if
    (
        !::isfinite(soundSquared)
     || !::isfinite(impedance)
     || soundSquared <= 0.0
     || impedance <= 0.0
    )
    {
        return Increment{0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double rawNormal = raw.ux*nx + raw.uy*ny + raw.uz*nz;
    const double adjacentNormal =
        adjacentDifference.ux*nx
      + adjacentDifference.uy*ny
      + adjacentDifference.uz*nz;
    const double rawMinus = raw.p - impedance*rawNormal;
    const double rawContact = raw.rho - raw.p/soundSquared;
    const double rawPlus = raw.p + impedance*rawNormal;
    const double adjacentMinus =
        adjacentDifference.p - impedance*adjacentNormal;
    const double adjacentContact =
        adjacentDifference.rho
      - adjacentDifference.p/soundSquared;
    const double adjacentPlus =
        adjacentDifference.p + impedance*adjacentNormal;

    const double limitedMinus = boundedTowardAdjacent
    (
        rawMinus, adjacentMinus, faceFraction
    );
    const double limitedContact = boundedTowardAdjacent
    (
        rawContact, adjacentContact, faceFraction
    );
    const double limitedPlus = boundedTowardAdjacent
    (
        rawPlus, adjacentPlus, faceFraction
    );

    const double rawTangentialX = raw.ux - rawNormal*nx;
    const double rawTangentialY = raw.uy - rawNormal*ny;
    const double rawTangentialZ = raw.uz - rawNormal*nz;
    const double adjacentTangentialX =
        adjacentDifference.ux - adjacentNormal*nx;
    const double adjacentTangentialY =
        adjacentDifference.uy - adjacentNormal*ny;
    const double adjacentTangentialZ =
        adjacentDifference.uz - adjacentNormal*nz;
    const double limitedTangentialX = boundedTowardAdjacent
    (
        rawTangentialX, adjacentTangentialX, faceFraction
    );
    const double limitedTangentialY = boundedTowardAdjacent
    (
        rawTangentialY, adjacentTangentialY, faceFraction
    );
    const double limitedTangentialZ = boundedTowardAdjacent
    (
        rawTangentialZ, adjacentTangentialZ, faceFraction
    );

    const double limitedPressure = 0.5*(limitedPlus + limitedMinus);
    const double limitedNormal =
        (limitedPlus - limitedMinus)/(2.0*impedance);
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
    const double nx,
    const double ny,
    const double nz,
    const double roeDensity,
    const double roeSoundSpeed,
    const double ownerWeight
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
        1.0 - clamp01(ownerWeight)
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
