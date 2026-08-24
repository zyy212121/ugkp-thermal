#ifndef GPU_THERMAL_GPU_COLD_WALL_2D_SOLIDIFICATION_H
#define GPU_THERMAL_GPU_COLD_WALL_2D_SOLIDIFICATION_H

#include "GpuColdWallSolidification.H"

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define GPU_COLD_WALL_2D_HD __host__ __device__
#else
#define GPU_COLD_WALL_2D_HD
#endif

namespace Foam
{
namespace gpuThermal
{

constexpr int coldWall2DRadialNodeCount = 8;
constexpr int coldWall2DAxialNodeCount = 8;
constexpr int coldWall2DNodeCount =
    coldWall2DRadialNodeCount*coldWall2DAxialNodeCount;

struct ColdWall2DGeometry
{
    bool valid;
    double maximumRadiusM;
    double filmThicknessM;
    double axialNodeThicknessM;
    double ringAreaM2;
    double nodeMassKg;
};

GPU_COLD_WALL_2D_HD inline int coldWall2DNodeIndex
(
    const int ring,
    const int axialNode
) noexcept
{
    return ring*coldWall2DAxialNodeCount + axialNode;
}

GPU_COLD_WALL_2D_HD inline ColdWall2DGeometry coldWall2DGeometry
(
    const double particleVolumeM3,
    const double physicalParticleMassKg,
    const double maximumAreaM2
) noexcept
{
    const bool valid =
        finiteColdWallValue(particleVolumeM3)
     && finiteColdWallValue(physicalParticleMassKg)
     && finiteColdWallValue(maximumAreaM2)
     && particleVolumeM3 > 0.0
     && physicalParticleMassKg > 0.0
     && maximumAreaM2 > 0.0;
    if (!valid)
    {
        return {false, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    const double filmThickness = particleVolumeM3/maximumAreaM2;
    return
    {
        true,
        ::sqrt(maximumAreaM2/coldWallPi),
        filmThickness,
        filmThickness/static_cast<double>(coldWall2DAxialNodeCount),
        maximumAreaM2/static_cast<double>(coldWall2DRadialNodeCount),
        physicalParticleMassKg/static_cast<double>(coldWall2DNodeCount)
    };
}

GPU_COLD_WALL_2D_HD inline double coldWall2DRingBoundaryRadiusM
(
    const ColdWall2DGeometry& geometry,
    const int boundary
) noexcept
{
    return geometry.maximumRadiusM
       *::sqrt
        (
            coldWallClamp
            (
                static_cast<double>(boundary)
               /static_cast<double>(coldWall2DRadialNodeCount),
                0.0,
                1.0
            )
        );
}

GPU_COLD_WALL_2D_HD inline double coldWall2DRingCentroidRadiusM
(
    const ColdWall2DGeometry& geometry,
    const int ring
) noexcept
{
    const double inner = coldWall2DRingBoundaryRadiusM(geometry, ring);
    const double outer = coldWall2DRingBoundaryRadiusM(geometry, ring + 1);
    const double denominator = outer*outer - inner*inner;
    return (2.0/3.0)
       *(outer*outer*outer - inner*inner*inner)/(denominator + DBL_MIN);
}

GPU_COLD_WALL_2D_HD inline double coldWall2DRingWetAreaM2
(
    const ColdWall2DGeometry& geometry,
    const double intrinsicContactAreaM2,
    const int ring
) noexcept
{
    return coldWallClamp
    (
        intrinsicContactAreaM2
       -static_cast<double>(ring)*geometry.ringAreaM2,
        0.0,
        geometry.ringAreaM2
    );
}

GPU_COLD_WALL_2D_HD inline double coldWall2DRadialFaceConductanceWK
(
    const ColdWall2DGeometry& geometry,
    const int innerRing,
    const double innerConductivityWmK,
    const double outerConductivityWmK
) noexcept
{
    if
    (
        !geometry.valid
     || innerRing < 0
     || innerRing + 1 >= coldWall2DRadialNodeCount
     || !(innerConductivityWmK > 0.0)
     || !(outerConductivityWmK > 0.0)
    )
    {
        return 0.0;
    }
    const double faceRadius = coldWall2DRingBoundaryRadiusM
    (
        geometry,
        innerRing + 1
    );
    const double innerCenter = coldWall2DRingCentroidRadiusM
    (
        geometry,
        innerRing
    );
    const double outerCenter = coldWall2DRingCentroidRadiusM
    (
        geometry,
        innerRing + 1
    );
    const double harmonicConductivity =
        2.0*innerConductivityWmK*outerConductivityWmK
       /(innerConductivityWmK + outerConductivityWmK + DBL_MIN);
    return harmonicConductivity
       *(2.0*coldWallPi*faceRadius*geometry.axialNodeThicknessM)
       /(outerCenter - innerCenter + DBL_MIN);
}

GPU_COLD_WALL_2D_HD inline double coldWall2DConnectedSolidThicknessFraction
(
    const double* ringSpecificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (ringSpecificEnthalpyJkg == nullptr)
    {
        return 0.0;
    }
    double connectedFraction = 1.0;
    double connectedThickness = 0.0;
    for (int node = 0; node < coldWall2DAxialNodeCount; ++node)
    {
        const double solidFraction = coldWallSolidFraction
        (
            ringSpecificEnthalpyJkg[node],
            parameters
        );
        connectedFraction = solidFraction < connectedFraction
          ? solidFraction : connectedFraction;
        connectedThickness += connectedFraction;
    }
    return coldWallClamp
    (
        connectedThickness/static_cast<double>(coldWall2DAxialNodeCount),
        0.0,
        1.0
    );
}

GPU_COLD_WALL_2D_HD inline void initialiseColdWall2DProfile
(
    double* nodeSpecificEnthalpyJkg,
    double* ringContactAgeS,
    const double initialTemperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const double initialEnthalpy = coldWallSpecificEnthalpyJkg
    (
        initialTemperatureK,
        parameters
    );
    for (int node = 0; node < coldWall2DNodeCount; ++node)
    {
        nodeSpecificEnthalpyJkg[node] = initialEnthalpy;
    }
    for (int ring = 0; ring < coldWall2DRadialNodeCount; ++ring)
    {
        ringContactAgeS[ring] = 0.0;
    }
}

}
}

#undef GPU_COLD_WALL_2D_HD

#endif
