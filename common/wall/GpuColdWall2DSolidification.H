#include "GpuPrecisionTypes.H"
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
    GpuReal maximumRadiusM;
    GpuReal filmThicknessM;
    GpuReal axialNodeThicknessM;
    GpuReal ringAreaM2;
    GpuReal nodeMassKg;
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
    const GpuReal particleVolumeM3,
    const GpuReal physicalParticleMassKg,
    const GpuReal maximumAreaM2
) noexcept
{
    const GpuReal filmThickness = particleVolumeM3/maximumAreaM2;
    const GpuReal maximumRadius = ::sqrt(maximumAreaM2/coldWallPi);
    const GpuReal axialNodeThickness =
        filmThickness/static_cast<GpuReal>(coldWall2DAxialNodeCount);
    const GpuReal ringArea =
        maximumAreaM2/static_cast<GpuReal>(coldWall2DRadialNodeCount);
    const GpuReal nodeMass =
        physicalParticleMassKg/static_cast<GpuReal>(coldWall2DNodeCount);
    return
    {
        true,
        maximumRadius,
        filmThickness,
        axialNodeThickness,
        ringArea,
        nodeMass
    };
}

GPU_COLD_WALL_2D_HD inline GpuReal coldWall2DRingBoundaryRadiusM
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
                static_cast<GpuReal>(boundary)
               /static_cast<GpuReal>(coldWall2DRadialNodeCount),
                GPU_R(0.0),
                GPU_R(1.0)
            )
        );
}

GPU_COLD_WALL_2D_HD inline GpuReal coldWall2DRingCentroidRadiusM
(
    const ColdWall2DGeometry& geometry,
    const int ring
) noexcept
{
    const GpuReal inner = coldWall2DRingBoundaryRadiusM(geometry, ring);
    const GpuReal outer = coldWall2DRingBoundaryRadiusM(geometry, ring + 1);
    const GpuReal denominator = outer*outer - inner*inner;
    return (GPU_R(2.0)/GPU_R(3.0))
       *(outer*outer*outer - inner*inner*inner)/(denominator + GPU_REAL_MIN);
}

GPU_COLD_WALL_2D_HD inline GpuReal coldWall2DRingWetAreaM2
(
    const ColdWall2DGeometry& geometry,
    const GpuReal intrinsicContactAreaM2,
    const int ring
) noexcept
{
    return coldWallClamp
    (
        intrinsicContactAreaM2
       -static_cast<GpuReal>(ring)*geometry.ringAreaM2,
        GPU_R(0.0),
        geometry.ringAreaM2
    );
}

GPU_COLD_WALL_2D_HD inline GpuReal coldWall2DRadialFaceConductanceWK
(
    const ColdWall2DGeometry& geometry,
    const int innerRing,
    const GpuReal innerConductivityWmK,
    const GpuReal outerConductivityWmK
) noexcept
{
    if
    (
        !geometry.valid
     || innerRing < 0
     || innerRing + 1 >= coldWall2DRadialNodeCount
    )
    {
        return GPU_R(0.0);
    }
    const GpuReal faceRadius = coldWall2DRingBoundaryRadiusM
    (
        geometry,
        innerRing + 1
    );
    const GpuReal innerCenter = coldWall2DRingCentroidRadiusM
    (
        geometry,
        innerRing
    );
    const GpuReal outerCenter = coldWall2DRingCentroidRadiusM
    (
        geometry,
        innerRing + 1
    );
    const GpuReal harmonicConductivity =
        GPU_R(2.0)*innerConductivityWmK*outerConductivityWmK
       /(innerConductivityWmK + outerConductivityWmK + GPU_REAL_MIN);
    return harmonicConductivity
       *(GPU_R(2.0)*coldWallPi*faceRadius*geometry.axialNodeThicknessM)
       /(outerCenter - innerCenter + GPU_REAL_MIN);
}

GPU_COLD_WALL_2D_HD inline GpuReal coldWall2DConnectedSolidThicknessFraction
(
    const GpuReal* ringSpecificEnthalpyJkg,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    if (ringSpecificEnthalpyJkg == nullptr)
    {
        return GPU_R(0.0);
    }
    GpuReal connectedFraction = GPU_R(1.0);
    GpuReal connectedThickness = GPU_R(0.0);
    for (int node = 0; node < coldWall2DAxialNodeCount; ++node)
    {
        const GpuReal solidFraction = coldWallSolidFraction
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
        connectedThickness/static_cast<GpuReal>(coldWall2DAxialNodeCount),
        GPU_R(0.0),
        GPU_R(1.0)
    );
}

GPU_COLD_WALL_2D_HD inline void initialiseColdWall2DProfile
(
    GpuReal* nodeSpecificEnthalpyJkg,
    GpuTime* ringContactAgeS,
    const GpuReal initialTemperatureK,
    const ColdWallSolidificationParameters& parameters
) noexcept
{
    const GpuReal initialEnthalpy = coldWallSpecificEnthalpyJkg
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
        ringContactAgeS[ring] = GPU_R(0.0);
    }
}

}
}

#undef GPU_COLD_WALL_2D_HD

#endif
