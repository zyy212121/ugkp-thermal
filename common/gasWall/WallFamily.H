#pragma once

namespace ugkwp::gaswall
{
enum class WallFamily : int
{
    LowRe = 0,
    WallFunction = 1,
    BoundaryLayer = 2,
    CompressibleWallFunction = 4
};

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr int canonicalWallTreatment(int treatment)
{
    return treatment == int(WallFamily::CompressibleWallFunction)
        ? int(WallFamily::WallFunction) : treatment;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
inline constexpr bool usesAlgebraicWallFunction(int treatment)
{
    return canonicalWallTreatment(treatment) == int(WallFamily::WallFunction);
}
}
