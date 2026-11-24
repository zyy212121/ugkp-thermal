#pragma once

#include "../../../common/gasNumerics/OpenFoamWallFunctions.cuh"

#if defined(__CUDACC__)
#define UGKP_LEGACY_WALL_HD __host__ __device__ inline
#else
#define UGKP_LEGACY_WALL_HD inline
#endif

namespace ugkpwall
{
struct JayatillekeWallHeatState
{
    GpuReal heatFlux;
    GpuReal temperaturePlus;
    GpuReal yPlusThermal;
    int valid;
};

UGKP_LEGACY_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFluxPrecomputed
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal molecularPrandtl,
    const GpuReal turbulentPrandtl,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal P,
    const GpuReal thermalYPlus,
    const GpuReal uTau,
    const GpuReal yPlus,
    const GpuReal cellTemperature,
    const GpuReal wallTemperature
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal Pr = maximum(molecularPrandtl, small);
    const GpuReal Prt = maximum(turbulentPrandtl, small);
    const GpuReal yPlusSafe = maximum(yPlus, GPU_R(0.0));
    const bool valid =
        density > GPU_R(0.0)
     && heatCapacity > GPU_R(0.0)
     && uTau > small
     && yPlusSafe > small
     && molecularPrandtl > GPU_R(0.0)
     && turbulentPrandtl > GPU_R(0.0)
     && kappa > GPU_R(0.0) && E > GPU_R(0.0)
     && thermalYPlus > GPU_R(0.0)
     && std::isfinite(density) && std::isfinite(heatCapacity)
     && std::isfinite(uTau) && std::isfinite(yPlus)
     && std::isfinite(thermalYPlus) && std::isfinite(P);
    if (!valid)
    {
        return JayatillekeWallHeatState{std::numeric_limits<GpuReal>::quiet_NaN(), std::numeric_limits<GpuReal>::quiet_NaN(), thermalYPlus, 0};
    }

    const GpuReal temperaturePlus = yPlusSafe < thermalYPlus
      ? Pr*yPlusSafe
      : Prt*
        (
            log(maximum(E*yPlusSafe, GPU_R(1.0) + small))/maximum(kappa, small)
          + P
        );
    if (!(temperaturePlus > GPU_R(0.0)) || !std::isfinite(temperaturePlus))
    {
        return JayatillekeWallHeatState{std::numeric_limits<GpuReal>::quiet_NaN(), temperaturePlus, thermalYPlus, 0};
    }
    const GpuReal heatFlux =
        density*heatCapacity*uTau*(cellTemperature - wallTemperature)/temperaturePlus;
    return JayatillekeWallHeatState
    {
        heatFlux,
        temperaturePlus,
        thermalYPlus,
        std::isfinite(heatFlux) ? 1 : 0
    };
}

UGKP_LEGACY_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFlux
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal molecularPrandtl,
    const GpuReal turbulentPrandtl,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal uTau,
    const GpuReal yPlus,
    const GpuReal cellTemperature,
    const GpuReal wallTemperature,
    const int thermalMaxIterations = 80,
    const GpuReal thermalRelativeTolerance = GPU_R(8)*GPU_REAL_EPSILON
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal Pr = maximum(molecularPrandtl, small);
    const GpuReal Prt = maximum(turbulentPrandtl, small);
    const GpuReal PrRatio = Pr/Prt;
    return jayatillekeWallHeatFluxPrecomputed
    (
        density,
        heatCapacity,
        molecularPrandtl,
        turbulentPrandtl,
        kappa,
        E,
        jayatillekeSmoothP(PrRatio),
        jayatillekeThermalYPlus(PrRatio, kappa, E, nullptr, -1, thermalMaxIterations, thermalRelativeTolerance),
        uTau,
        yPlus,
        cellTemperature,
        wallTemperature
    );
}

UGKP_LEGACY_WALL_HD JayatillekeThermalTransport sstJayatillekeThermalTransport
(
    const GpuReal molecularConductivity,
    const GpuReal Pr,
    const GpuReal Prt,
    const GpuReal kappa,
    const GpuReal E,
    const GpuReal P,
    const GpuReal yPlusThermal,
    const GpuReal yPlus,
    const GpuReal temperatureNormalGradient
)
{
    if (!std::isfinite(yPlus) || !std::isfinite(yPlusThermal))
        return {molecularConductivity, 0};

    GpuReal conductivity = molecularConductivity;
    if (yPlus > yPlusThermal && yPlus > GPU_R(0))
    {
        const GpuReal tPlus = Prt*(log(E*yPlus)/kappa + P);
        if (!std::isfinite(tPlus) || tPlus == GPU_R(0))
            return {conductivity, 0};
        conductivity *= maximum(GPU_R(1), Pr*yPlus/tPlus);
    }

    const GpuReal heatFlux = -conductivity*temperatureNormalGradient;
    return {conductivity, std::isfinite(conductivity) && std::isfinite(heatFlux), heatFlux};
}
}

#undef UGKP_LEGACY_WALL_HD
