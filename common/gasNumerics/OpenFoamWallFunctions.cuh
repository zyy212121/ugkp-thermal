#include "GpuPrecisionTypes.H"
#pragma once

  
                                                              
  
                                         
                                                                      
                                                                             
                                                                        
                                                                         
                                                             
  
                                           
  
                                                                   
                                                                             
                                                                           
                                                            
   

#include <cmath>

#if defined(__CUDACC__)
#define UGKP_WALL_HD __host__ __device__ __forceinline__
#else
#define UGKP_WALL_HD inline
#endif

namespace ugkpwall
{

struct SpaldingWallState
{
    GpuReal uTau;
    GpuReal yPlus;
    GpuReal nut;
};

struct WallSubgridTransport
{
    GpuReal dynamicViscosity;
    GpuReal thermalConductivity;
};

struct OmegaWallFunctionState
{
    GpuReal omega;
    GpuReal production;
    GpuReal yPlus;
};

struct JayatillekeWallHeatState
{
    GpuReal heatFlux;
    GpuReal temperaturePlus;
    GpuReal yPlusThermal;
    int valid;
};

UGKP_WALL_HD GpuReal maximum(const GpuReal a, const GpuReal b)
{
    return a > b ? a : b;
}

UGKP_WALL_HD GpuReal minimum(const GpuReal a, const GpuReal b)
{
    return a < b ? a : b;
}

UGKP_WALL_HD GpuReal wallYPlusLaminar
(
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8)
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    GpuReal yPlus = GPU_R(11.0);
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const GpuReal argument = maximum(E*yPlus, GPU_R(1.0) + small);
        const GpuReal function = yPlus - log(argument)/kappa;
        const GpuReal derivative = GPU_R(1.0) - GPU_R(1.0)/(kappa*yPlus);
        const GpuReal updated = yPlus - function/maximum(derivative, small);
        if (updated <= small)
        {
            break;
        }
        if (fabs(updated - yPlus) <= GPU_R(1.0e-8)*maximum(yPlus, GPU_R(1.0)))
        {
            yPlus = updated;
            break;
        }
        yPlus = updated;
    }
    return maximum(yPlus, GPU_R(0.0));
}

UGKP_WALL_HD SpaldingWallState spaldingWallState
(
    const GpuReal velocityDifference,
    const GpuReal wallDistance,
    const GpuReal kinematicViscosity,
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8)
)
{
    constexpr GpuReal rootVSmall = GPU_TINY(1.4916681462400413e-154);
    const GpuReal up = maximum(velocityDifference, GPU_R(0.0));
    const GpuReal y = maximum(wallDistance, rootVSmall);
    const GpuReal nu = maximum(kinematicViscosity, rootVSmall);
    const GpuReal magGradU = up/y;
    GpuReal uTau = sqrt(nu*magGradU);

    if (uTau > rootVSmall)
    {
        int iteration = 0;
        GpuReal error = GPU_LARGE(1.0e300);
        do
        {
            const GpuReal kUu = minimum(kappa*up/uTau, GPU_R(50.0));
            const GpuReal fkUu =
                exp(kUu) - GPU_R(1.0) - kUu*(GPU_R(1.0) + GPU_R(0.5)*kUu);
            const GpuReal f =
                -uTau*y/nu
              + up/uTau
              + (fkUu - (GPU_R(1.0)/GPU_R(6.0))*kUu*kUu*kUu)/E;
            const GpuReal df =
                y/nu
              + up/(uTau*uTau)
              + kUu*fkUu/(E*uTau);
            const GpuReal uTauNew = uTau + f/maximum(df, rootVSmall);
            error = fabs((uTau - uTauNew)/uTau);
            uTau = uTauNew;
        }
        while
        (
            uTau > rootVSmall
         && error > GPU_R(0.01)
         && ++iteration < 10
        );
    }

    uTau = maximum(uTau, GPU_R(0.0));
    const GpuReal nut = maximum
    (
        uTau*uTau/(magGradU + rootVSmall) - nu,
        GPU_R(0.0)
    );
    return SpaldingWallState{uTau, y*uTau/nu, nut};
}

UGKP_WALL_HD WallSubgridTransport wallSubgridTransport
(
    const GpuReal density,
    const GpuReal heatCapacity,
    const GpuReal turbulentPrandtl,
    const GpuReal wallNut
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal muT =
        maximum(density, GPU_R(0.0))*maximum(wallNut, GPU_R(0.0));
    return WallSubgridTransport
    {
        muT,
        maximum(heatCapacity, GPU_R(0.0))*muT
       /maximum(turbulentPrandtl, small)
    };
}

UGKP_WALL_HD OmegaWallFunctionState omegaWallFunctionState
(
    const GpuReal k,
    const GpuReal velocityNormalGradient,
    const GpuReal wallDistance,
    const GpuReal kinematicViscosity,
    const GpuReal beta1 = GPU_R(0.075),
    const GpuReal Cmu = GPU_R(0.09),
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8),
    const GpuReal cellProduction = GPU_R(0.0)
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal kSafe = maximum(k, GPU_R(0.0));
    const GpuReal y = maximum(wallDistance, small);
    const GpuReal nu = maximum(kinematicViscosity, small);
    const GpuReal CmuSafe = maximum(Cmu, small);
    const GpuReal Cmu25 = sqrt(sqrt(CmuSafe));
    const GpuReal Cmu5 = sqrt(CmuSafe);
    const GpuReal reynoldsY = y*sqrt(kSafe)/nu;
    const GpuReal yPlus = Cmu25*reynoldsY;
    const GpuReal omegaViscous = GPU_R(6.0)*nu/(maximum(beta1, small)*y*y);

    if (yPlus < wallYPlusLaminar(kappa, E))
    {
        return OmegaWallFunctionState
        {
            omegaViscous,
            cellProduction,
            yPlus
        };
    }

    const GpuReal uPlus = log(maximum(E*yPlus, GPU_R(1.0) + small))
      /maximum(kappa, small);
    const GpuReal uStar = Cmu25*sqrt(kSafe);
    const GpuReal omegaLog =
        uStar/(Cmu5*maximum(kappa, small)*y);
    const GpuReal scaledShear =
        uStar*maximum(velocityNormalGradient, GPU_R(0.0))*y
       /maximum(uPlus, small);
    const GpuReal productionLog =
        scaledShear*scaledShear
       /(nu*maximum(kappa, small)*maximum(yPlus, small));
    return OmegaWallFunctionState
    {
        maximum(omegaLog, GPU_R(0.0)),
        maximum(productionLog, GPU_R(0.0)),
        yPlus
    };
}

UGKP_WALL_HD GpuReal jayatillekeSmoothP(const GpuReal Prat)
{
    const GpuReal ratio = maximum(Prat, GPU_R(2.2204460492503131e-16));
    return
        GPU_R(9.24)*(pow(ratio, GPU_R(0.75)) - GPU_R(1.0))
       *(GPU_R(1.0) + GPU_R(0.28)*exp(-GPU_R(0.007)*ratio));
}

UGKP_WALL_HD GpuReal jayatillekeThermalYPlus
(
    const GpuReal Prat,
    const GpuReal kappa = GPU_R(0.41),
    const GpuReal E = GPU_R(9.8)
)
{
    constexpr GpuReal small = GPU_R(2.2204460492503131e-16);
    const GpuReal ratio = maximum(Prat, small);
    const GpuReal P = jayatillekeSmoothP(ratio);
    GpuReal yPlus = GPU_R(11.0);
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const GpuReal argument = maximum(E*yPlus, GPU_R(1.0) + small);
        const GpuReal function =
            yPlus - (log(argument)/maximum(kappa, small) + P)/ratio;
        const GpuReal derivative =
            GPU_R(1.0) - GPU_R(1.0)/(yPlus*maximum(kappa, small)*ratio);
        const GpuReal updated = yPlus - function/maximum(derivative, small);
        if (updated <= small)
        {
            return GPU_R(0.0);
        }
        if (fabs(updated - yPlus) < GPU_R(0.01))
        {
            return updated;
        }
        yPlus = updated;
    }
    return maximum(yPlus, GPU_R(0.0));
}

UGKP_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFluxPrecomputed
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
     && yPlusSafe > small;
    if (!valid)
    {
        return JayatillekeWallHeatState{GPU_R(0.0), GPU_R(0.0), thermalYPlus, 0};
    }

    const GpuReal temperaturePlus = yPlusSafe < thermalYPlus
      ? Pr*yPlusSafe
      : Prt*
        (
            log(maximum(E*yPlusSafe, GPU_R(1.0) + small))/maximum(kappa, small)
          + P
        );
    const GpuReal heatFlux =
        density*heatCapacity*uTau*(cellTemperature - wallTemperature)
       /maximum(temperaturePlus, small);
    return JayatillekeWallHeatState
    {
        heatFlux,
        temperaturePlus,
        thermalYPlus,
        1
    };
}

UGKP_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFlux
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
    const GpuReal wallTemperature
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
        jayatillekeThermalYPlus(PrRatio, kappa, E),
        uTau,
        yPlus,
        cellTemperature,
        wallTemperature
    );
}

}                      

#undef UGKP_WALL_HD
