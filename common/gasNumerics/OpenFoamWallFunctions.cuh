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
    double uTau;
    double yPlus;
    double nut;
};

struct WallSubgridTransport
{
    double dynamicViscosity;
    double thermalConductivity;
};

struct OmegaWallFunctionState
{
    double omega;
    double production;
    double yPlus;
};

struct JayatillekeWallHeatState
{
    double heatFlux;
    double temperaturePlus;
    double yPlusThermal;
    int valid;
};

UGKP_WALL_HD double maximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKP_WALL_HD double minimum(const double a, const double b)
{
    return a < b ? a : b;
}

UGKP_WALL_HD double wallYPlusLaminar
(
    const double kappa = 0.41,
    const double E = 9.8
)
{
    constexpr double small = 2.2204460492503131e-16;
    double yPlus = 11.0;
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const double argument = maximum(E*yPlus, 1.0 + small);
        const double function = yPlus - log(argument)/kappa;
        const double derivative = 1.0 - 1.0/(kappa*yPlus);
        const double updated = yPlus - function/maximum(derivative, small);
        if (updated <= small)
        {
            break;
        }
        if (fabs(updated - yPlus) <= 1.0e-8*maximum(yPlus, 1.0))
        {
            yPlus = updated;
            break;
        }
        yPlus = updated;
    }
    return maximum(yPlus, 0.0);
}

UGKP_WALL_HD SpaldingWallState spaldingWallState
(
    const double velocityDifference,
    const double wallDistance,
    const double kinematicViscosity,
    const double kappa = 0.41,
    const double E = 9.8
)
{
    constexpr double rootVSmall = 1.4916681462400413e-154;
    const double up = maximum(velocityDifference, 0.0);
    const double y = maximum(wallDistance, rootVSmall);
    const double nu = maximum(kinematicViscosity, rootVSmall);
    const double magGradU = up/y;
    double uTau = sqrt(nu*magGradU);

    if (uTau > rootVSmall)
    {
        int iteration = 0;
        double error = 1.0e300;
        do
        {
            const double kUu = minimum(kappa*up/uTau, 50.0);
            const double fkUu =
                exp(kUu) - 1.0 - kUu*(1.0 + 0.5*kUu);
            const double f =
                -uTau*y/nu
              + up/uTau
              + (fkUu - (1.0/6.0)*kUu*kUu*kUu)/E;
            const double df =
                y/nu
              + up/(uTau*uTau)
              + kUu*fkUu/(E*uTau);
            const double uTauNew = uTau + f/maximum(df, rootVSmall);
            error = fabs((uTau - uTauNew)/uTau);
            uTau = uTauNew;
        }
        while
        (
            uTau > rootVSmall
         && error > 0.01
         && ++iteration < 10
        );
    }

    uTau = maximum(uTau, 0.0);
    const double nut = maximum
    (
        uTau*uTau/(magGradU + rootVSmall) - nu,
        0.0
    );
    return SpaldingWallState{uTau, y*uTau/nu, nut};
}

UGKP_WALL_HD WallSubgridTransport wallSubgridTransport
(
    const double density,
    const double heatCapacity,
    const double turbulentPrandtl,
    const double wallNut
)
{
    constexpr double small = 2.2204460492503131e-16;
    const double muT =
        maximum(density, 0.0)*maximum(wallNut, 0.0);
    return WallSubgridTransport
    {
        muT,
        maximum(heatCapacity, 0.0)*muT
       /maximum(turbulentPrandtl, small)
    };
}

UGKP_WALL_HD OmegaWallFunctionState omegaWallFunctionState
(
    const double k,
    const double velocityNormalGradient,
    const double wallDistance,
    const double kinematicViscosity,
    const double beta1 = 0.075,
    const double Cmu = 0.09,
    const double kappa = 0.41,
    const double E = 9.8,
    const double cellProduction = 0.0
)
{
    constexpr double small = 2.2204460492503131e-16;
    const double kSafe = maximum(k, 0.0);
    const double y = maximum(wallDistance, small);
    const double nu = maximum(kinematicViscosity, small);
    const double CmuSafe = maximum(Cmu, small);
    const double Cmu25 = sqrt(sqrt(CmuSafe));
    const double Cmu5 = sqrt(CmuSafe);
    const double reynoldsY = y*sqrt(kSafe)/nu;
    const double yPlus = Cmu25*reynoldsY;
    const double omegaViscous = 6.0*nu/(maximum(beta1, small)*y*y);

    if (yPlus < wallYPlusLaminar(kappa, E))
    {
        return OmegaWallFunctionState
        {
            omegaViscous,
            cellProduction,
            yPlus
        };
    }

    const double uPlus = log(maximum(E*yPlus, 1.0 + small))
      /maximum(kappa, small);
    const double uStar = Cmu25*sqrt(kSafe);
    const double omegaLog =
        uStar/(Cmu5*maximum(kappa, small)*y);
    const double scaledShear =
        uStar*maximum(velocityNormalGradient, 0.0)*y
       /maximum(uPlus, small);
    const double productionLog =
        scaledShear*scaledShear
       /(nu*maximum(kappa, small)*maximum(yPlus, small));
    return OmegaWallFunctionState
    {
        maximum(omegaLog, 0.0),
        maximum(productionLog, 0.0),
        yPlus
    };
}

UGKP_WALL_HD double jayatillekeSmoothP(const double Prat)
{
    const double ratio = maximum(Prat, 2.2204460492503131e-16);
    return
        9.24*(pow(ratio, 0.75) - 1.0)
       *(1.0 + 0.28*exp(-0.007*ratio));
}

UGKP_WALL_HD double jayatillekeThermalYPlus
(
    const double Prat,
    const double kappa = 0.41,
    const double E = 9.8
)
{
    constexpr double small = 2.2204460492503131e-16;
    const double ratio = maximum(Prat, small);
    const double P = jayatillekeSmoothP(ratio);
    double yPlus = 11.0;
    for (int iteration = 0; iteration < 10; ++iteration)
    {
        const double argument = maximum(E*yPlus, 1.0 + small);
        const double function =
            yPlus - (log(argument)/maximum(kappa, small) + P)/ratio;
        const double derivative =
            1.0 - 1.0/(yPlus*maximum(kappa, small)*ratio);
        const double updated = yPlus - function/maximum(derivative, small);
        if (updated <= small)
        {
            return 0.0;
        }
        if (fabs(updated - yPlus) < 0.01)
        {
            return updated;
        }
        yPlus = updated;
    }
    return maximum(yPlus, 0.0);
}

UGKP_WALL_HD JayatillekeWallHeatState jayatillekeWallHeatFluxPrecomputed
(
    const double density,
    const double heatCapacity,
    const double molecularPrandtl,
    const double turbulentPrandtl,
    const double kappa,
    const double E,
    const double P,
    const double thermalYPlus,
    const double uTau,
    const double yPlus,
    const double cellTemperature,
    const double wallTemperature
)
{
    constexpr double small = 2.2204460492503131e-16;
    const double Pr = maximum(molecularPrandtl, small);
    const double Prt = maximum(turbulentPrandtl, small);
    const double yPlusSafe = maximum(yPlus, 0.0);
    const bool valid =
        density > 0.0
     && heatCapacity > 0.0
     && uTau > small
     && yPlusSafe > small;
    if (!valid)
    {
        return JayatillekeWallHeatState{0.0, 0.0, thermalYPlus, 0};
    }

    const double temperaturePlus = yPlusSafe < thermalYPlus
      ? Pr*yPlusSafe
      : Prt*
        (
            log(maximum(E*yPlusSafe, 1.0 + small))/maximum(kappa, small)
          + P
        );
    const double heatFlux =
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
    const double density,
    const double heatCapacity,
    const double molecularPrandtl,
    const double turbulentPrandtl,
    const double kappa,
    const double E,
    const double uTau,
    const double yPlus,
    const double cellTemperature,
    const double wallTemperature
)
{
    constexpr double small = 2.2204460492503131e-16;
    const double Pr = maximum(molecularPrandtl, small);
    const double Prt = maximum(turbulentPrandtl, small);
    const double PrRatio = Pr/Prt;
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
