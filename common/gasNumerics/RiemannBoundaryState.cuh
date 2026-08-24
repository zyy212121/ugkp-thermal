#pragma once

  
                                                                      
  
                                                                        
                                                                              
                                                                        
                                                                             
                                                                          
   

#include <cmath>

#if defined(__CUDACC__)
#define UGKP_BOUNDARY_HD __host__ __device__ __forceinline__
#else
#define UGKP_BOUNDARY_HD inline
#endif

namespace ugkpboundary
{

struct Primitive
{
    double rho;
    double ux;
    double uy;
    double uz;
    double p;
    double T;
};

UGKP_BOUNDARY_HD double maximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKP_BOUNDARY_HD double minimum(const double a, const double b)
{
    return a < b ? a : b;
}

UGKP_BOUNDARY_HD double finiteOr(const double value, const double fallback)
{
    return ::isfinite(value) ? value : fallback;
}

UGKP_BOUNDARY_HD double subsonicInletMach
(
    const double outgoingInvariant,
    const double reservoirSoundSpeed,
    const double gamma
)
{
    const double gammaMinusOne = maximum(gamma - 1.0, 1.0e-12);
    const double coefficient = 0.5*gammaMinusOne;
    const double invariantRatio =
        outgoingInvariant/maximum(reservoirSoundSpeed, 1.0e-300);
    const double zeroMachRatio = 2.0/gammaMinusOne;
    const double sonicRatio =
        (zeroMachRatio - 1.0)/::sqrt(1.0 + coefficient);
    if (invariantRatio >= zeroMachRatio)
    {
        return 0.0;
    }
    if (invariantRatio <= sonicRatio)
    {
        return 1.0;
    }

    double lower = 0.0;
    double upper = 1.0;
    for (int iteration = 0; iteration < 48; ++iteration)
    {
        const double middle = 0.5*(lower + upper);
        const double ratio =
            (zeroMachRatio - middle)
           /::sqrt(1.0 + coefficient*middle*middle);
        if (ratio > invariantRatio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return 0.5*(lower + upper);
}

UGKP_BOUNDARY_HD Primitive totalConditionInletState
(
    const Primitive& owner,
    const double nx,
    const double ny,
    const double nz,
    const double totalPressure,
    const double totalTemperature,
    const double gamma,
    const double gasConstant,
    const double densityFloor,
    const double temperatureFloor
)
{
    const double safeGamma = maximum(finiteOr(gamma, 1.4), 1.0 + 1.0e-12);
    const double gammaMinusOne = safeGamma - 1.0;
    const double safeR = maximum(finiteOr(gasConstant, 1.0), 1.0e-300);
    const double safeRhoFloor =
        maximum(finiteOr(densityFloor, 1.0e-300), 1.0e-300);
    const double safeTemperatureFloor =
        maximum(finiteOr(temperatureFloor, 1.0), 1.0e-300);
    const double safeTotalTemperature = maximum
    (
        finiteOr(totalTemperature, safeTemperatureFloor),
        safeTemperatureFloor
    );
    const double safeTotalPressure = maximum
    (
        finiteOr(totalPressure, safeRhoFloor*safeR*safeTotalTemperature),
        safeRhoFloor*safeR*safeTotalTemperature
    );

    const double ownerRho = maximum(finiteOr(owner.rho, safeRhoFloor), safeRhoFloor);
    const double ownerP = maximum
    (
        finiteOr(owner.p, ownerRho*safeR*safeTemperatureFloor),
        ownerRho*safeR*safeTemperatureFloor
    );
    const double ownerSoundSpeed =
        ::sqrt(maximum(safeGamma*ownerP/ownerRho, 1.0e-300));
    const double ownerNormalVelocity =
        finiteOr(owner.ux, 0.0)*nx
      + finiteOr(owner.uy, 0.0)*ny
      + finiteOr(owner.uz, 0.0)*nz;
    const double outgoingInvariant =
        ownerNormalVelocity + 2.0*ownerSoundSpeed/gammaMinusOne;
    const double reservoirSoundSpeed =
        ::sqrt(safeGamma*safeR*safeTotalTemperature);
    const double mach = subsonicInletMach
    (
        outgoingInvariant,
        reservoirSoundSpeed,
        safeGamma
    );
    const double totalFactor =
        1.0 + 0.5*gammaMinusOne*mach*mach;

    Primitive boundary;
    boundary.T = maximum
    (
        safeTotalTemperature/totalFactor,
        safeTemperatureFloor
    );
    boundary.p = maximum
    (
        safeTotalPressure
       /::pow(totalFactor, safeGamma/gammaMinusOne),
        safeRhoFloor*safeR*boundary.T
    );
    boundary.rho = maximum
    (
        boundary.p/(safeR*boundary.T),
        safeRhoFloor
    );
    boundary.p = boundary.rho*safeR*boundary.T;
    const double speed =
        mach*::sqrt(safeGamma*safeR*boundary.T);
    boundary.ux = -speed*nx;
    boundary.uy = -speed*ny;
    boundary.uz = -speed*nz;
    return boundary;
}

}                          

#undef UGKP_BOUNDARY_HD
