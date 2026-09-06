#include "GpuPrecisionTypes.H"
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
    GpuReal rho;
    GpuReal ux;
    GpuReal uy;
    GpuReal uz;
    GpuReal p;
    GpuReal T;
};

UGKP_BOUNDARY_HD GpuReal maximum(const GpuReal a, const GpuReal b)
{
    return a > b ? a : b;
}

UGKP_BOUNDARY_HD GpuReal minimum(const GpuReal a, const GpuReal b)
{
    return a < b ? a : b;
}

UGKP_BOUNDARY_HD GpuReal finiteOr(const GpuReal value, const GpuReal fallback)
{
    return ::isfinite(value) ? value : fallback;
}

UGKP_BOUNDARY_HD GpuReal subsonicInletMach
(
    const GpuReal outgoingInvariant,
    const GpuReal reservoirSoundSpeed,
    const GpuReal gamma
)
{
    const GpuReal gammaMinusOne = maximum(gamma - GPU_R(1.0), GPU_R(1.0e-12));
    const GpuReal coefficient = GPU_R(0.5)*gammaMinusOne;
    const GpuReal invariantRatio =
        outgoingInvariant/maximum(reservoirSoundSpeed, GPU_TINY(1.0e-300));
    const GpuReal zeroMachRatio = GPU_R(2.0)/gammaMinusOne;
    const GpuReal sonicRatio =
        (zeroMachRatio - GPU_R(1.0))/::sqrt(GPU_R(1.0) + coefficient);
    if (invariantRatio >= zeroMachRatio)
    {
        return GPU_R(0.0);
    }
    if (invariantRatio <= sonicRatio)
    {
        return GPU_R(1.0);
    }

    GpuReal lower = GPU_R(0.0);
    GpuReal upper = GPU_R(1.0);
    for (int iteration = 0; iteration < 48; ++iteration)
    {
        const GpuReal middle = GPU_R(0.5)*(lower + upper);
        const GpuReal ratio =
            (zeroMachRatio - middle)
           /::sqrt(GPU_R(1.0) + coefficient*middle*middle);
        if (ratio > invariantRatio)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return GPU_R(0.5)*(lower + upper);
}

UGKP_BOUNDARY_HD Primitive totalConditionInletState
(
    const Primitive& owner,
    const GpuReal nx,
    const GpuReal ny,
    const GpuReal nz,
    const GpuReal totalPressure,
    const GpuReal totalTemperature,
    const GpuReal gamma,
    const GpuReal gasConstant,
    const GpuReal densityFloor,
    const GpuReal temperatureFloor
)
{
    const GpuReal safeGamma = maximum(finiteOr(gamma, GPU_R(1.4)), GPU_R(1.0) + GPU_R(1.0e-12));
    const GpuReal gammaMinusOne = safeGamma - GPU_R(1.0);
    const GpuReal safeR = maximum(finiteOr(gasConstant, GPU_R(1.0)), GPU_TINY(1.0e-300));
    const GpuReal safeRhoFloor =
        maximum(finiteOr(densityFloor, GPU_TINY(1.0e-300)), GPU_TINY(1.0e-300));
    const GpuReal safeTemperatureFloor =
        maximum(finiteOr(temperatureFloor, GPU_R(1.0)), GPU_TINY(1.0e-300));
    const GpuReal safeTotalTemperature = maximum
    (
        finiteOr(totalTemperature, safeTemperatureFloor),
        safeTemperatureFloor
    );
    const GpuReal safeTotalPressure = maximum
    (
        finiteOr(totalPressure, safeRhoFloor*safeR*safeTotalTemperature),
        safeRhoFloor*safeR*safeTotalTemperature
    );

    const GpuReal ownerRho = maximum(finiteOr(owner.rho, safeRhoFloor), safeRhoFloor);
    const GpuReal ownerP = maximum
    (
        finiteOr(owner.p, ownerRho*safeR*safeTemperatureFloor),
        ownerRho*safeR*safeTemperatureFloor
    );
    const GpuReal ownerSoundSpeed =
        ::sqrt(maximum(safeGamma*ownerP/ownerRho, GPU_TINY(1.0e-300)));
    const GpuReal ownerNormalVelocity =
        finiteOr(owner.ux, GPU_R(0.0))*nx
      + finiteOr(owner.uy, GPU_R(0.0))*ny
      + finiteOr(owner.uz, GPU_R(0.0))*nz;
    const GpuReal outgoingInvariant =
        ownerNormalVelocity + GPU_R(2.0)*ownerSoundSpeed/gammaMinusOne;
    const GpuReal reservoirSoundSpeed =
        ::sqrt(safeGamma*safeR*safeTotalTemperature);
    const GpuReal mach = subsonicInletMach
    (
        outgoingInvariant,
        reservoirSoundSpeed,
        safeGamma
    );
    const GpuReal totalFactor =
        GPU_R(1.0) + GPU_R(0.5)*gammaMinusOne*mach*mach;

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
    const GpuReal speed =
        mach*::sqrt(safeGamma*safeR*boundary.T);
    boundary.ux = -speed*nx;
    boundary.uy = -speed*ny;
    boundary.uz = -speed*nz;
    return boundary;
}

}                          

#undef UGKP_BOUNDARY_HD
