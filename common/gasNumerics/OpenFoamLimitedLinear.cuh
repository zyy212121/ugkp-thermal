#include "GpuPrecisionTypes.H"
#pragma once

  
                                                                         
  
                                            
                                            
                                                   
                                                         
  
                                                                          
                                                                       
   

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define UGKP_INTERPOLATION_HD __host__ __device__ __forceinline__
#else
#define UGKP_INTERPOLATION_HD inline
#endif

namespace ugkpinterpolation
{

struct Vector3
{
    GpuReal x;
    GpuReal y;
    GpuReal z;
};

UGKP_INTERPOLATION_HD GpuReal dot(const Vector3& a, const Vector3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

UGKP_INTERPOLATION_HD GpuReal absolute(const GpuReal value)
{
    return value >= GPU_R(0.0) ? value : -value;
}

UGKP_INTERPOLATION_HD GpuReal sign(const GpuReal value)
{
    return value > GPU_R(0.0) ? GPU_R(1.0) : (value < GPU_R(0.0) ? -GPU_R(1.0) : GPU_R(0.0));
}

UGKP_INTERPOLATION_HD GpuReal clamp01(const GpuReal value)
{
    return value < GPU_R(0.0) ? GPU_R(0.0) : (value > GPU_R(1.0) ? GPU_R(1.0) : value);
}

UGKP_INTERPOLATION_HD GpuReal nvdTvdRatio
(
    const GpuReal faceFlux,
    const GpuReal ownerValue,
    const GpuReal neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta
)
{
    const GpuReal faceDifference = neighbourValue - ownerValue;
    const GpuReal upwindProjectedGradient =
        dot
        (
            centreDelta,
            faceFlux > GPU_R(0.0) ? ownerGradient : neighbourGradient
        );
    if
    (
        absolute(upwindProjectedGradient)
     >= GPU_R(1000.0)*absolute(faceDifference)
    )
    {
        return
            GPU_R(2000.0)*sign(upwindProjectedGradient)*sign(faceDifference) - GPU_R(1.0);
    }
    if (absolute(faceDifference) <= GPU_REAL_MIN)
    {
        return -GPU_R(1.0);
    }
    return GPU_R(2.0)*(upwindProjectedGradient/faceDifference) - GPU_R(1.0);
}

UGKP_INTERPOLATION_HD GpuReal limitedLinearLimiter
(
    const GpuReal faceFlux,
    const GpuReal ownerValue,
    const GpuReal neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta,
    const GpuReal coefficient
)
{
    const GpuReal safeCoefficient =
        coefficient > GPU_REAL_MIN ? coefficient : GPU_REAL_MIN;
    const GpuReal ratio = nvdTvdRatio
    (
        faceFlux,
        ownerValue,
        neighbourValue,
        ownerGradient,
        neighbourGradient,
        centreDelta
    );
    return clamp01((GPU_R(2.0)/safeCoefficient)*ratio);
}

UGKP_INTERPOLATION_HD GpuReal limitedLinearFaceValue
(
    const GpuReal ownerValue,
    const GpuReal neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta,
    const GpuReal ownerCentralWeight,
    const GpuReal faceFlux,
    const GpuReal coefficient
)
{
    const GpuReal limiter = limitedLinearLimiter
    (
        faceFlux,
        ownerValue,
        neighbourValue,
        ownerGradient,
        neighbourGradient,
        centreDelta,
        coefficient
    );
    const GpuReal upwindOwnerWeight = faceFlux >= GPU_R(0.0) ? GPU_R(1.0) : GPU_R(0.0);
    const GpuReal ownerWeight =
        limiter*clamp01(ownerCentralWeight)
      + (GPU_R(1.0) - limiter)*upwindOwnerWeight;
    return
        ownerWeight*ownerValue
      + (GPU_R(1.0) - ownerWeight)*neighbourValue;
}

  
                                                                    
                                                        
  
                                                                            
                                                                          
                                                                         
                                                                        
                                                
   
UGKP_INTERPOLATION_HD GpuReal limitedLinearRiemannEnergyFlux
(
    const GpuReal baselineRiemannEnergyFlux,
    const GpuReal massFlux,
    const GpuReal upwindSpecificEnergy,
    const GpuReal limitedSpecificEnergy
)
{
    return
        baselineRiemannEnergyFlux
      + massFlux*(limitedSpecificEnergy - upwindSpecificEnergy);
}

}                               

#undef UGKP_INTERPOLATION_HD
