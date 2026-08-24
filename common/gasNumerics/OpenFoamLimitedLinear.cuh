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
    double x;
    double y;
    double z;
};

UGKP_INTERPOLATION_HD double dot(const Vector3& a, const Vector3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

UGKP_INTERPOLATION_HD double absolute(const double value)
{
    return value >= 0.0 ? value : -value;
}

UGKP_INTERPOLATION_HD double sign(const double value)
{
    return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

UGKP_INTERPOLATION_HD double clamp01(const double value)
{
    return value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
}

UGKP_INTERPOLATION_HD double nvdTvdRatio
(
    const double faceFlux,
    const double ownerValue,
    const double neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta
)
{
    const double faceDifference = neighbourValue - ownerValue;
    const double upwindProjectedGradient =
        dot
        (
            centreDelta,
            faceFlux > 0.0 ? ownerGradient : neighbourGradient
        );
    if
    (
        absolute(upwindProjectedGradient)
     >= 1000.0*absolute(faceDifference)
    )
    {
        return
            2000.0*sign(upwindProjectedGradient)*sign(faceDifference) - 1.0;
    }
    if (absolute(faceDifference) <= DBL_MIN)
    {
        return -1.0;
    }
    return 2.0*(upwindProjectedGradient/faceDifference) - 1.0;
}

UGKP_INTERPOLATION_HD double limitedLinearLimiter
(
    const double faceFlux,
    const double ownerValue,
    const double neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta,
    const double coefficient
)
{
    const double safeCoefficient =
        coefficient > DBL_MIN ? coefficient : DBL_MIN;
    const double ratio = nvdTvdRatio
    (
        faceFlux,
        ownerValue,
        neighbourValue,
        ownerGradient,
        neighbourGradient,
        centreDelta
    );
    return clamp01((2.0/safeCoefficient)*ratio);
}

UGKP_INTERPOLATION_HD double limitedLinearFaceValue
(
    const double ownerValue,
    const double neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const Vector3& centreDelta,
    const double ownerCentralWeight,
    const double faceFlux,
    const double coefficient
)
{
    const double limiter = limitedLinearLimiter
    (
        faceFlux,
        ownerValue,
        neighbourValue,
        ownerGradient,
        neighbourGradient,
        centreDelta,
        coefficient
    );
    const double upwindOwnerWeight = faceFlux >= 0.0 ? 1.0 : 0.0;
    const double ownerWeight =
        limiter*clamp01(ownerCentralWeight)
      + (1.0 - limiter)*upwindOwnerWeight;
    return
        ownerWeight*ownerValue
      + (1.0 - ownerWeight)*neighbourValue;
}

  
                                                                    
                                                        
  
                                                                            
                                                                          
                                                                         
                                                                        
                                                
   
UGKP_INTERPOLATION_HD double limitedLinearRiemannEnergyFlux
(
    const double baselineRiemannEnergyFlux,
    const double massFlux,
    const double upwindSpecificEnergy,
    const double limitedSpecificEnergy
)
{
    return
        baselineRiemannEnergyFlux
      + massFlux*(limitedSpecificEnergy - upwindSpecificEnergy);
}

}                               

#undef UGKP_INTERPOLATION_HD
