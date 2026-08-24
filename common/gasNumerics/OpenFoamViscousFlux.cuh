#pragma once

  
                                                                   
                                               
  
                                                      
  
                                    
                                                
  
        
  
                                             
                                               
                                                      
  
                                                                     
  
                                                 
  
                                                                        
                                                                         
                                                                        
                                                                    
   

#include <cmath>

#if defined(__CUDACC__)
#define UGKP_TRANSPORT_HD __host__ __device__ __forceinline__
#else
#define UGKP_TRANSPORT_HD inline
#endif

namespace ugkptransport
{

struct Vector3
{
    double x;
    double y;
    double z;
};

struct SnGradGeometry
{
    double delta;
    Vector3 correction;
};

UGKP_TRANSPORT_HD double dot(const Vector3& a, const Vector3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

UGKP_TRANSPORT_HD Vector3 add(const Vector3& a, const Vector3& b)
{
    return Vector3{a.x + b.x, a.y + b.y, a.z + b.z};
}

UGKP_TRANSPORT_HD Vector3 subtract(const Vector3& a, const Vector3& b)
{
    return Vector3{a.x - b.x, a.y - b.y, a.z - b.z};
}

UGKP_TRANSPORT_HD Vector3 scale(const Vector3& value, const double factor)
{
    return Vector3
    {
        factor*value.x,
        factor*value.y,
        factor*value.z
    };
}

UGKP_TRANSPORT_HD double magnitude(const Vector3& value)
{
    return sqrt(dot(value, value));
}

UGKP_TRANSPORT_HD double maximum(const double a, const double b)
{
    return a > b ? a : b;
}

UGKP_TRANSPORT_HD Vector3 linearInterpolate
(
    const Vector3& owner,
    const Vector3& neighbour,
    const double ownerWeight
)
{
    return add
    (
        scale(owner, ownerWeight),
        scale(neighbour, 1.0 - ownerWeight)
    );
}

UGKP_TRANSPORT_HD SnGradGeometry makeInternalSnGradGeometry
(
    const Vector3& ownerCentre,
    const Vector3& neighbourCentre,
    const Vector3& unitNormal
)
{
    const Vector3 centreDelta = subtract(neighbourCentre, ownerCentre);
    const double centreDistance = magnitude(centreDelta);
    const double normalDistance = dot(unitNormal, centreDelta);
    const double denominator = maximum
    (
        normalDistance,
        0.05*centreDistance
    );
    const double delta =
        denominator > 2.22507385850720138309e-308
      ? 1.0/denominator
      : 0.0;
    return SnGradGeometry
    {
        delta,
        subtract(unitNormal, scale(centreDelta, delta))
    };
}

UGKP_TRANSPORT_HD double correctedSnGrad
(
    const double ownerValue,
    const double neighbourValue,
    const Vector3& ownerGradient,
    const Vector3& neighbourGradient,
    const double ownerWeight,
    const SnGradGeometry& geometry
)
{
    const Vector3 interpolatedGradient = linearInterpolate
    (
        ownerGradient,
        neighbourGradient,
        ownerWeight
    );
    return
        geometry.delta*(neighbourValue - ownerValue)
      + dot(geometry.correction, interpolatedGradient);
}

UGKP_TRANSPORT_HD Vector3 openFoamNewtonianTraction
(
    const double dynamicViscosity,
    const Vector3& unitNormal,
    const Vector3& compactSnGradU,
    const Vector3& gradUx,
    const Vector3& gradUy,
    const Vector3& gradUz
)
{
    const double divU = gradUx.x + gradUy.y + gradUz.z;
    const Vector3 transposedGradientNormal
    (
        Vector3
        {
            unitNormal.x*gradUx.x
          + unitNormal.y*gradUy.x
          + unitNormal.z*gradUz.x,
            unitNormal.x*gradUx.y
          + unitNormal.y*gradUy.y
          + unitNormal.z*gradUz.y,
            unitNormal.x*gradUx.z
          + unitNormal.y*gradUy.z
          + unitNormal.z*gradUz.z
        }
    );
    return scale
    (
        add
        (
            add(compactSnGradU, transposedGradientNormal),
            scale(unitNormal, -(2.0/3.0)*divU)
        ),
        dynamicViscosity
    );
}

}                           

#undef UGKP_TRANSPORT_HD
