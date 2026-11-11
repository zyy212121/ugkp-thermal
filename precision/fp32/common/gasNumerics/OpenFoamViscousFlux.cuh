#include "GpuPrecisionTypes.H"
#pragma once

  
                                                                   
                                               
  
                                                      
  
                                    
                                                
  
        
  
                                             
                                               
                                                      
  
                                                                     
  
                                                 
  
                                                                        
                                                                         
                                                                        
                                                                    
   

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define UGKP_TRANSPORT_HD __host__ __device__ __forceinline__
#else
#define UGKP_TRANSPORT_HD inline
#endif

namespace ugkptransport
{

template<class Real> struct Vector3T
{
    Real x;
    Real y;
    Real z;
    UGKP_TRANSPORT_HD Vector3T() :x(0),y(0),z(0) {}
    template<class X,class Y,class Z> UGKP_TRANSPORT_HD Vector3T(X a,Y b,Z c)
      :x(Real(a)),y(Real(b)),z(Real(c)) {}
};

template<class Real> struct SnGradGeometryT
{
    Real delta;
    Vector3T<Real> correction;
};

template<class Real>
UGKP_TRANSPORT_HD Real dot(const Vector3T<Real>& a, const Vector3T<Real>& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

template<class Real>
UGKP_TRANSPORT_HD Vector3T<Real> add(const Vector3T<Real>& a, const Vector3T<Real>& b)
{
    return Vector3T<Real>{a.x + b.x, a.y + b.y, a.z + b.z};
}

template<class Real>
UGKP_TRANSPORT_HD Vector3T<Real> subtract(const Vector3T<Real>& a, const Vector3T<Real>& b)
{
    return Vector3T<Real>{a.x - b.x, a.y - b.y, a.z - b.z};
}

template<class Real>
UGKP_TRANSPORT_HD Vector3T<Real> scale(const Vector3T<Real>& value, const Real factor)
{
    return Vector3T<Real>
    {
        factor*value.x,
        factor*value.y,
        factor*value.z
    };
}

template<class Real>
UGKP_TRANSPORT_HD Real magnitude(const Vector3T<Real>& value)
{
    return sqrt(dot(value, value));
}

template<class Real>
UGKP_TRANSPORT_HD Real maximum(const Real a, const Real b)
{
    return a > b ? a : b;
}

template<class Real>
UGKP_TRANSPORT_HD Vector3T<Real> linearInterpolate
(
    const Vector3T<Real>& owner,
    const Vector3T<Real>& neighbour,
    const Real ownerWeight
)
{
    return add
    (
        scale(owner, ownerWeight),
        scale(neighbour, Real(1.0) - ownerWeight)
    );
}

template<class Real,class NormalReal>
UGKP_TRANSPORT_HD SnGradGeometryT<Real> makeInternalSnGradGeometry
(
    const Vector3T<Real>& ownerCentre,
    const Vector3T<Real>& neighbourCentre,
    const Vector3T<NormalReal>& inputNormal
)
{
    const Vector3T<Real> unitNormal{inputNormal.x,inputNormal.y,inputNormal.z};
    const Vector3T<Real> centreDelta = subtract(neighbourCentre, ownerCentre);
    const Real centreDistance = magnitude(centreDelta);
    const Real normalDistance = dot(unitNormal, centreDelta);
    const Real denominator = maximum
    (
        normalDistance,
        Real(0.05)*centreDistance
    );
    const Real delta =
        denominator > std::numeric_limits<Real>::min()
      ? Real(1.0)/denominator
      : Real(0.0);
    return SnGradGeometryT<Real>
    {
        delta,
        subtract(unitNormal, scale(centreDelta, delta))
    };
}

template<class Real,class Geometry,class Value>
UGKP_TRANSPORT_HD Real correctedSnGrad
(
    const Value ownerValue,
    const Value neighbourValue,
    const Vector3T<Real>& ownerGradient,
    const Vector3T<Real>& neighbourGradient,
    const Real ownerWeight,
    const SnGradGeometryT<Geometry>& geometry
)
{
    const Vector3T<Real> interpolatedGradient = linearInterpolate
    (
        ownerGradient,
        neighbourGradient,
        ownerWeight
    );
    return
        Real(geometry.delta)*(Real(neighbourValue) - Real(ownerValue))
      + dot(Vector3T<Real>{geometry.correction.x,geometry.correction.y,geometry.correction.z}, interpolatedGradient);
}

template<class Real>
UGKP_TRANSPORT_HD Vector3T<Real> openFoamNewtonianTraction
(
    const Real dynamicViscosity,
    const Vector3T<Real>& unitNormal,
    const Vector3T<Real>& compactSnGradU,
    const Vector3T<Real>& gradUx,
    const Vector3T<Real>& gradUy,
    const Vector3T<Real>& gradUz
)
{
    const Real divU = gradUx.x + gradUy.y + gradUz.z;
    const Vector3T<Real> transposedGradientNormal
    (
        Vector3T<Real>
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
            scale(unitNormal, -(Real(2.0)/Real(3.0))*divU)
        ),
        dynamicViscosity
    );
}

using Vector3 = Vector3T<GpuReal>;
using SnGradGeometry = SnGradGeometryT<GpuGeometry>;

}                           

#undef UGKP_TRANSPORT_HD
