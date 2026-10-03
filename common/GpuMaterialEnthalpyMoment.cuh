#pragma once
template<class Real>
__device__ __forceinline__ Real materialEnthalpyMoment(const Real mass, const Real temperature)
{
    return mass*particleSpecificEnthalpyDevice(temperature);
}
