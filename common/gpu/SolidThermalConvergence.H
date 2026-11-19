#pragma once
#include "ScaledThermalSum.H"
namespace ugkp { namespace gpu {
template<class Real>
struct SolidThermalConvergenceFields {
    ScaledThermalSum enthalpyChange,externalEnergy,interfaceEnergy;
    Real temperatureChange=0,surfaceChange=0,surfaceFluxChange=0;
    bool boundaryChanged=false;
};
}}
