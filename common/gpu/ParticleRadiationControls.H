#pragma once
#include "RadiationPrecision.H"
namespace ugkp { namespace gpu {
// Typed numerical controls copied once into the material owner. Defaults retain
// the previous precision-specific stopping and settlement semantics.
struct ParticleRadiationSolveControls {
    int boundedMaximumIterations=100;
    RadiationReal boundedTemperatureRelativeTolerance=radiationConvergenceTolerance(RadiationReal(1e-10),8);
    RadiationReal boundedEnergyRelativeTolerance=radiationConvergenceTolerance(RadiationReal(1e-9),32);
    int caloricMaximumIterations=256;
    RadiationReal caloricEnergyRelativeTolerance=RadiationReal(64)*radiationEpsilon;
    RadiationReal boundaryEnergyRelativeTolerance=RadiationReal(1e-10);
};
}}
