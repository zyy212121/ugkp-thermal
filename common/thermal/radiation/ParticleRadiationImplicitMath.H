#ifndef GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H
#define GPU_THERMAL_PARTICLE_RADIATION_IMPLICIT_MATH_H
#include "label.H"
#include "scalar.H"
#include "../../../common/gpu/RadiationImplicitMath.H"
#include <stdexcept>
namespace Foam { namespace gpuThermal {
class ParticleRadiationSolveError : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};
namespace detail {
using ugkp::gpu::radiationMath::ImplicitTemperatureRootResult;
using ugkp::gpu::radiationMath::ConservativeBoundaryEnergyScaleResult;
using ugkp::gpu::radiationMath::finiteParticleRadiationScalar;
using ugkp::gpu::radiationMath::conservativeBoundaryEnergyScale;
using ugkp::gpu::radiationMath::oppositeSigns;
using ugkp::gpu::radiationMath::solveImplicitMaterialTemperature;
using ugkp::gpu::radiationMath::solveRadiativeEquilibriumTemperature;
using ugkp::gpu::radiationMath::solveBoundedImplicitMaterialTemperature;
[[noreturn]] inline void implicitRootFailure(const char* reason){throw ParticleRadiationSolveError(reason);}
}
}}
#endif
