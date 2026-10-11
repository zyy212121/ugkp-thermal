#ifndef CHMT_GPU_COMMONGASTRACTION_H
#define CHMT_GPU_COMMONGASTRACTION_H
#include "core/Types.H"
#include "../../../common/gasNumerics/OpenFoamViscousFlux.cuh"
namespace chmt {
static_assert(sizeof(GpuReal)==sizeof(Real),"CHMT coupling uses the binary64 gas precision contract");
// Coupling supplies its normal gradient and maps vector storage only. The
// Newtonian stress formula is owned by the shared gas transport operator.
CHMT_HD inline Vec3 viscousTraction(const GasGradient& gradient,Vec3 area,
    Real viscosity,Real turbulentPressure) {
    using V=ugkptransport::Vector3;
    const V a{area.x,area.y,area.z};
    const V gx{gradient.velocity[0].x,gradient.velocity[0].y,gradient.velocity[0].z};
    const V gy{gradient.velocity[1].x,gradient.velocity[1].y,gradient.velocity[1].z};
    const V gz{gradient.velocity[2].x,gradient.velocity[2].y,gradient.velocity[2].z};
    const V normalGradient{dot(gradient.velocity[0],area),dot(gradient.velocity[1],area),dot(gradient.velocity[2],area)};
    const auto traction=ugkptransport::openFoamNewtonianTraction(viscosity,a,normalGradient,gx,gy,gz);
    return Vec3{traction.x,traction.y,traction.z}-area*turbulentPressure;
}
}
#endif
