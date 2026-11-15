#ifndef CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#define CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#include "materials/EnergyStencilMath.H"
namespace chmt {





CHMT_HD inline PhysicalReal materialReconstructionLimiter(PhysicalReal value, PhysicalReal lower,
    PhysicalReal upper, PhysicalReal increment) {
    const PhysicalReal scale=maxValue(absValue(value),maxValue(absValue(lower),absValue(upper)));
    const PhysicalReal roundoff=64*std::numeric_limits<PhysicalReal>::epsilon()*scale;
    if (upper-lower<=roundoff && absValue(increment)<=roundoff) return 1;
    return materialEnergyStencil::barthJespersen(value,lower,upper,increment);
}
}
#endif
