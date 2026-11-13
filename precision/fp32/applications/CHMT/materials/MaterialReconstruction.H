#ifndef CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#define CHMT_MATERIALS_MATERIALRECONSTRUCTION_H
#include "materials/EnergyStencilMath.H"
namespace chmt {
// Inventories divided by unequal cell volumes can give a few ulps of variation
// in an exactly constant component. Such unresolved noise must not switch off
// the shared limiter for every other (resolved) material component. There is
// no absolute floor: physically small inventories still receive normal limiting.
// This changes only the limiter decision, never the state or its inventories.
CHMT_HD inline PhysicalReal materialReconstructionLimiter(PhysicalReal value, PhysicalReal lower,
    PhysicalReal upper, PhysicalReal increment) {
    const PhysicalReal scale=maxValue(absValue(value),maxValue(absValue(lower),absValue(upper)));
    const PhysicalReal roundoff=64*std::numeric_limits<PhysicalReal>::epsilon()*scale;
    if (upper-lower<=roundoff && absValue(increment)<=roundoff) return 1;
    return materialEnergyStencil::barthJespersen(value,lower,upper,increment);
}
} // namespace chmt
#endif
