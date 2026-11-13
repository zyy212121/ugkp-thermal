#ifndef CHMT_MATERIALS_PHYSICAL_STENCIL_H
#define CHMT_MATERIALS_PHYSICAL_STENCIL_H
#include "materials/EnergyStencilMath.H"
namespace chmt { namespace materialPhysicalStencil {
using Symmetric3=materialEnergyStencil::Symmetric3;
CHMT_HD inline void addLeastSquares(Symmetric3& m,PhysicalVec3& rhs,PhysicalVec3 dx,PhysicalReal difference){
    materialEnergyStencil::Vec3 b{rhs.x,rhs.y,rhs.z};
    materialEnergyStencil::addLeastSquares(m,b,{dx.x,dx.y,dx.z},difference);
    rhs={b.x,b.y,b.z};
}
CHMT_HD inline PhysicalVec3 solveLeastSquares(const Symmetric3& m,PhysicalVec3 rhs,int maximumIterations=32){
    const auto value=materialEnergyStencil::solveLeastSquares(m,{rhs.x,rhs.y,rhs.z},maximumIterations);
    return {value.x,value.y,value.z};
}
}}
#endif
