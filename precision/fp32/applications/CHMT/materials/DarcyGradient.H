#ifndef CHMT_MATERIALS_DARCYGRADIENT_H
#define CHMT_MATERIALS_DARCYGRADIENT_H
#include "materials/MaterialPhysicalStencil.H"
namespace chmt {



template<bool Terminal=false>
CHMT_HD inline bool darcyNormalPressureGradient(PhysicalVec3 displacement, PhysicalVec3 normal,
    PhysicalReal pressureDifference, PhysicalVec3 gradientLeft, PhysicalVec3 gradientRight, PhysicalReal& output) {
    const PhysicalReal distance=dot(displacement,normal);
    if (!Terminal&&(!finite(displacement)||!finite(normal)||!finite(pressureDifference)
        ||!finite(gradientLeft)||!finite(gradientRight)||!finite(distance)||distance<=0)) return false;
    const PhysicalReal value=pressureDifference/distance
        +dot(normal-displacement/distance,(gradientLeft+gradientRight)*0.5f);
    if (!Terminal&&!finite(value)) return false;
    output=value; return true;
}



CHMT_HD inline PhysicalReal darcyGradientSensitivity(const materialPhysicalStencil::Symmetric3& matrix, PhysicalVec3 displacement,int maximumIterations=32) {
    const PhysicalReal squared=dot(displacement,displacement);
    if (!(squared>0)||!finite(squared)) return undefinedValue();
    return 2*mag(materialPhysicalStencil::solveLeastSquares(matrix,displacement/squared,maximumIterations));
}






template<bool Terminal=false>
CHMT_HD inline bool darcyPressureStencilInverseLength(PhysicalVec3 displacement, PhysicalVec3 normal,
    PhysicalReal leftSensitivity, PhysicalReal rightSensitivity, PhysicalReal& output) {
    const PhysicalReal distance=dot(displacement,normal);
    if (!Terminal&&(!finite(displacement)||!finite(normal)||!finite(distance)||distance<=0
        ||!finite(leftSensitivity)||leftSensitivity<0
        ||!finite(rightSensitivity)||rightSensitivity<0)) return false;
    const PhysicalReal value=1/distance+0.25f*mag(normal-displacement/distance)
        *(leftSensitivity+rightSensitivity);
    if (!Terminal&&(!finite(value)||value<=0)) return false;
    output=value; return true;
}
}
#endif
