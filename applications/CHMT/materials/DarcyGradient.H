#ifndef CHMT_MATERIALS_DARCYGRADIENT_H
#define CHMT_MATERIALS_DARCYGRADIENT_H
#include "materials/StencilMath.H"
namespace chmt {



template<bool Terminal=false>
CHMT_HD inline bool darcyNormalPressureGradient(Vec3 displacement, Vec3 normal,
    Real pressureDifference, Vec3 gradientLeft, Vec3 gradientRight, Real& output) {
    const Real distance=dot(displacement,normal);
    if (!Terminal&&(!finite(displacement)||!finite(normal)||!finite(pressureDifference)
        ||!finite(gradientLeft)||!finite(gradientRight)||!finite(distance)||distance<=0)) return false;
    const Real value=pressureDifference/distance
        +dot(normal-displacement/distance,(gradientLeft+gradientRight)*0.5);
    if (!Terminal&&!finite(value)) return false;
    output=value; return true;
}



CHMT_HD inline Real darcyGradientSensitivity(const Symmetric3& matrix, Vec3 displacement,int maximumIterations=32) {
    const Real squared=dot(displacement,displacement);
    if (!(squared>0)||!finite(squared)) return undefinedValue();
    return 2*mag(solveLeastSquares(matrix,displacement/squared,maximumIterations));
}






template<bool Terminal=false>
CHMT_HD inline bool darcyPressureStencilInverseLength(Vec3 displacement, Vec3 normal,
    Real leftSensitivity, Real rightSensitivity, Real& output) {
    const Real distance=dot(displacement,normal);
    if (!Terminal&&(!finite(displacement)||!finite(normal)||!finite(distance)||distance<=0
        ||!finite(leftSensitivity)||leftSensitivity<0
        ||!finite(rightSensitivity)||rightSensitivity<0)) return false;
    const Real value=1/distance+0.25*mag(normal-displacement/distance)
        *(leftSensitivity+rightSensitivity);
    if (!Terminal&&(!finite(value)||value<=0)) return false;
    output=value; return true;
}
}
#endif
