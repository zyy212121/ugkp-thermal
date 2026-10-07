#ifndef CHMT_MATERIALS_DARCYGRADIENT_H
#define CHMT_MATERIALS_DARCYGRADIENT_H
#include "gas/AleFlux.H"
namespace chmt {
// Original-cell pressure differences supply the resolved normal derivative;
// the least-squares gradients supply only the non-orthogonal correction.
// A face reversal negates this derivative, preserving a single shared flux.
CHMT_HD inline bool darcyNormalPressureGradient(Vec3 displacement, Vec3 normal,
    Real pressureDifference, Vec3 gradientLeft, Vec3 gradientRight, Real& output) {
    const Real distance=dot(displacement,normal);
    if (!finite(displacement)||!finite(normal)||!finite(pressureDifference)
        ||!finite(gradientLeft)||!finite(gradientRight)||!finite(distance)||distance<=0) return false;
    const Real value=pressureDifference/distance
        +dot(normal-displacement/distance,(gradientLeft+gradientRight)*0.5);
    if (!finite(value)) return false;
    output=value; return true;
}

// For grad(p)=sum_j w_j*(p_j-p_c), 2*sum_j |w_j| bounds the
// vector-valued stencil's absolute pressure sensitivity, including p_c.
CHMT_HD inline Real darcyGradientSensitivity(const Symmetric3& matrix, Vec3 displacement) {
    const Real squared=dot(displacement,displacement);
    if (!(squared>0)||!finite(squared)) return std::numeric_limits<Real>::max();
    return 2*mag(solveLeastSquares(matrix,displacement/squared));
}

// Half an upper bound on the corrected face derivative's absolute row sum.
// The orthogonal value is 1/dn. The extra term accounts for BOTH cell
// gradient stencils; using |d| alone misses even the two-point stiffness.
// This is a frozen-density pressure-relaxation estimate, not a claim of
// monotonicity of an unlimited correction on an arbitrary distorted mesh.
CHMT_HD inline bool darcyPressureStencilInverseLength(Vec3 displacement, Vec3 normal,
    Real leftSensitivity, Real rightSensitivity, Real& output) {
    const Real distance=dot(displacement,normal);
    if (!finite(displacement)||!finite(normal)||!finite(distance)||distance<=0
        ||!finite(leftSensitivity)||leftSensitivity<0
        ||!finite(rightSensitivity)||rightSensitivity<0) return false;
    const Real value=1/distance+0.25*mag(normal-displacement/distance)
        *(leftSensitivity+rightSensitivity);
    if (!finite(value)||value<=0) return false;
    output=value; return true;
}
} // namespace chmt
#endif
