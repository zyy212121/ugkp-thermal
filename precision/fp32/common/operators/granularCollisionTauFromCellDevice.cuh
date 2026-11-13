#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL granularCollisionTauFromCellDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return OfGreat;
    }

    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return OfGreat;
    }

    const GPU_OPERATOR_REAL eps =
        clampRange(rhoP/clampMin(s.rhoSolid, GPU_OPERATOR_TINY(1.0e-300)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));

    const GPU_OPERATOR_REAL theta = collisionCentralTheta(s.collisionCellMoments[c]);

    const GPU_OPERATOR_REAL dPart =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/rhoP, s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );

    const GPU_OPERATOR_REAL g0 = radialDistributionG0Device(eps);
    const GPU_OPERATOR_REAL lmfp = ugkwp::granularMeanFreePath
    (
        OfPi,
        dPart,
        eps,
        g0,
        OfSmall
    );

    return clampMin
    (
        ugkwp::granularCollisionTime(lmfp, theta, OfSmall),
        OfSmall
    );
}
