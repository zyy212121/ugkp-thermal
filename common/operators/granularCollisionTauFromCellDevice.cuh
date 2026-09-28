#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL granularCollisionTauFromCellDevice(const DeviceState& s, const int c)
{
    if (c < 0 || c >= s.nCells)
    {
        return OfGreat;
    }

    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return OfGreat;
    }

    const GPU_OPERATOR_REAL eps =
        clampRange(rhoP/clampMin(s.rhoSolid, 1.0e-300), 0.0, 1.0);

    const GPU_OPERATOR_REAL usx = finiteOr(s.momRhoUPx[c], 0.0)/rhoP;
    const GPU_OPERATOR_REAL usy = finiteOr(s.momRhoUPy[c], 0.0)/rhoP;
    const GPU_OPERATOR_REAL usz = finiteOr(s.momRhoUPz[c], 0.0)/rhoP;

    const GPU_OPERATOR_REAL e =
        clampMin(finiteOr(s.momRhoEP[c], 0.0), 0.0)/rhoP;

    const GPU_OPERATOR_REAL theta =
        clampMin((e - 0.5*sqr3(usx, usy, usz))/1.5, 0.0);

    const GPU_OPERATOR_REAL dPart =
        clampMin
        (
            finiteOr(s.momRhoPD[c]/rhoP, s.particleDiameterFallback),
            1.0e-12
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
