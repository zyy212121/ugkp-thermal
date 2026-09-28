#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL solidPressureFromMomentsDevice
(
    const DeviceState& s,
    const int c
)
{
    if (c < 0 || c >= s.nCells)
    {
        return 0.0;
    }

    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], 0.0), 0.0);
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return 0.0;
    }

    const GPU_OPERATOR_REAL eps =
        clampRange(rhoP/clampMin(s.rhoSolid, 1.0e-300), 0.0, 1.0);
    const GPU_OPERATOR_REAL ux = finiteOr(s.momRhoUPx[c], 0.0)/rhoP;
    const GPU_OPERATOR_REAL uy = finiteOr(s.momRhoUPy[c], 0.0)/rhoP;
    const GPU_OPERATOR_REAL uz = finiteOr(s.momRhoUPz[c], 0.0)/rhoP;
    const GPU_OPERATOR_REAL kinetic = 0.5*rhoP*sqr3(ux, uy, uz);
    const GPU_OPERATOR_REAL theta =
        clampMin
        (
            (finiteOr(s.momRhoEP[c], kinetic) - kinetic)/(1.5*rhoP),
            0.0
        );
    GPU_OPERATOR_REAL pColl = 0.0;
    if (s.collisionalPressureEnabled)
    {
        const GPU_OPERATOR_REAL g0 = radialDistributionG0Device(eps);
        pColl = ugkwp::collisionalPressure
        (
            s.collisionalRestitution,
            s.rhoSolid,
            eps,
            g0,
            theta
        );
        pColl = clampRange(finiteOr(pColl, 0.0), 0.0, OfGreat);
    }

    return clampRange(finiteOr(pColl, OfGreat), 0.0, OfGreat);
}
