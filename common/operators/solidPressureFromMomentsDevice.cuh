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
        return GPU_OPERATOR_R(0.0);
    }

    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        return GPU_OPERATOR_R(0.0);
    }

    const GPU_OPERATOR_REAL eps =
        clampRange(rhoP/clampMin(s.rhoSolid, GPU_OPERATOR_TINY(1.0e-300)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    const GPU_OPERATOR_REAL ux = finiteOr(s.momRhoUPx[c], GPU_OPERATOR_R(0.0))/rhoP;
    const GPU_OPERATOR_REAL uy = finiteOr(s.momRhoUPy[c], GPU_OPERATOR_R(0.0))/rhoP;
    const GPU_OPERATOR_REAL uz = finiteOr(s.momRhoUPz[c], GPU_OPERATOR_R(0.0))/rhoP;
    const GPU_OPERATOR_REAL kinetic = GPU_OPERATOR_R(0.5)*rhoP*sqr3(ux, uy, uz);
    const GPU_OPERATOR_REAL theta =
        clampMin
        (
            (finiteOr(s.momRhoEP[c], kinetic) - kinetic)/(GPU_OPERATOR_R(1.5)*rhoP),
            GPU_OPERATOR_R(0.0)
        );
    GPU_OPERATOR_REAL pColl = GPU_OPERATOR_R(0.0);
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
        if (!finiteDevice(pColl)) return pColl;
        pColl = clampRange(pColl, GPU_OPERATOR_R(0.0), OfGreat);
    }

    return clampRange(finiteOr(pColl, OfGreat), GPU_OPERATOR_R(0.0), OfGreat);
}
