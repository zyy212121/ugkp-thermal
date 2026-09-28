#pragma once
// Shared mechanical closure with native material-temperature closure.
__device__ GPU_RECOVERY_INLINE void solidRecoveryFromParticleMomentsCell(DeviceState& s, const int c)
{
    const GPU_OPERATOR_REAL rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    if (rhoP <= s.epsSMin*s.rhoSolid)
    {
        clearSolidCell(s, c);
        return;
    }

    const GPU_OPERATOR_REAL totalMomX = finiteOr(s.momRhoUPx[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL totalMomY = finiteOr(s.momRhoUPy[c], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL totalMomZ = finiteOr(s.momRhoUPz[c], GPU_OPERATOR_R(0.0));
    GPU_OPERATOR_REAL totalEnergy = clampMin(finiteOr(s.momRhoEP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL totalDiameter = clampMin(finiteOr(s.momRhoPD[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL totalHeat = clampMin(finiteOr(s.momRhoHpP[c], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));

    s.epsS[c] = rhoP/s.rhoSolid;
    s.rhoUsx[c] = totalMomX;
    s.rhoUsy[c] = totalMomY;
    s.rhoUsz[c] = totalMomZ;
    s.rhoDs[c] = totalDiameter;
    s.rhoHp[c] = totalHeat;
    s.Usx[c] = totalMomX/rhoP;
    s.Usy[c] = totalMomY/rhoP;
    s.Usz[c] = totalMomZ/rhoP;

    const GPU_OPERATOR_REAL invRhoP = GPU_OPERATOR_R(1.0)/rhoP;
    const GPU_OPERATOR_REAL kinetic =
        GPU_OPERATOR_R(0.5)*(sqr3(totalMomX*invRhoP, totalMomY*invRhoP, totalMomZ*invRhoP)*rhoP);
    if (totalEnergy < kinetic)
    {
        totalEnergy = kinetic;
    }
    s.rhoEs[c] = totalEnergy;
    s.theta[c] = clampMin((totalEnergy - kinetic)/(GPU_OPERATOR_R(1.5)*rhoP), GPU_OPERATOR_R(0.0));
    s.dMeanCell[c] =
        clampMin(finiteOr(totalDiameter/rhoP, s.particleDiameterFallback), GPU_OPERATOR_R(1.0e-12));

#if GPU_OPERATOR_THERMAL
    if (s.solveParticleTemperature != 0)
    {
        const GPU_OPERATOR_REAL specificEnthalpy =
            finiteOr(totalHeat/(rhoP + OfSmall), -GPU_OPERATOR_R(1.0));
        s.Tp[c] = clampRange
        (
            finiteOr
            (
                particleTemperatureFromSpecificEnthalpyDevice
                (
                    specificEnthalpy
                ),
                s.TpMin
            ),
            s.TpMin,
            s.TpMax
        );
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = GPU_OPERATOR_R(0.0);
    }
#else
    const GPU_OPERATOR_REAL heatFactor = particleHeatFactorDevice(s);
    if (heatFactor > 0.0)
    {
        s.Tp[c] =
            clampRange
            (
                finiteOr(totalHeat/(rhoP*heatFactor), s.TpMin),
                s.TpMin,
                s.TpMax
            );
    }
    else
    {
        s.Tp[c] = s.TpMin;
        s.rhoHp[c] = 0.0;
    }
#endif
}
