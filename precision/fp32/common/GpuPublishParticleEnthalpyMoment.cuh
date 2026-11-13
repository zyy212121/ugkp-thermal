#pragma once
// Shared physical update; the including backend supplies its state and scalar adapters.
__device__ void publishParticleEnthalpyMoment
(
    DeviceState& s,
    const int c,
    const GpuReal heatDensity
)
{
    const GpuReal rhoP = clampMin(finiteOr(s.momRhoP[c], GPU_R(0.0)), GPU_R(0.0));
    if
    (
        s.solveParticleTemperature == 0
     || rhoP <= s.epsSMin*s.rhoSolid
    )
    {
        s.momRhoHpP[c] = GPU_R(0.0);
        s.rhoHp[c] = GPU_R(0.0);
        s.Tp[c] = s.TpMin;
        return;
    }
    const GpuReal totalHeat = clampMin(finiteOr(heatDensity, GPU_R(0.0)), GPU_R(0.0));
    s.momRhoHpP[c] = totalHeat;
    s.rhoHp[c] = totalHeat;
    const GpuReal specificEnthalpy =
        finiteOr(totalHeat/(rhoP + OfSmall), -GPU_R(1.0));
    s.Tp[c] = clampRange
    (
        finiteOr
        (
            particleTemperatureFromSpecificEnthalpyDevice(specificEnthalpy),
            s.TpMin
        ),
        s.TpMin,
        s.TpMax
    );
}
