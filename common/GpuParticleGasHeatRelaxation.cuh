#pragma once
__device__ __forceinline__ GPU_OPERATOR_REAL particleTemperatureAfterGasRelaxation
(const DeviceState& s, const GPU_OPERATOR_REAL oldTemperature,
 const GPU_OPERATOR_REAL gasTemperature, const GPU_OPERATOR_REAL reynolds,
 const GPU_OPERATOR_REAL diameter, const GPU_OPERATOR_REAL volumetricHeatCapacity,
 const GPU_OPERATOR_TIME dt)
{
    const GPU_OPERATOR_REAL conductivity = molecularGasConductivity(s);
    const GPU_OPERATOR_REAL nu = ugkwp::ranzMarshallNuFromPrOneThird
        (clampMin(reynolds, GPU_OPERATOR_R(0.0)), s.gasPrOneThird);
    const GPU_OPERATOR_REAL rate = GPU_OPERATOR_R(6.0)*nu*conductivity
        /(volumetricHeatCapacity*diameter*diameter + GPU_OPERATOR_TINY(1.0e-300));
    const GPU_OPERATOR_REAL decay = exp(-clampMin(rate, GPU_OPERATOR_R(0.0))*dt);
    return clampRange(gasTemperature + (oldTemperature-gasTemperature)*decay, s.TpMin, s.TpMax);
}
__device__ __forceinline__ void decayParticleUnresolvedTheta(DeviceState& s, const int i, const int c)
{
    const GPU_OPERATOR_REAL alpha = clampRange(finiteOr(s.thetaDragAlpha[c], GPU_OPERATOR_R(1.0)),
        GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(1.0));
    const GPU_OPERATOR_REAL theta = clampMin(finiteOr(s.pTheta[i], GPU_OPERATOR_R(0.0)), GPU_OPERATOR_R(0.0));
    s.pTheta[i] = theta*alpha;
}
