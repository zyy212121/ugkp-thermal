#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ GPU_OPERATOR_REAL normalDevice(unsigned long long& state)
{
    const GPU_OPERATOR_REAL u1 = clampMin(uniform01Device(state), GPU_OPERATOR_R(1.0e-12));
    const GPU_OPERATOR_REAL u2 = uniform01Device(state);
    return sqrt(-GPU_OPERATOR_R(2.0)*log(u1))*cos(GPU_OPERATOR_R(6.28318530717958647692)*u2);
}
__device__ void normalPairDevice(
    unsigned long long& state,
    GPU_OPERATOR_REAL& z0,
    GPU_OPERATOR_REAL& z1
)
{
    const GPU_OPERATOR_REAL u1 = clampMin(uniform01Device(state), GPU_OPERATOR_R(1.0e-12));
    const GPU_OPERATOR_REAL u2 = uniform01Device(state);

    const GPU_OPERATOR_REAL r = sqrt(-GPU_OPERATOR_R(2.0)*log(u1));
    GPU_OPERATOR_REAL sVal = GPU_OPERATOR_R(0.0);
    GPU_OPERATOR_REAL cVal = GPU_OPERATOR_R(0.0);

    sincos(GPU_OPERATOR_R(6.28318530717958647692)*u2, &sVal, &cVal);

    z0 = r*cVal;
    z1 = r*sVal;

}

__device__ GPU_OPERATOR_REAL sampleDiameterAroundDevice
(
    const DeviceState& s,
    const GPU_OPERATOR_REAL dCenter,
    unsigned long long& rng
)
{
    const GPU_OPERATOR_REAL dMin = clampMin(s.particleDiameterMin, GPU_OPERATOR_R(1.0e-12));
    const GPU_OPERATOR_REAL dMax =
        clampMin(s.particleDiameterMax, clampMin(dMin, GPU_OPERATOR_R(1.0e-12)));
    const GPU_OPERATOR_REAL dLocal = clampRange
    (
        finiteOr(dCenter, s.particleDiameterFallback),
        dMin,
        dMax
    );

    if (dMax <= dMin || s.particleDiameterSigma <= OfSmall)
    {
        return dLocal;
    }

    const GPU_OPERATOR_REAL sigma = s.particleDiameterSigma;
    const GPU_OPERATOR_REAL lnMin = log(dMin);
    const GPU_OPERATOR_REAL lnMax = log(dMax);
    const GPU_OPERATOR_REAL lnD =
        log(clampMin(dLocal, GPU_OPERATOR_R(1.0e-12)))
      - GPU_OPERATOR_R(0.5)*sigma*sigma
      + sigma*normalDevice(rng);

    if (!finiteDevice(lnD))
    {
        return lnD < GPU_OPERATOR_R(0.0) ? dMin : dMax;
    }
    if (lnD <= lnMin)
    {
        return dMin;
    }
    if (lnD >= lnMax)
    {
        return dMax;
    }
    return exp(lnD);
}

__device__ GPU_OPERATOR_REAL sampleDiameterFromPoolMomentDevice
(
    const DeviceState& s,
    const GPU_OPERATOR_REAL meanD,
    const GPU_OPERATOR_REAL secondD,
    unsigned long long& rng
)
{
    const GPU_OPERATOR_REAL dMin = clampMin(s.particleDiameterMin, GPU_OPERATOR_R(1.0e-12));
    const GPU_OPERATOR_REAL dMax = clampMin(s.particleDiameterMax, dMin);

    const GPU_OPERATOR_REAL mean =
        clampRange(finiteOr(meanD, s.particleDiameterFallback), dMin, dMax);
    const GPU_OPERATOR_REAL second = clampMin(finiteOr(secondD, mean*mean), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL varD = second - mean*mean;

    if (!finiteDevice(varD) || varD <= GPU_OPERATOR_R(0.0) || dMax <= dMin)
    {
        return mean;
    }

    const GPU_OPERATOR_REAL sampledD = mean + sqrt(varD)*normalDevice(rng);
    if (!finiteDevice(sampledD))
    {
        return mean;
    }

    return clampRange(sampledD, dMin, dMax);
}

__device__ GPU_OPERATOR_REAL radialDistributionG0Device(const GPU_OPERATOR_REAL eps)
{
    GPU_OPERATOR_REAL cRatio = clampRange(eps/(GPU_OPERATOR_R(0.63) + GPU_OPERATOR_R(1.0e-6)), GPU_OPERATOR_R(0.0), GPU_OPERATOR_R(0.99));
    return ugkwp::radialDistributionG0FromRatio(cRatio);
}
