#pragma once
// One operator implementation; scalar/time adapters are compile-time only.
__device__ void updateBarthLimiter
(
    const GPU_OPERATOR_REAL centre,
    const GPU_OPERATOR_REAL predicted,
    const GPU_OPERATOR_REAL minimum,
    const GPU_OPERATOR_REAL maximum,
    GPU_OPERATOR_REAL& limiter
)
{
    const GPU_OPERATOR_REAL delta = predicted - centre;
    // Limiting is homogeneous: a trace scalar has the same admissible
    // relative slope as a bulk scalar. Only an exactly zero slope is inactive.
    if (delta > GPU_OPERATOR_R(0.0))
    {
        limiter = fmin(limiter, (maximum - centre)/delta);
    }
    else if (delta < GPU_OPERATOR_R(0.0))
    {
        limiter = fmin(limiter, (minimum - centre)/delta);
    }
}

__device__ void updateVenkatakrishnanLimiter
(
    const GPU_OPERATOR_REAL centre,
    const GPU_OPERATOR_REAL predicted,
    const GPU_OPERATOR_REAL minimum,
    const GPU_OPERATOR_REAL maximum,
    GPU_OPERATOR_REAL& limiter
)
{
    const GPU_OPERATOR_REAL delta = predicted - centre;
    if (delta == GPU_OPERATOR_R(0.0))
    {
        return;
    }
    const GPU_OPERATOR_REAL admissible =
        delta > GPU_OPERATOR_R(0.0) ? maximum - centre : minimum - centre;
    // Test signs without a product, which underflows for trace inventories.
    if ((delta > GPU_OPERATOR_R(0.0) && admissible <= GPU_OPERATOR_R(0.0))
        || (delta < GPU_OPERATOR_R(0.0) && admissible >= GPU_OPERATOR_R(0.0)))
    {
        limiter = GPU_OPERATOR_R(0.0);
        return;
    }



    // Evaluate the same rational function without squaring an unbounded
    // ratio. For a very small slope admissible/delta may itself overflow.
    GPU_OPERATOR_REAL value;
    if (fabs(admissible) >= fabs(delta))
    {
        const GPU_OPERATOR_REAL inverse = delta/admissible;
        value = (GPU_OPERATOR_R(1.0) + GPU_OPERATOR_R(2.0)*inverse)
            /(GPU_OPERATOR_R(1.0) + inverse
                + GPU_OPERATOR_R(2.0)*inverse*inverse);
    }
    else
    {
        const GPU_OPERATOR_REAL ratio = admissible/delta;
        value = (ratio*ratio + GPU_OPERATOR_R(2.0)*ratio)
            /(ratio*ratio + ratio + GPU_OPERATOR_R(2.0));
    }
    limiter = fmin(limiter, value);
}

__device__ void updateConfiguredGasLimiter
(
    const int limiterScheme,
    const GPU_OPERATOR_REAL centre,
    const GPU_OPERATOR_REAL predicted,
    const GPU_OPERATOR_REAL minimum,
    const GPU_OPERATOR_REAL maximum,
    GPU_OPERATOR_REAL& limiter
)
{
    if (limiterScheme == 2)
    {
        updateVenkatakrishnanLimiter
        (
            centre, predicted, minimum, maximum, limiter
        );
    }
    else
    {
        updateBarthLimiter
        (
            centre, predicted, minimum, maximum, limiter
        );
    }
}
