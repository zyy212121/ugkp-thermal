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
    if (delta > OfSmall)
    {
        limiter = fmin(limiter, (maximum - centre)/delta);
    }
    else if (delta < -OfSmall)
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
    if (fabs(delta) <= OfSmall)
    {
        return;
    }
    const GPU_OPERATOR_REAL admissible =
        delta > GPU_OPERATOR_R(0.0) ? maximum - centre : minimum - centre;
    if (admissible*delta <= GPU_OPERATOR_R(0.0))
    {
        limiter = GPU_OPERATOR_R(0.0);
        return;
    }

                                                                          
                                                                    
    const GPU_OPERATOR_REAL ratio = fmax(admissible/delta, GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL numerator = ratio*ratio + GPU_OPERATOR_R(2.0)*ratio;
    const GPU_OPERATOR_REAL denominator = ratio*ratio + ratio + GPU_OPERATOR_R(2.0);
    limiter = fmin
    (
        limiter,
        denominator > OfSmall ? numerator/denominator : GPU_OPERATOR_R(0.0)
    );
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
