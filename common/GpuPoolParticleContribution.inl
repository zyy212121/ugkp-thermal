// Inputs: s, i, theta, arithmetic helpers and GPU_POOL_MASS_FALLBACK.
// GPU_POOL_LOAD_LATE_THETA() supplies the declared theta-read point. Exports m,
// ux/uy/uz, d and specificEnergy to the caller; invalid contributions trap.
// Caller owns RNG selection/commit and accumulation order.
    const GPU_OPERATOR_REAL m = clampMin(finiteOr(s.pm[i], GPU_POOL_MASS_FALLBACK), GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL ux = finiteOr(s.pux[i], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uy = finiteOr(s.puy[i], GPU_OPERATOR_R(0.0));
    const GPU_OPERATOR_REAL uz = finiteOr(s.puz[i], GPU_OPERATOR_R(0.0));
    GPU_POOL_LOAD_LATE_THETA()
    const GPU_OPERATOR_REAL d =
        clampMin
        (
            finiteOr(s.pd[i], s.particleDiameterFallback),
            GPU_OPERATOR_R(1.0e-12)
        );
    const GPU_OPERATOR_REAL specificEnergy =
        GPU_OPERATOR_R(0.5)*sqr3(ux, uy, uz) + GPU_OPERATOR_R(1.5)*theta;

    if
    (
        nonFiniteDevice(m) || m < GPU_OPERATOR_R(0.0)
     || nonFiniteDevice(ux) || nonFiniteDevice(uy)
     || nonFiniteDevice(uz)
     || nonFiniteDevice(theta) || theta < GPU_OPERATOR_R(0.0)
     || nonFiniteDevice(specificEnergy)
    )
    {
        asm("trap;");
    }

