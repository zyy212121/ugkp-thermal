const GPU_MOMENT_REAL m = clampMin(finiteOr(s.pm[i], GPU_MOMENT_R(0.0)), GPU_MOMENT_R(0.0));
const GPU_MOMENT_REAL ux = finiteOr(s.pux[i], GPU_MOMENT_R(0.0));
const GPU_MOMENT_REAL uy = finiteOr(s.puy[i], GPU_MOMENT_R(0.0));
const GPU_MOMENT_REAL uz = finiteOr(s.puz[i], GPU_MOMENT_R(0.0));
#if GPU_MOMENT_THERMAL
const GPU_MOMENT_REAL theta = particleMomentThetaDevice(s, i);
#else
const GPU_MOMENT_REAL theta = clampMin(finiteOr(s.pTheta[i], GPU_MOMENT_R(0.0)), GPU_MOMENT_R(0.0));
#endif
const GPU_MOMENT_REAL d = clampMin(finiteOr(s.pd[i], s.particleDiameterFallback), GPU_MOMENT_R(1.0e-12));
const GPU_MOMENT_REAL particleEnergy = m*(GPU_MOMENT_R(0.5)*sqr3(ux, uy, uz) + GPU_MOMENT_R(1.5)*theta);
GPU_MOMENT_REAL particleHeat = GPU_MOMENT_R(0.0);
#if GPU_MOMENT_THERMAL
const GPU_MOMENT_REAL tp = clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
particleHeat = materialEnthalpyMoment(m, tp);
#else
if (heatFactor > GPU_MOMENT_R(0.0))
    particleHeat = m*heatFactor*clampRange(finiteOr(s.pT[i], s.TpMin), s.TpMin, s.TpMax);
#endif
