#pragma once
// Free-particle pressure projection shared by gas and constrained thermal adapters.
// Thermal wall constraints are applied before entering this operation.
template<bool CompactParticles, class Real>
__device__ __forceinline__ void updateMobilePressureParticle
(
    DeviceState& s, const int i,
    const Real ux0, const Real uy0, const Real uz0,
    const Real ux1, const Real uy1, const Real uz1,
    const Real theta1, const Real thermalScale, const Real thetaScale,
    const bool resolved
)
{
    const Real dux = finiteOr((CompactParticles ? s.compactPux[i] : s.pux[i]), ux0) - ux0;
    const Real duy = finiteOr((CompactParticles ? s.compactPuy[i] : s.puy[i]), uy0) - uy0;
    const Real duz = finiteOr((CompactParticles ? s.compactPuz[i] : s.puz[i]), uz0) - uz0;
    (CompactParticles ? s.compactPux[i] : s.pux[i]) = resolved ? ux1 + thermalScale*dux : ux1;
    (CompactParticles ? s.compactPuy[i] : s.puy[i]) = resolved ? uy1 + thermalScale*duy : uy1;
    (CompactParticles ? s.compactPuz[i] : s.puz[i]) = resolved ? uz1 + thermalScale*duz : uz1;
    (CompactParticles ? s.compactPTheta[i] : s.pTheta[i]) = resolved
      ? clampMin(finiteOr((CompactParticles ? s.compactPTheta[i] : s.pTheta[i]), Real(0.0))*thetaScale, Real(0.0))
      : theta1;
}
