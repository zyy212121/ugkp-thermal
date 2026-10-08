#pragma once
// The read-only preflight and the writer evaluate this identical expression.
template<class Real> struct PressureParticleCandidate { Real ux,uy,uz,theta; };
template<class Real>
__device__ __forceinline__ PressureParticleCandidate<Real> pressureParticleCandidate
(Real ux,Real uy,Real uz,Real theta,Real ux0,Real uy0,Real uz0,
 Real ux1,Real uy1,Real uz1,Real theta1,Real thermalScale,Real thetaScale,bool resolved)
{
    return {resolved?ux1+thermalScale*(ux-ux0):ux1,
            resolved?uy1+thermalScale*(uy-uy0):uy1,
            resolved?uz1+thermalScale*(uz-uz0):uz1,
            resolved?theta*thetaScale:theta1};
}
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
    const auto q=pressureParticleCandidate
    ((CompactParticles?s.compactPux[i]:s.pux[i]),
     (CompactParticles?s.compactPuy[i]:s.puy[i]),
     (CompactParticles?s.compactPuz[i]:s.puz[i]),
     (CompactParticles?s.compactPTheta[i]:s.pTheta[i]),
     ux0,uy0,uz0,ux1,uy1,uz1,theta1,thermalScale,thetaScale,resolved);
    (CompactParticles?s.compactPux[i]:s.pux[i])=q.ux;
    (CompactParticles?s.compactPuy[i]:s.puy[i])=q.uy;
    (CompactParticles?s.compactPuz[i]:s.puz[i])=q.uz;
    (CompactParticles?s.compactPTheta[i]:s.pTheta[i])=q.theta;
}
