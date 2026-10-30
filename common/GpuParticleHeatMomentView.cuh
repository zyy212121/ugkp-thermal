#pragma once
// The gas reference reads published cell moments. Thermal/contact owners can
// supply the mobile subset without changing the common finite-capacity law.
struct ParticleGasHeatMoments
{
    GPU_OPERATOR_REAL mass, px, py, pz, diameter, enthalpy;
};
template<class State>
__device__ ParticleGasHeatMoments particleCellHeatMoments(const State& s,int c)
{return {s.momRhoP[c],s.momRhoUPx[c],s.momRhoUPy[c],s.momRhoUPz[c],s.momRhoPD[c],s.momRhoHpP[c]};}
template<class State>
__device__ auto particleGasHeatMoments(const State& s,int c,int)
    -> decltype(s.particleMobileHeatMoments[0],ParticleGasHeatMoments{})
{
    if(s.particleStuckModelConfigured && s.particleMobileHeatMoments) {
        const auto* m=s.particleMobileHeatMoments+6*c;
        return {m[0],m[1],m[2],m[3],m[4],m[5]};
    }
    return particleCellHeatMoments(s,c);
}
template<class State>
__device__ ParticleGasHeatMoments particleGasHeatMoments(const State& s,int c,long)
{return particleCellHeatMoments(s,c);}
