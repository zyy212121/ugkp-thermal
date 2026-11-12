#pragma once
#include "GpuParticleCaloric.H"
namespace ugkwp {
#if defined(__CUDACC__)
#define UGKP_CALORIC_RESPONSE_HD __host__ __device__
#else
#define UGKP_CALORIC_RESPONSE_HD
#endif
struct ParticleHeatSegment {
    float lower=-3.4028234663852886e38f,upper=3.4028234663852886e38f,curvature=0.f;
};
struct ParticleCaloricHeatResponse {
    ParticleCaloric law;
    float initialTemperature,fraction,minimumTemperature,maximumTemperature;
    // An open interval in gas temperature on which this response is exactly
    // quadratic. Cuts include clipping and every caloric/phase boundary.
    // At a cut, evaluate the original response so its one-sided derivative
    // convention is preserved. The roundoff margin only shrinks the cache.
    UGKP_CALORIC_RESPONSE_HD ParticleHeatSegment segment(float gasTemperature)const noexcept {
        ParticleHeatSegment out;
        if(fraction==0.f)return out;
        if(!(fraction>0.f)||!std::isfinite(fraction)){
            out.lower=out.upper=gasTemperature;return out;
        }
        const float delta=fraction*(gasTemperature-initialTemperature);
        const bool active=delta>minimumTemperature-initialTemperature
            &&delta<maximumTemperature-initialTemperature;
        const float endpoint=initialTemperature+delta;
        if(active&&endpoint>2327.f&&endpoint<4000.f
            &&(!law.phaseChange||endpoint>law.liquidus()))
            out.curvature=(0.0031f/0.101961276f)*fraction*fraction;
        const float cuts[6]={minimumTemperature,maximumTemperature,2327.f,4000.f,
            law.phaseChange?law.solidus():2327.f,law.phaseChange?law.liquidus():4000.f};
        for(int k=0;k<6;++k){
            // Caloric cuts cannot affect an already clipped response.
            if(k>=2&&!active)continue;
            const float offset=(cuts[k]-initialTemperature)/fraction;
            const float cut=initialTemperature+offset;
            if(!std::isfinite(cut))continue;
            const float margin=16.f*1.1920928955078125e-7f
                *(std::fabs(initialTemperature)+std::fabs(offset)+std::fabs(gasTemperature)+1.f);
            if(gasTemperature<cut)out.upper=std::fmin(out.upper,cut-margin);
            else out.lower=std::fmax(out.lower,cut+margin);
        }
        return out;
    }
    // The relaxation fraction is frozen at the source epoch, exactly as in
    // the reference gas particle exchange. Particle bounds are independent
    // of the gas thermo table. Clipping is part of the evaluated equation.
    UGKP_CALORIC_RESPONSE_HD void operator()(float gasTemperature,
        float& specificHeat,float& derivative)const noexcept {
        const float rawDelta=fraction*(gasTemperature-initialTemperature);
        const float lower=minimumTemperature-initialTemperature;
        const float upper=maximumTemperature-initialTemperature;
        const float delta=rawDelta<lower?lower:(rawDelta>upper?upper:rawDelta);
        specificHeat=law.enthalpyIncrement(initialTemperature,delta);
        derivative=rawDelta<lower || rawDelta>upper?0.f:
            fraction*law.enthalpyDerivative(initialTemperature+delta);
    }
};
#undef UGKP_CALORIC_RESPONSE_HD
}
