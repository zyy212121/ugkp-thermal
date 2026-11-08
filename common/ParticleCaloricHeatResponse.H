#pragma once
#include "GpuParticleCaloric.H"
namespace ugkwp {
#if defined(__CUDACC__)
#define UGKP_CALORIC_RESPONSE_HD __host__ __device__
#else
#define UGKP_CALORIC_RESPONSE_HD
#endif
struct ParticleHeatSegment {
    double lower=-1.7976931348623157e308,upper=1.7976931348623157e308,curvature=0.;
};
struct ParticleCaloricHeatResponse {
    ParticleCaloric law;
    double initialTemperature,fraction,minimumTemperature,maximumTemperature;
    // An open interval in gas temperature on which this response is exactly
    // quadratic. Cuts include clipping and every caloric/phase boundary.
    // At a cut, evaluate the original response so its one-sided derivative
    // convention is preserved. The roundoff margin only shrinks the cache.
    UGKP_CALORIC_RESPONSE_HD ParticleHeatSegment segment(double gasTemperature)const noexcept {
        ParticleHeatSegment out;
        if(fraction==0.)return out;
        if(!(fraction>0.)||!std::isfinite(fraction)){
            out.lower=out.upper=gasTemperature;return out;
        }
        const double delta=fraction*(gasTemperature-initialTemperature);
        const bool active=delta>minimumTemperature-initialTemperature
            &&delta<maximumTemperature-initialTemperature;
        const double endpoint=initialTemperature+delta;
        if(active&&endpoint>2327.&&endpoint<4000.
            &&(!law.phaseChange||endpoint>law.liquidus()))
            out.curvature=(0.0031/0.101961276)*fraction*fraction;
        const double cuts[6]={minimumTemperature,maximumTemperature,2327.,4000.,
            law.phaseChange?law.solidus():2327.,law.phaseChange?law.liquidus():4000.};
        for(int k=0;k<6;++k){
            // Caloric cuts cannot affect an already clipped response.
            if(k>=2&&!active)continue;
            const double offset=(cuts[k]-initialTemperature)/fraction;
            const double cut=initialTemperature+offset;
            if(!std::isfinite(cut))continue;
            const double margin=16.*2.2204460492503131e-16
                *(::fabs(initialTemperature)+::fabs(offset)+::fabs(gasTemperature)+1.);
            if(gasTemperature<cut)out.upper=::fmin(out.upper,cut-margin);
            else out.lower=::fmax(out.lower,cut+margin);
        }
        return out;
    }
    // The relaxation fraction is frozen at the source epoch, exactly as in
    // the reference gas particle exchange. Particle bounds are independent
    // of the gas thermo table. Clipping is part of the evaluated equation.
    UGKP_CALORIC_RESPONSE_HD void operator()(double gasTemperature,
        double& specificHeat,double& derivative)const noexcept {
        const double rawDelta=fraction*(gasTemperature-initialTemperature);
        const double lower=minimumTemperature-initialTemperature;
        const double upper=maximumTemperature-initialTemperature;
        const double delta=rawDelta<lower?lower:(rawDelta>upper?upper:rawDelta);
        specificHeat=law.enthalpyIncrement(initialTemperature,delta);
        derivative=rawDelta<lower || rawDelta>upper?0.:
            fraction*law.enthalpyDerivative(initialTemperature+delta);
    }
};
#undef UGKP_CALORIC_RESPONSE_HD
}
